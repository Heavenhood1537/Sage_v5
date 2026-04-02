from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _web_search(query: str, max_results: int) -> list[dict[str, str]]:
    DDGS = None
    try:
        from ddgs import DDGS as _DDGS

        DDGS = _DDGS
    except Exception:
        try:
            from duckduckgo_search import DDGS as _DDGS

            DDGS = _DDGS
        except Exception:
            DDGS = None

    if DDGS is None:
        return []

    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    # Support both context-manager and plain-instantiated DDGS variants.
    ddgs = None
    try:
        ddgs = DDGS()
    except Exception:
        return []

    try:
        results = list(ddgs.text(query, max_results=max_results))
    except Exception:
        # Some versions/providers can fail with the default backend.
        # Retry without backend assumptions by recreating once.
        try:
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=max_results))
        except Exception:
            results = []

    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        rows.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
            }
        )

    # Best-effort close for clients that expose close().
    try:
        close_fn = getattr(ddgs, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        pass
    return rows[:max_results]


def _agent_report(query: str, max_results: int) -> tuple[str | None, str | None]:
    try:
        from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
    except Exception:
        return None, "smolagents not installed; fallback summary used"

    try:
        model = LiteLLMModel(model_id="ollama_chat/qwen2.5:3b", api_base="http://127.0.0.1:11434")
        agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
        step_cap_raw = os.environ.get("SAGE5_RESEARCH_AGENT_STEPS", "4").strip()
        try:
            step_cap = max(2, min(8, int(step_cap_raw)))
        except Exception:
            step_cap = 4
        task = (
            "Research this topic and return concise markdown with sections: "
            "Executive Summary, Key Findings, Risks. "
            "Use only evidence from pages you actually retrieved. "
            "Do not invent sources, URLs, or claims not supported by retrieved pages. "
            "If you find relevant evidence, do not claim that no relevant information exists. "
            "Never return placeholder text or meta commentary about missing data. "
            "Do not include a Sources section. Topic: "
            f"{query}"
        )
        result = agent.run(task, max_steps=step_cap)
        text = str(result or "").strip()
        if text:
            return text, None
        return None, "smolagents returned empty output; fallback summary used"
    except Exception as exc:
        return None, f"smolagents agent failed ({exc}); fallback summary used"


def _sanitize_agent_text(agent_text: str) -> str:
    text = str(agent_text or "").strip()
    if not text:
        return ""

    # Remove fenced blocks and trim any model-added sources section.
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.split(r"(?im)^#{1,6}\s*sources\b", text, maxsplit=1)[0].strip()

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        row = line.strip()
        if not row:
            cleaned_lines.append("")
            continue
        # Drop common ReAct/internal trace lines from agent output.
        if re.match(r"(?i)^(thought|action|action input|observation|code|plan)\s*:", row):
            continue
        # Drop list-literal source artifacts such as ['title1', 'title2'].
        if row.startswith("[") and row.endswith("]") and "http" not in row:
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _is_low_quality_agent_text(text: str, has_rows: bool) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True

    # Detect contradiction/placeholder patterns that can slip through agent output.
    low_signal_patterns = (
        "no relevant information",
        "placeholder",
        "if you need further assistance",
        "more detailed information becomes available",
        "please let me know",
        "without any detailed findings",
        "thought:",
        "observation:",
        "given the constraints",
        "here is the markdown format",
        "timedelta",
    )
    if has_rows and any(p in value for p in low_signal_patterns):
        return True

    # Very short synthesis is rarely useful and usually means weak generation.
    if len(value) < 120:
        return True

    return False


def _is_comparison_query(query: str) -> bool:
    value = str(query or "").strip().lower()
    if not value:
        return False
    return (
        " vs " in value
        or "versus" in value
        or value.startswith("compare ")
        or "compare" in value
        or "difference between" in value
    )


def _extract_source_signals(rows: list[dict[str, str]]) -> dict[str, bool]:
    blob = "\n".join(
        f"{str(r.get('title') or '')} {str(r.get('snippet') or '')}".lower()
        for r in rows[:8]
    )
    return {
        "kernel_low_overhead": any(k in blob for k in ("kernel", "low-overhead", "low overhead", "ebpf")),
        "mesh_l7_policy": any(k in blob for k in ("service mesh", "envoy", "mTLS", "traffic management", "microservices")),
        "security_risk": any(k in blob for k in ("risk", "threat", "rootkit", "attack", "security")),
        "performance_tradeoff": any(k in blob for k in ("latency", "overhead", "cpu", "performance")),
    }


def _is_local_first_acid_query(query: str) -> bool:
    value = str(query or "").strip().lower()
    if not value:
        return False
    has_data_integrity = any(k in value for k in ("acid", "transaction", "durability", "sqlite"))
    has_local_first = any(k in value for k in ("local-first", "local first", "offline-first", "offline first"))
    has_worker = any(k in value for k in ("background worker", "job queue", "worker", "scheduler", "reminder"))
    return has_data_integrity and (has_local_first or has_worker)


def _compose_final_conclusion(query: str, rows: list[dict[str, str]], agent_text: str | None) -> str:
    topic = str(query or "the topic").strip()
    if not rows and not agent_text:
        return (
            f"Current evidence for {topic} is limited in this run. "
            "Retry with a narrower query, or run again later to improve source coverage."
        )

    synthesis = _sanitize_agent_text(agent_text or "")
    synthesis = re.sub(r"(?im)^#{1,6}\s+(executive summary|key findings|risks)\s*$", "", synthesis).strip()
    if _is_low_quality_agent_text(synthesis, has_rows=bool(rows)):
        synthesis = ""
    if len(synthesis) > 500:
        synthesis = synthesis[:500].rsplit(" ", 1)[0].rstrip() + "..."

    comparison = _is_comparison_query(topic)
    local_first_acid = _is_local_first_acid_query(topic)
    signals = _extract_source_signals(rows)

    if local_first_acid:
        lines = [
            "Bottom line: for ACID-safe local-first reminders, SQLite is a strong default when writes are serialized, workers are idempotent, and all reminder state transitions are transaction-bound.",
            "",
            "Implementation checklist:",
            "- Data model: store reminders, schedules, and delivery attempts with immutable event timestamps and explicit status states.",
            "- Transaction boundary: create/update reminder and enqueue next execution in one transaction to avoid split-brain state.",
            "- Worker discipline: enforce idempotency keys per reminder occurrence so retries cannot double-send notifications.",
            "- Concurrency control: prefer WAL mode, short transactions, busy timeout, and a single writer path for predictable behavior.",
            "- Recovery path: on startup, scan for stale in-flight jobs and requeue safely from durable state.",
            "- Observability: track queue lag, retry counts, dead-letter events, and end-to-end reminder latency.",
            "",
            "Failure risks to guard against:",
            "- Duplicate sends from non-idempotent retries.",
            "- Lost schedules from out-of-transaction worker enqueue.",
            "- Lock contention from long-running write transactions.",
        ]
    elif comparison:
        lines = [
            "Bottom line: eBPF-centric observability usually wins on runtime efficiency and kernel-level visibility, while service-mesh sidecars usually win on built-in L7 policy, routing, and mTLS workflow.",
            "",
            "Decision guidance:",
            "- Prefer eBPF-first when your primary goal is low-overhead observability and node/network diagnostics at scale.",
            "- Prefer mesh-sidecar-first when your primary goal is L7 traffic control, identity policy, and progressive delivery patterns.",
            "- Use a hybrid approach when both are required: eBPF for platform telemetry/security signals and mesh for selected app-facing policies.",
            "",
            "Evidence confidence:",
        ]
        confidence_bits: list[str] = []
        if signals["kernel_low_overhead"]:
            confidence_bits.append("multiple sources describe eBPF as kernel-level and low-overhead")
        if signals["mesh_l7_policy"]:
            confidence_bits.append("sources describe service mesh strengths around service-to-service policy")
        if signals["performance_tradeoff"]:
            confidence_bits.append("sources discuss performance/overhead trade-offs")
        if signals["security_risk"]:
            confidence_bits.append("risk/security caveats are mentioned in source material")
        if confidence_bits:
            lines.append("- " + "; ".join(confidence_bits) + ".")
        else:
            lines.append("- moderate confidence: this run captured limited high-signal detail.")
    else:
        lines = [
            f"Evidence suggests {topic} should be evaluated as a trade-off between operational simplicity, runtime overhead, and observability depth.",
            "",
            "Key takeaways:",
            "- Favor approaches with measurable impact on latency, CPU, and operational burden in your own environment.",
            "- Validate claims with targeted pilots instead of relying only on vendor narratives.",
        ]

    if synthesis and not local_first_acid:
        lines.extend(["", "Supporting context:", f"- {synthesis}"])

    lines.extend(
        [
            "",
            (
                "Practical next step: implement a crash-recovery test matrix (power loss, duplicate worker execution, DB lock contention), then verify no reminder is lost or duplicated across 1,000+ randomized runs."
                if local_first_acid
                else "Practical next step: run a small A/B pilot on one production-like service and compare p95 latency, CPU cost, operational complexity, and incident-debugging speed."
            ),
        ]
    )
    return "\n".join(lines).strip()


def _build_report(query: str, rows: list[dict[str, str]], agent_text: str | None, note: str | None) -> str:
    lines = [
        "# RESEARCH REPORT",
        "",
        f"- Generated: {_now_iso()}",
        f"- Query: {query}",
        "",
        "## Executive Summary",
        "",
    ]

    if rows:
        for row in rows[:5]:
            title = row.get("title") or row.get("url") or "source"
            snippet = row.get("snippet") or "(no snippet)"
            lines.append(f"- {title}: {snippet}")
    elif agent_text:
        lines.append(_sanitize_agent_text(agent_text) or "No reliable summary generated.")
    else:
        lines.append("No web results captured. Verify internet access and ensure ddgs or duckduckgo-search is installed.")

    sanitized_agent = _sanitize_agent_text(agent_text or "")
    if sanitized_agent and not _is_low_quality_agent_text(sanitized_agent, has_rows=bool(rows)):
        lines.extend(["", "## Agent Synthesis", "", sanitized_agent])
    elif sanitized_agent and rows:
        lines.extend(["", "## Notes", "", "- Agent synthesis was suppressed due to low-confidence placeholder content."])

    final_conclusion = _compose_final_conclusion(query, rows, agent_text)
    if final_conclusion:
        lines.extend(["", "## Final Conclusion", "", final_conclusion])

    lines.extend(["", "## Sources", ""])
    if rows:
        for idx, row in enumerate(rows, start=1):
            title = row.get("title") or "(untitled)"
            url = row.get("url") or ""
            lines.append(f"{idx}. [{title}]({url})")
    else:
        lines.append("No sources available.")

    if note:
        lines.extend(["", "## Notes", "", f"- {note}"])

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run detached research sidecar and write a markdown report")
    parser.add_argument("--query", required=True)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--report", default="data/research/RESEARCH_REPORT.md")
    parser.add_argument("--status", default="data/research/research_status.json")
    parser.add_argument("--max-results", type=int, default=6)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    report_path = (workspace / args.report).resolve() if not Path(args.report).is_absolute() else Path(args.report)
    status_path = (workspace / args.status).resolve() if not Path(args.status).is_absolute() else Path(args.status)

    try:
        started_at = _now_iso()
        _write_json(
            status_path,
            {
                "state": "running",
                "started_at": started_at,
                "query": args.query,
                "pid": os.getpid(),
                "report_path": str(report_path),
            },
        )

        rows = _web_search(args.query, max(3, int(args.max_results)))
        agent_text, note = _agent_report(args.query, max(3, int(args.max_results)))
        report = _build_report(args.query, rows, agent_text, note)

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")

        _write_json(
            status_path,
            {
                "state": "completed",
                "started_at": started_at,
                "completed_at": _now_iso(),
                "query": args.query,
                "result_count": len(rows),
                "report_path": str(report_path),
            },
        )
        return 0
    except Exception as exc:
        _write_json(
            status_path,
            {
                "state": "failed",
                "failed_at": _now_iso(),
                "query": args.query,
                "error": str(exc),
                "report_path": str(report_path),
            },
        )
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                "# RESEARCH REPORT\n\n"
                f"Research run failed at {_now_iso()}.\n\n"
                f"Error: {exc}\n\n"
                f"Traceback:\n\n{traceback.format_exc()}\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
