from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from core.config import AppConfig, load_config
from services.llm_provider import LlmProvider
from services.ocr_rapid import OcrResult, OcrService
from services.tts_kokoro import TtsService


Lane = Literal["sage_local", "gemma"]
Mode = Literal["single", "conversation"]


@dataclass
class SessionHistoryLogger:
    file_path: Path
    _fh: object | None = None

    def __post_init__(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered handle ensures every turn is flushed immediately.
        self._fh = self.file_path.open("a", encoding="utf-8", buffering=1)

    def append(self, role: str, text: str, mode: Mode, lane: Lane) -> None:
        row = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "role": str(role or "").strip(),
            "mode": mode,
            "lane": lane,
            "text": str(text or ""),
        }
        if self._fh is not None:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()

    def tail(self, lines: int = 15) -> list[str]:
        total = max(1, int(lines or 1))
        if not self.file_path.exists():
            return []
        raw = self.file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return raw[-total:]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


@dataclass
class ShellState:
    mode: Mode
    lane: Lane


def _lane_label(lane: Lane) -> str:
    if lane == "sage_local":
        return "SAGE (LOCAL)"
    return "GEMMA (LOCAL)"


def _status_line(state: ShellState) -> str:
    mode_label = "CONVERSATION" if state.mode == "conversation" else "SINGLE-TURN"
    return f"[MODE: {mode_label}] [LANE: {_lane_label(state.lane)}]"


def _print_help() -> None:
    print("Commands:")
    print("  s <text>   | s: <text>  -> S button equivalent (single-turn Sage local)")
    print("  g <text>   | g: <text>  -> G button equivalent (single-turn Gemma local)")
    print("  cs [text]               -> CS equivalent (conversation mode on Sage)")
    print("  cg [text]               -> CG equivalent (conversation mode on Gemma)")
    print("  aliases: b/w/cb/cw still work")
    print("  single                  -> switch to single-turn mode")
    print("  lane sage|gemma         -> change active lane (legacy alias still works)")
    print("  ocr <filename>           -> OCR image, then Sage summarizes it")
    print("  ocr inbox [count]        -> OCR newest inbox image(s), summarize via Sage")
    print("  save <filename>          -> save last assistant response to data/notes/<filename>.txt")
    print("  saveocr [filename]       -> save last OCR summary (auto-named by date+image if omitted)")
    print("  history | h             -> show last 15 session log lines")
    print("  status                  -> print status bar line")
    print("  help                    -> show this help")
    print("  exit                    -> quit shell")


def _parse_prefixed(line: str, prefix: str) -> str | None:
    raw = str(line or "").strip()
    if not raw:
        return None
    low = raw.lower()
    p = prefix.lower()
    if low == p:
        return ""
    if low.startswith(p + " "):
        return raw[len(prefix) + 1 :].strip()
    if low.startswith(p + ":"):
        return raw[len(prefix) + 1 :].strip()
    return None


def _resolve_memory_file(cfg: AppConfig) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / cfg.paths.memory_dir / "session_history.jsonl"


def _resolve_notes_dir(cfg: AppConfig) -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "notes"


def _normalize_note_filename(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("Filename is required")

    safe = "".join(ch for ch in raw if ch.isalnum() or ch in ("-", "_", "."))
    safe = safe.strip(" .")
    if not safe:
        raise ValueError("Filename has no valid characters")

    stem = Path(safe).stem or "note"
    return f"{stem}.txt"


def _safe_stem(text: str) -> str:
    raw = str(text or "").strip().lower()
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in raw)
    safe = safe.strip("_")
    return safe or "ocr"


def _auto_ocr_filename(image_path: str | None) -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    stem = _safe_stem(Path(str(image_path or "ocr")).stem)
    return f"{stamp}_{stem}.txt"


def _save_assistant_note(cfg: AppConfig, filename: str, text: str) -> Path:
    notes_dir = _resolve_notes_dir(cfg)
    notes_dir.mkdir(parents=True, exist_ok=True)

    target = notes_dir / _normalize_note_filename(filename)
    stamp = datetime.now().isoformat(timespec="seconds")
    payload = (text or "").strip() or "[empty]"
    block = f"\n[{stamp}]\n{payload}\n"
    target.open("a", encoding="utf-8").write(block)
    return target


def _build_ocr_summary_prompt(ocr: OcrResult) -> str:
    body = (ocr.text or "").strip()
    return (
        "You are Sage. Read this OCR extract and provide a concise practical summary. "
        "Include key facts, likely intent, and any obvious OCR uncertainty.\n\n"
        f"Source image: {ocr.image_path}\n"
        f"OCR confidence: {ocr.confidence:.2f}\n"
        "Raw OCR text:\n"
        f"{body}"
    )


def _spawn_tts_task(coro: asyncio.Future | asyncio.coroutines) -> None:
    task = asyncio.create_task(coro)

    def _done(t: asyncio.Task) -> None:
        try:
            _ = t.exception()
        except Exception:
            pass

    task.add_done_callback(_done)


async def _stream_reply(provider: LlmProvider, lane: Lane, user_text: str, tts: TtsService | None = None) -> str:
    parts: list[str] = []
    speak_buffer = ""
    print("sage > ", end="", flush=True)
    async for token in provider.chat(user_text=user_text, target=lane, stream=True):
        if not token:
            continue
        print(token, end="", flush=True)
        parts.append(token)
        speak_buffer += token

        # Start speaking completed sentences while remaining tokens continue streaming.
        if tts is not None and any(p in speak_buffer for p in (".", "!", "?", "。", "！", "？")):
            split = []
            current = ""
            for ch in speak_buffer:
                current += ch
                if ch in ".!?。！？":
                    split.append(current.strip())
                    current = ""
            if split:
                speak_buffer = current
                for sentence in split:
                    if sentence:
                            _spawn_tts_task(tts.speak_chunked(sentence, target=lane))

    if tts is not None and speak_buffer.strip():
        _spawn_tts_task(tts.speak_chunked(speak_buffer.strip(), target=lane))
    print()
    return "".join(parts).strip()


async def run_shell() -> None:
    cfg = load_config()
    llm = LlmProvider(cfg)
    tts = TtsService(cfg)
    ocr = OcrService(cfg)

    initial_lane = "sage_local" if cfg.models.active in {"sage_local", "bitnet"} else "gemma"
    state = ShellState(mode="single", lane=initial_lane)
    history = SessionHistoryLogger(_resolve_memory_file(cfg))
    last_assistant_response = ""
    last_ocr_assistant_response = ""
    last_ocr_image_path = ""

    print("Sage v5 shell")
    print("type 'help' for commands")
    print(_status_line(state))

    try:
        while True:
            user = (await asyncio.to_thread(input, "you > ")).strip()
            if not user or user.lower() in {"exit", "quit"}:
                break

            cmd = user.strip()
            cmd_low = cmd.lower()

            if cmd_low in {"help", "?"}:
                _print_help()
                continue

            if cmd_low in {"history", "h"}:
                print("[HISTORY] last 15 lines")
                rows = history.tail(lines=15)
                if not rows:
                    print("(empty)")
                else:
                    for row in rows:
                        print(row)
                continue

            if cmd_low == "status":
                print(_status_line(state))
                continue

            if cmd_low == "single":
                state.mode = "single"
                print(_status_line(state))
                continue

            save_target = _parse_prefixed(cmd, "save")
            if save_target is not None:
                if not save_target:
                    print("Usage: save <filename>")
                    continue
                if not last_assistant_response.strip():
                    print("[SAVE] No assistant response to save yet.")
                    continue
                try:
                    path = _save_assistant_note(cfg, save_target, last_assistant_response)
                except Exception as exc:
                    print(f"[SAVE ERROR] {exc}")
                    continue
                print(f"[SAVE] Appended to: {path.as_posix()}")
                continue

            save_ocr_target = _parse_prefixed(cmd, "saveocr")
            if save_ocr_target is not None:
                if not last_ocr_assistant_response.strip():
                    print("[SAVEOCR] No OCR summary to save yet.")
                    continue
                filename = save_ocr_target.strip() if save_ocr_target.strip() else _auto_ocr_filename(last_ocr_image_path)
                try:
                    path = _save_assistant_note(cfg, filename, last_ocr_assistant_response)
                except Exception as exc:
                    print(f"[SAVEOCR ERROR] {exc}")
                    continue
                print(f"[SAVEOCR] Appended to: {path.as_posix()}")
                continue

            if cmd_low.startswith("lane "):
                lane_arg = cmd_low.split(" ", 1)[1].strip()
                if lane_arg in {"sage_local", "sage", "bitnet", "local", "bitney"}:
                    state.lane = "sage_local"
                elif lane_arg in {"gemma", "local-gemma", "g"}:
                    state.lane = "gemma"
                else:
                    print("[WARNING] lane must be sage or gemma")
                    continue
                print(_status_line(state))
                continue

            if cmd_low.startswith("ocr "):
                ocr_arg = cmd[4:].strip()
                if not ocr_arg:
                    print("Usage: ocr <filename> | ocr inbox [count]")
                    continue

                if ocr_arg.lower().startswith("inbox"):
                    parts = ocr_arg.split()
                    count = 1
                    if len(parts) >= 2:
                        try:
                            count = max(1, int(parts[1]))
                        except Exception:
                            print("[WARNING] Invalid count. Using 1.")
                            count = 1

                    candidates = ocr.scan_inbox_for_new(limit=count)
                    if not candidates:
                        print("[OCR] No new images in inbox.")
                        continue

                    for image_path in candidates:
                        try:
                            result = await asyncio.to_thread(ocr.extract_text, str(image_path))
                        except Exception as exc:
                            print(f"[OCR ERROR] {exc}")
                            continue

                        print(f"[OCR] file: {result.image_path}")
                        print(f"[OCR] confidence: {result.confidence:.2f}")
                        print("[OCR] raw text:")
                        print(result.text or "(empty)")

                        prompt = _build_ocr_summary_prompt(result)
                        history.append("user", f"ocr {image_path.name}", mode="single", lane="sage_local")
                        assistant = await _stream_reply(llm, "sage_local", prompt, tts=tts)
                        last_assistant_response = assistant
                        last_ocr_assistant_response = assistant
                        last_ocr_image_path = result.image_path
                        history.append("assistant", assistant or "[empty]", mode="single", lane="sage_local")
                    continue

                try:
                    result = await asyncio.to_thread(ocr.extract_text, ocr_arg)
                except Exception as exc:
                    print(f"[OCR ERROR] {exc}")
                    continue

                print(f"[OCR] file: {result.image_path}")
                print(f"[OCR] confidence: {result.confidence:.2f}")
                print("[OCR] raw text:")
                print(result.text or "(empty)")

                prompt = _build_ocr_summary_prompt(result)
                history.append("user", f"ocr {ocr_arg}", mode="single", lane="sage_local")
                assistant = await _stream_reply(llm, "sage_local", prompt, tts=tts)
                last_assistant_response = assistant
                last_ocr_assistant_response = assistant
                last_ocr_image_path = result.image_path
                history.append("assistant", assistant or "[empty]", mode="single", lane="sage_local")
                continue

            # S / G single-turn mappings (with backward-compatible aliases).
            s_text = _parse_prefixed(cmd, "s")
            if s_text is None:
                s_text = _parse_prefixed(cmd, "b")
            if s_text is not None:
                if not s_text:
                    print("Usage: s <text>")
                    continue
                history.append("user", s_text, mode="single", lane="sage_local")
                assistant = await _stream_reply(llm, "sage_local", s_text, tts=tts)
                last_assistant_response = assistant
                history.append("assistant", assistant or "[empty]", mode="single", lane="sage_local")
                continue

            g_text = _parse_prefixed(cmd, "g")
            if g_text is None:
                g_text = _parse_prefixed(cmd, "w")
            if g_text is not None:
                if not g_text:
                    print("Usage: g <text>")
                    continue
                history.append("user", g_text, mode="single", lane="gemma")
                assistant = await _stream_reply(llm, "gemma", g_text, tts=tts)
                last_assistant_response = assistant
                history.append("assistant", assistant or "[empty]", mode="single", lane="gemma")
                continue

            # CS / CG conversation mappings (with backward-compatible aliases).
            cs_text = _parse_prefixed(cmd, "cs")
            if cs_text is None:
                cs_text = _parse_prefixed(cmd, "cb")
            if cs_text is not None:
                state.mode = "conversation"
                state.lane = "sage_local"
                print(_status_line(state))
                if cs_text:
                    history.append("user", cs_text, mode=state.mode, lane=state.lane)
                    assistant = await _stream_reply(llm, state.lane, cs_text, tts=tts)
                    last_assistant_response = assistant
                    history.append("assistant", assistant or "[empty]", mode=state.mode, lane=state.lane)
                continue

            cg_text = _parse_prefixed(cmd, "cg")
            if cg_text is None:
                cg_text = _parse_prefixed(cmd, "cw")
            if cg_text is not None:
                state.mode = "conversation"
                state.lane = "gemma"
                print(_status_line(state))
                if cg_text:
                    history.append("user", cg_text, mode=state.mode, lane=state.lane)
                    assistant = await _stream_reply(llm, state.lane, cg_text, tts=tts)
                    last_assistant_response = assistant
                    history.append("assistant", assistant or "[empty]", mode=state.mode, lane=state.lane)
                continue

            # Default behavior.
            lane = state.lane

            history.append("user", user, mode=state.mode, lane=lane)
            reply = await _stream_reply(llm, lane, user, tts=tts)
            last_assistant_response = reply
            history.append("assistant", reply or "[empty]", mode=state.mode, lane=lane)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Shutting down gracefully...")
    finally:
        history.close()

    print("bye")
