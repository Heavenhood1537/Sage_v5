from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.config import AppConfig


@dataclass
class ResearchService:
    """Detached research-sidecar launcher for Sage v5."""

    cfg: AppConfig

    def __post_init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]

    def launch_detached(self, query: str) -> bool:
        text = str(query or "").strip()
        if not text:
            return False
        research_enabled = bool(getattr(self.cfg.sidecars, "research_enabled", True))
        if not research_enabled:
            return False

        python_scripts_dir = self.project_root / ".venv" / "Scripts"
        pythonw_exe = python_scripts_dir / "pythonw.exe"
        python_exe = pythonw_exe if pythonw_exe.exists() else (python_scripts_dir / "python.exe")
        sidecar_file = self.project_root / "research_sidecar.py"
        if not python_exe.exists() or not sidecar_file.exists():
            return False

        report = self.project_root / "data" / "research" / "RESEARCH_REPORT.md"
        status = self.project_root / "data" / "research" / "research_status.json"
        report.parent.mkdir(parents=True, exist_ok=True)

        flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if hasattr(subprocess, "DETACHED_PROCESS"):
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        if hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
            flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

        max_results_cfg = int(getattr(self.cfg.sidecars, "research_max_results", 8) or 8)
        eco_mode = os.environ.get("SAGE5_RESEARCH_ECO", "1").strip().lower() not in {"0", "false", "off"}
        if eco_mode:
            max_results_cfg = min(max_results_cfg, 6)
        cmd = [
            str(python_exe),
            str(sidecar_file),
            "--query",
            text,
            "--workspace",
            str(self.project_root),
            "--report",
            str(report),
            "--status",
            str(status),
            "--max-results",
            str(max(3, max_results_cfg)),
        ]

        subprocess.Popen(
            cmd,
            cwd=str(self.project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
        return True


__all__ = ["ResearchService"]
