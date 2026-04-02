from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess


def _prompt_with_default(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value or default


def build_settings_yaml(llama_url: str, notes_dir: str) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return (
        "# Sage v5 first-run settings\n"
        f"# generated_at: {stamp}\n\n"
        "llama_server:\n"
        f"  url: {llama_url}\n\n"
        "paths:\n"
        f"  notes_dir: {notes_dir}\n"
    )


def check_tesseract() -> tuple[bool, str]:
    exe = shutil.which("tesseract")
    if not exe:
        return False, "Tesseract executable not found in PATH"

    try:
        proc = subprocess.run(
            [exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
            check=False,
        )
        first_line = (proc.stdout or "").splitlines()[0].strip() if proc.stdout else "installed"
        return True, f"{first_line} ({exe})"
    except Exception as exc:
        return False, f"Found at {exe}, but version check failed: {exc}"


def main() -> int:
    root = Path(__file__).resolve().parent
    config_dir = root / "config"
    settings_path = config_dir / "settings.yaml"

    print("Sage v5 first-run setup")
    print(f"Project root: {root.as_posix()}")

    if settings_path.exists():
        print(f"[INFO] settings file already exists: {settings_path.as_posix()}")
    else:
        config_dir.mkdir(parents=True, exist_ok=True)

        llama_url = _prompt_with_default(
            "Enter local model endpoint URL",
            "http://127.0.0.1:11434",
        )
        notes_dir = _prompt_with_default(
            "Enter notes directory (relative or absolute)",
            "data/notes",
        )

        settings_text = build_settings_yaml(llama_url=llama_url, notes_dir=notes_dir)
        settings_path.write_text(settings_text, encoding="utf-8")
        print(f"[SUCCESS] Created settings template: {settings_path.as_posix()}")

    ok, detail = check_tesseract()
    if ok:
        print(f"[SUCCESS] Tesseract check: {detail}")
    else:
        print(f"[WARNING] Tesseract check: {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
