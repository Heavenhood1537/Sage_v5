# Sage v5: Local AI Personal Assistant

Sage v5 is a local-first assistant designed for private, high-trust personal workflows.

## The Philosophy

In an age of data-mining, Sage v5 is built as a Digital Vault.

Core AI processing runs locally via Ollama using qwen2.5:3b for the Sage lane, so private data can stay on your own machine instead of being pushed to third-party services by default.

## Engineering Standard

This project follows a high-integrity engineering standard:

- no shortcuts
- clear behavior and traceability
- practical transparency in setup, routing, and runtime state

## Setup Guide

### Requirements

- A running Ollama endpoint (default expected URL: `http://127.0.0.1:11434`)
- Python virtual environment and dependencies from `requirements.txt`

### Quick Start

```powershell
cd <path-to-sage-v5>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m interface.gui
```

### First-Run Configuration

Use the Dashboard button `Run First Setup` in the GUI to generate `config/settings.yaml` if needed.

The first-run flow will:

- prompt for local model endpoint URL (default `http://127.0.0.1:11434`)
- prompt for notes directory
- run a Tesseract check and report Success/Warning

You can also run setup from terminal:

```powershell
.\.venv\Scripts\python.exe first_run.py
```

## Sidecar Manual

This section defines how Sage v5 sidecars are expected to run and how to verify each one quickly.

### Sidecar Inventory

- Engine sidecar: local Ollama service on `127.0.0.1:11434`
- OCR sidecar: RapidOCR local inference through `services/ocr_rapid.py`
- TTS sidecar: Kokoro API (default `http://127.0.0.1:5003`)
- Research sidecar: detached report generator in `research_sidecar.py`

### Startup Order

Recommended order for stable desktop operation:

1. Start local engine (`Start-Sage_v5-Engine.bat`) if not already running
2. Start GUI (`Start-Sage_v5-GUI.bat`)
3. Use Dashboard `Health` to verify sidecar readiness

Notes:

- GUI startup waits briefly for engine and then continues even if engine is still offline.
- Single-instance protection is enabled for the GUI to avoid duplicate windows.

### Engine Sidecar (Ollama)

Launcher:

```powershell
cmd /c .\Start-Sage_v5-Engine.bat
```

Expected behavior:

- Ollama listens on port `11434`
- GUI status indicator shows `Engine: online`

Quick check:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 11434
```

### OCR Sidecar (RapidOCR)

Flow:

1. Drop image files into `data/ocr_inbox`
2. Open `OCR Inbox`
3. Use `Selected` or `Newest`

Expected behavior:

- OCR text appears in output panel
- Summary appears under local model summary block
- OCR log updates in real time

### TTS Sidecar (Kokoro)

Configuration source: `core/config.py`

- enabled by `sidecars.tts_enabled`
- endpoint from `sidecars.kokoro_api_url`
- voices constrained to `af_sky` and `bm_george`

Lane mapping:

- Sage (qwen2.5:3b) lane -> `bm_george`
- Gemma local lane -> `af_sky`

Expected behavior:

- `Speak` actions play audio without blocking UI
- `⏹` stops active playback

### Research Sidecar

Purpose:

- Run detached web research and write a markdown report for later review.

Files:

- sidecar script: `research_sidecar.py`
- launcher script: `Start-Sage_v5-Research.bat`
- report output: `data/research/RESEARCH_REPORT.md`
- status output: `data/research/research_status.json`

Run from launcher:

```powershell
cmd /c .\Start-Sage_v5-Research.bat
```

Run directly:

```powershell
.\.venv\Scripts\python.exe .\research_sidecar.py --query "your topic" --workspace .
```

Optional dependencies for richer research results:

```powershell
.\.venv\Scripts\pip.exe install smolagents ddgs duckduckgo-search litellm
```

### Health and Diagnostics

GUI health button validates:

- engine reachability
- OCR inbox availability
- notes archive availability
- voice service initialization

Operational logs:

- engine watchdog log: `data/logs/engine_watchdog.log`
- research status: `data/research/research_status.json`

### Troubleshooting Quick Guide

- Engine offline:
	- run `Start-Sage_v5-Engine.bat`
	- verify `127.0.0.1:11434` listener
- OCR import/engine issues:
	- ensure `rapidocr-onnxruntime` is installed in `.venv`
- No TTS audio:
	- verify Kokoro endpoint and `sidecars.tts_enabled`
- Research report not generated:
	- inspect `data/research/research_status.json`
	- install optional research dependencies if missing

## Notes Reminders (Phase 1)

Phase 1 reminders are fully in-app and portable. They run only while the Sage GUI is open.

- No Windows Task Scheduler integration is required in this phase.
- Reminder scan interval is approximately every 30 seconds.
- Reminder notes are `.txt` files in `data/notes` with frontmatter.

Example reminder note:

```text
---
type: reminder
title: Pay electricity bill
status: pending
due_at: 2026-04-01 18:30
repeat: monthly
---

Pay online before 9 PM.
```

Supported repeat values:

- `none` / `once` for one-time reminders
- `daily`
- `weekly`
- `monthly`

When a reminder fires:

- one-time reminders are marked `done`
- repeating reminders are moved to the next `due_at`

### Gold Master Sync Policy

- Active workspace: `C:\Sage_v5`
- Backup workspace: `D:\Sage_v5`
- Sync strategy: manual, one-direction copy when needed to avoid merge conflicts

## License

MIT

