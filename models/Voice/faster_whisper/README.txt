Sage Voice Sidecar (faster-whisper)
===================================

Purpose
-------
Local microphone speech-to-text sidecar used by Sage commands:
- voice-check
- voice-once [seconds]

Setup (Windows, from C:\Sage)
------------------------------
1) Create sidecar venv:
   py -3.14 -m venv VOICE/faster_whisper/.venv

2) Install sidecar dependencies:
   VOICE/faster_whisper/.venv/Scripts/python.exe -m pip install -r VOICE/faster_whisper/requirements.txt

3) Verify from Sage:
   voice-check

Notes
-----
- First transcription run will download the selected whisper model.
- Default model is "base" (CPU-friendly balance).
- You can change runtime settings in sage_config.json under "voice".

Manual sidecar test
-------------------
VOICE/faster_whisper/.venv/Scripts/python.exe VOICE/faster_whisper/run_stt.py --check
VOICE/faster_whisper/.venv/Scripts/python.exe VOICE/faster_whisper/run_stt.py --seconds 5 --model-size base --language en
