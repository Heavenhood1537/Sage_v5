from __future__ import annotations

import asyncio
import re
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from core.config import AppConfig

# ── model file names (Kokoro v1.0 release) ─────────────────────────────────
_ONNX_NAME = "kokoro-v1.0.onnx"
_VOICES_NAME = "voices-v1.0.bin"

# Canonical Kokoro model directory for Sage v5 on Windows.
_CANONICAL_MODEL_DIR = Path(r"C:\Sage_v5\models\voice\kokoro")


def _model_dir_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "models" / "voice" / "kokoro",
        project_root / "models" / "Voice" / "kokoro",
        _CANONICAL_MODEL_DIR,
    ]
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve(strict=False)).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _find_model_dir(cfg: AppConfig) -> Path | None:
    """Return canonical Sage v5 model directory when required Kokoro files exist."""
    for d in _model_dir_candidates():
        if (d / _ONNX_NAME).exists() and (d / _VOICES_NAME).exists():
            return d
    return None


def _build_wav_bytes(pcm_int16: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw PCM int16 samples in a minimal WAV container."""
    data_len = len(pcm_int16)
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_len, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", data_len,
    )
    return hdr + pcm_int16


@dataclass
class TtsService:
    cfg: AppConfig
    _play_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)
    _engine: object = field(default=None, init=False, repr=False)
    _engine_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ── static helpers ──────────────────────────────────────────────────────

    def voice_for_target(self, target: Literal["sage_local", "bitnet", "gemma"]) -> str:
        if target in {"sage_local", "bitnet"}:
            return "bm_george"
        return "af_sky"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        value = str(text or "").strip()
        if not value:
            return []
        chunks = re.split(r"(?<=[.!?。！？])\s+", value)
        return [c.strip() for c in chunks if c.strip()]

    # ── engine lifecycle ────────────────────────────────────────────────────

    def _get_engine(self) -> object:
        """Return a cached Kokoro engine instance, creating it on first call."""
        with self._engine_lock:
            if self._engine is not None:
                return self._engine

            model_dir = _find_model_dir(self.cfg)
            if model_dir is None:
                expected_dirs = " or ".join(p.as_posix() for p in _model_dir_candidates())
                raise RuntimeError(
                    "Kokoro model files not found. "
                    f"Expected '{_ONNX_NAME}' and '{_VOICES_NAME}' in: "
                    f"{expected_dirs}"
                )

            try:
                import kokoro_onnx  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro-onnx is not installed. "
                    r"Run: .venv\Scripts\pip install kokoro-onnx"
                ) from exc

            self._engine = kokoro_onnx.Kokoro(
                model_path=str(model_dir / _ONNX_NAME),
                voices_path=str(model_dir / _VOICES_NAME),
            )
            return self._engine

    # ── synthesis + playback ────────────────────────────────────────────────

    def _synthesise_and_play(self, text: str, voice: str) -> None:
        """Synthesise text to PCM and play synchronously via winsound."""
        import numpy as np   # type: ignore[import]
        import winsound

        engine = self._get_engine()

        with self._play_lock:
            if self._stop_requested:
                return

            # af_sky: keep pitch/intonation more feminine by avoiding WAV-rate
            # downshift; use a mild native speed control instead.
            # bm_george (Sage): slightly slower delivery for clarity.
            synth_speed = 0.92 if voice == "af_sky" else 0.90
            audio, sample_rate = engine.create(text, voice=voice, speed=synth_speed, lang="en-us")
            if audio is None:
                return

            arr = np.asarray(audio, dtype=np.float32)
            arr = np.clip(arr, -1.0, 1.0)

            # Per-voice volume attenuation.
            if voice == "af_sky":
                arr = arr * 0.446   # −55%
            elif voice == "bm_george":
                arr = arr * 0.286   # +5% vs previous Sage gain (0.272 -> 0.286)

            # bm_george: brighten timbre without changing duration.
            # Use a stronger pitch-lift resample, then stretch back to original length.
            if voice == "bm_george" and len(arr) > 16:
                ratio = 2.0 ** (3.8 / 12.0)  # +3.8 semitones (clearer, less bass)
                src_idx = np.arange(len(arr), dtype=np.float32)
                pitched_idx = np.arange(0.0, float(len(arr)), ratio, dtype=np.float32)
                pitched = np.interp(pitched_idx, src_idx, arr).astype(np.float32)
                if len(pitched) > 8:
                    out_idx = np.linspace(0.0, float(len(pitched) - 1), num=len(arr), dtype=np.float32)
                    arr = np.interp(out_idx, np.arange(len(pitched), dtype=np.float32), pitched).astype(np.float32)
                    arr = np.clip(arr, -1.0, 1.0)

            # Trim leading and trailing silence.
            sr = int(sample_rate or 24000)
            threshold = 0.003
            non_silent = np.where(np.abs(arr) > threshold)[0]
            if non_silent.size:
                tail_samples = int(sr * 0.12) if voice == "bm_george" else sr // 3
                arr = arr[non_silent[0]: min(int(non_silent[-1]) + tail_samples, len(arr))]

            pcm = (arr * 32767.0).astype(np.int16)

            # Keep declared WAV rate unchanged to avoid lowering bm_george pitch.
            speed_factor = 1.0
            effective_sr = int(sr * speed_factor)
            wav_bytes = _build_wav_bytes(pcm.tobytes(), sample_rate=effective_sr)

            winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)

    # ── public API ──────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop_requested = True
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    async def speak(self, text: str, target: Literal["sage_local", "bitnet", "gemma"] = "sage_local") -> None:
        if not bool(self.cfg.sidecars.tts_enabled):
            return
        value = str(text or "").strip()
        if not value:
            return
        if self._stop_requested:
            self._stop_requested = False

        voice = self.voice_for_target(target)
        if voice not in self.cfg.kokoro.allowed_voices:
            raise ValueError(f"Unsupported voice: {voice}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._synthesise_and_play, value, voice)

    async def speak_chunked(self, text: str, target: Literal["sage_local", "bitnet", "gemma"] = "sage_local") -> None:
        for chunk in self._split_sentences(text):
            if self._stop_requested:
                break
            await self.speak(chunk, target=target)


# Backward-compatible alias during migration.
KokoroTTS = TtsService
