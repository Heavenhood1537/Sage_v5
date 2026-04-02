# pyright: reportMissingImports=false
import argparse
import importlib
import json
import sys
import tempfile
import wave
from pathlib import Path


def _json_out(payload: dict):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


_MODEL_CACHE = {}


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _get_whisper_model(model_size: str):
    faster_whisper = importlib.import_module("faster_whisper")
    WhisperModel = faster_whisper.WhisperModel

    key = str(model_size or "base").strip() or "base"
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(key, device="cpu", compute_type="int8")
        _MODEL_CACHE[key] = model
    return model


def _resolve_input_device(sd, requested: str | None):
    value = str(requested or "").strip()
    if not value:
        return None

    try:
        if value.isdigit():
            idx = int(value)
            dev = sd.query_devices(idx)
            if int(dev.get("max_input_channels", 0) or 0) <= 0:
                raise ValueError(f"Device index {idx} is not an input device")
            return idx
    except Exception:
        pass

    low = value.lower()
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        try:
            if int(dev.get("max_input_channels", 0) or 0) <= 0:
                continue
            name = str(dev.get("name") or "")
            if low == name.lower() or low in name.lower():
                return idx
        except Exception:
            continue

    raise ValueError(f"Input device not found: {value}")


def _get_audio_devices(requested_device: str | None = None):
    sd = importlib.import_module("sounddevice")

    devices = sd.query_devices()
    input_devices = []
    for dev in devices:
        try:
            if int(dev.get("max_input_channels", 0) or 0) > 0:
                input_devices.append(dev)
        except Exception:
            continue

    default_input_name = "unknown"
    try:
        default_info = sd.query_devices(kind="input")
        default_input_name = str(default_info.get("name") or "unknown")
    except Exception:
        pass

    selected_input_name = default_input_name
    selected_input_index = None
    if requested_device:
        selected = _resolve_input_device(sd, requested_device)
        selected_input_index = int(selected)
        chosen = sd.query_devices(selected_input_index)
        selected_input_name = str(chosen.get("name") or selected_input_name)

    return len(input_devices), default_input_name, selected_input_name, selected_input_index


def _record_audio(
    seconds: int,
    samplerate: int,
    input_device=None,
    chunk_size: int = 1024,
    silence_reset_seconds: float = 4.0,
    min_stop_seconds: float = 5.5,
    stop_on_silence: bool = False,
):
    np = importlib.import_module("numpy")
    sd = importlib.import_module("sounddevice")

    frame_count = int(seconds * samplerate)
    if frame_count <= 0:
        frame_count = samplerate

    chunk = max(256, min(int(chunk_size or 1024), 4096))
    silence_reset = max(0.5, min(float(silence_reset_seconds or 4.0), 10.0))
    min_stop = max(0.0, min(float(min_stop_seconds or 0.0), float(seconds)))
    silence_floor = 0.0045

    frames_read = 0
    silent_seconds = 0.0
    chunks = []
    has_voiced_audio = False

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        blocksize=chunk,
        device=input_device,
    ) as stream:
        while frames_read < frame_count:
            to_read = min(chunk, frame_count - frames_read)
            data, overflowed = stream.read(to_read)
            block = np.squeeze(data)
            if block.ndim == 0:
                block = np.array([float(block)], dtype="float32")

            frames_read += len(block)
            chunks.append(block)

            peak = float(np.max(np.abs(block))) if len(block) else 0.0
            block_seconds = len(block) / float(samplerate)
            if peak <= silence_floor:
                silent_seconds += block_seconds
                if silent_seconds >= silence_reset:
                    if not has_voiced_audio:
                        chunks.clear()
                    # Never auto-stop before user has started speaking.
                    if stop_on_silence and has_voiced_audio and (frames_read / float(samplerate)) >= min_stop:
                        break
                    silent_seconds = 0.0
            else:
                has_voiced_audio = True
                silent_seconds = 0.0

    if not chunks:
        return np.zeros(max(1, min(chunk, frame_count)), dtype="float32")

    audio = np.concatenate(chunks)
    return audio.astype("float32")


def _write_wav(path: Path, audio, samplerate: int):
    np = importlib.import_module("numpy")

    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())


def _transcribe_wav(path: Path, model_size: str, language: str | None):
    model = _get_whisper_model(model_size)

    kwargs = {
        "beam_size": 2,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language

    segments, info = model.transcribe(str(path), **kwargs)
    text_parts = []
    for seg in segments:
        piece = (seg.text or "").strip()
        if piece:
            text_parts.append(piece)

    full_text = " ".join(text_parts).strip()
    detected_language = str(getattr(info, "language", "") or "")
    return full_text, detected_language


def _handle_check(input_device: str | None) -> dict:
    importlib.import_module("sounddevice")
    importlib.import_module("faster_whisper")
    count, default_input, selected_input, selected_index = _get_audio_devices(input_device)
    return {
        "ok": True,
        "input_devices": count,
        "default_input": default_input,
        "selected_input": selected_input,
        "selected_input_index": selected_index,
    }


def _handle_transcribe(
    seconds: int,
    samplerate: int,
    model_size: str,
    language: str,
    input_device: str | None,
    chunk_size: int = 1024,
    silence_reset_seconds: float = 4.0,
    min_stop_seconds: float = 5.5,
    stop_on_silence: bool = False,
) -> dict:
    seconds = max(1, min(int(seconds), 45))
    samplerate = max(8000, min(int(samplerate), 48000))
    model_size = (model_size or "base").strip() or "base"
    language = (language or "").strip()
    if language.lower() == "auto":
        language = ""

    sd = importlib.import_module("sounddevice")
    selected_input = None
    selected_input_name = "default"
    selected_input_index = None
    requested_input = str(input_device or "").strip()
    if requested_input:
        selected_input = _resolve_input_device(sd, requested_input)
        selected_input_index = int(selected_input)
        selected_info = sd.query_devices(selected_input_index)
        selected_input_name = str(selected_info.get("name") or "default")

    audio = _record_audio(
        seconds=seconds,
        samplerate=samplerate,
        input_device=selected_input,
        chunk_size=chunk_size,
        silence_reset_seconds=silence_reset_seconds,
        min_stop_seconds=min_stop_seconds,
        stop_on_silence=stop_on_silence,
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = Path(tf.name)

        _write_wav(tmp_path, audio, samplerate)
        text, detected_lang = _transcribe_wav(tmp_path, model_size=model_size, language=language or None)

        return {
            "ok": True,
            "text": text,
            "language": detected_lang,
            "seconds": seconds,
            "samplerate": samplerate,
            "model_size": model_size,
            "input_device": selected_input_name,
            "input_device_index": selected_input_index,
            "chunk_size": int(chunk_size),
            "silence_reset_seconds": float(silence_reset_seconds),
            "min_stop_seconds": float(min_stop_seconds),
            "stop_on_silence": bool(stop_on_silence),
        }
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _serve_forever(default_input_device: str):
    while True:
        line = sys.stdin.readline()
        if line == "":
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception:
            _json_out({"ok": False, "error": "invalid_json"})
            continue

        op = str(req.get("op") or "transcribe").strip().lower()
        try:
            if op == "shutdown":
                _json_out({"ok": True, "shutdown": True})
                break
            if op == "check":
                payload = _handle_check(str(req.get("input_device") or default_input_device))
                _json_out(payload)
                continue

            payload = _handle_transcribe(
                seconds=int(req.get("seconds", 5) or 5),
                samplerate=int(req.get("samplerate", 16000) or 16000),
                model_size=str(req.get("model_size") or "base"),
                language=str(req.get("language") or "en"),
                input_device=str(req.get("input_device") or default_input_device),
                chunk_size=int(req.get("chunk_size", 1024) or 1024),
                silence_reset_seconds=float(req.get("silence_reset_seconds", 2.0) or 2.0),
                min_stop_seconds=float(req.get("min_stop_seconds", 2.0) or 2.0),
                stop_on_silence=_as_bool(req.get("stop_on_silence", False)),
            )
            _json_out(payload)
        except Exception as e:
            _json_out({"ok": False, "error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="Local microphone STT sidecar for Sage.")
    parser.add_argument("--check", action="store_true", help="Probe imports/devices only")
    parser.add_argument("--seconds", type=int, default=5, help="Recording duration in seconds")
    parser.add_argument("--samplerate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--model-size", type=str, default="base", help="faster-whisper model size")
    parser.add_argument("--language", type=str, default="en", help="Language code, or 'auto'")
    parser.add_argument("--input-device", type=str, default="", help="Preferred input device name fragment or numeric index")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio input chunk size (recommended 512/1024)")
    parser.add_argument("--silence-reset-seconds", type=float, default=2.0, help="Clear buffered audio after continuous silence seconds")
    parser.add_argument("--min-stop-seconds", type=float, default=2.0, help="Minimum capture duration before stop-on-silence can end recording")
    parser.add_argument("--stop-on-silence", action="store_true", help="End recording early after continuous silence")
    parser.add_argument("--server", action="store_true", help="Run persistent JSON-line request server")
    args = parser.parse_args()

    try:
        if args.server:
            _serve_forever(args.input_device)
            return

        if args.check:
            _json_out(_handle_check(args.input_device))
            return

        _json_out(_handle_transcribe(
            seconds=args.seconds,
            samplerate=args.samplerate,
            model_size=args.model_size,
            language=args.language,
            input_device=args.input_device,
            chunk_size=args.chunk_size,
            silence_reset_seconds=args.silence_reset_seconds,
            min_stop_seconds=args.min_stop_seconds,
            stop_on_silence=args.stop_on_silence,
        ))

    except Exception as e:
        _json_out({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
