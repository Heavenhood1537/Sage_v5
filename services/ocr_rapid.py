from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path
from statistics import fmean
from typing import Any

from core.config import AppConfig


@dataclass
class OcrResult:
    image_path: str
    text: str
    confidence: float


@dataclass
class OcrService:
    cfg: AppConfig
    _engine: Any | None = None
    _seen_signatures: set[tuple[str, int, int]] | None = None

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    def __post_init__(self) -> None:
        self._seen_signatures = set()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def inbox_dir(self) -> Path:
        return self.project_root / self.cfg.paths.ocr_inbox_dir

    @property
    def models_dir(self) -> Path:
        return self.project_root / self.cfg.paths.models_ocr_dir

    def scan_inbox_for_new(self, limit: int | None = None) -> list[Path]:
        files = [
            p
            for p in sorted(self.inbox_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        fresh: list[Path] = []
        seen = self._seen_signatures if self._seen_signatures is not None else set()
        for p in files:
            st = p.stat()
            sig = (str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
            if sig in seen:
                continue
            seen.add(sig)
            fresh.append(p)
            if limit is not None and len(fresh) >= max(1, int(limit)):
                break
        return fresh

    def extract_text(self, image_path: str) -> OcrResult:
        path = self._resolve_image_path(image_path)
        engine = self._get_engine()

        result = engine(str(path))
        rows, text = self._normalize_rows_and_text(result)

        confidences = [score for _, score in rows]
        confidence = float(fmean(confidences)) if confidences else 0.0

        return OcrResult(
            image_path=str(path),
            text=(text or "").strip(),
            confidence=round(max(0.0, min(1.0, confidence)), 4),
        )

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(str(image_path or "").strip())
        if not candidate:
            raise FileNotFoundError("Image path is empty")

        if not candidate.is_absolute():
            direct = self.project_root / candidate
            inbox = self.inbox_dir / candidate
            if direct.exists():
                candidate = direct
            elif inbox.exists():
                candidate = inbox

        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if candidate.suffix.lower() not in self.IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {candidate.suffix}")
        return candidate.resolve()

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine

        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[reportMissingImports]
        except Exception as exc:
            raise RuntimeError(
                "RapidOCR is not available. Install dependency: rapidocr-onnxruntime"
            ) from exc

        self._bootstrap_models_from_package()

        det_model = self._find_model("det")
        rec_model = self._find_model("rec")
        cls_model = self._find_model("cls")

        kwargs: dict[str, Any] = {}
        if det_model is not None:
            kwargs["det_model_path"] = str(det_model)
        if rec_model is not None:
            kwargs["rec_model_path"] = str(rec_model)
        if cls_model is not None:
            kwargs["cls_model_path"] = str(cls_model)

        self._engine = RapidOCR(**kwargs)
        return self._engine

    def _bootstrap_models_from_package(self) -> None:
        """
        Copy packaged ONNX OCR models into models/ocr for portable local reuse.
        """
        try:
            import rapidocr_onnxruntime as pkg  # type: ignore[reportMissingImports]
        except Exception:
            return

        pkg_root = Path(getattr(pkg, "__file__", "")).resolve().parent
        if not pkg_root.exists():
            return

        for onnx_file in pkg_root.rglob("*.onnx"):
            target = self.models_dir / onnx_file.name
            if target.exists():
                continue
            try:
                shutil.copy2(onnx_file, target)
            except Exception:
                continue

    def _find_model(self, kind: str) -> Path | None:
        candidates = sorted(self.models_dir.glob("*.onnx"))
        for p in candidates:
            name = p.name.lower()
            if kind == "det" and "det" in name:
                return p
            if kind == "rec" and "rec" in name:
                return p
            if kind == "cls" and "cls" in name:
                return p
        return None

    def _normalize_rows_and_text(self, result: Any) -> tuple[list[tuple[str, float]], str]:
        rows_obj = result
        if isinstance(result, tuple):
            rows_obj = result[0] if result else []

        rows: list[tuple[str, float]] = []

        if isinstance(rows_obj, list):
            for item in rows_obj:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    text = str(item[1] or "").strip()
                    try:
                        score = float(item[2])
                    except Exception:
                        score = 0.0
                    if text:
                        rows.append((text, score))
                elif isinstance(item, dict):
                    text = str(item.get("text") or item.get("transcription") or "").strip()
                    raw_score = item.get("score", item.get("confidence", 0.0))
                    try:
                        score = float(raw_score)
                    except Exception:
                        score = 0.0
                    if text:
                        rows.append((text, score))

        text_out = "\n".join([t for t, _ in rows]).strip()
        if not text_out and isinstance(result, str):
            text_out = result.strip()

        return rows, text_out


# Backward-compatible alias for early scaffold imports.
RapidLightOCR = OcrService
