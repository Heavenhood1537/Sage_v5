from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelEndpoints(BaseModel):
    # Keep endpoints externalized and simple for future swaps.
    sage_local: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("sage_local", "bitnet"),
    )
    gemma: str = Field(default="http://127.0.0.1:11434")


class ModelNames(BaseModel):
    # Scope intentionally narrow: only the two approved models.
    sage_local: str = Field(
        default="qwen2.5:3b",
        validation_alias=AliasChoices("sage_local", "bitnet"),
    )
    gemma: str = Field(default="gemma3n:e4b")


class ModelsConfig(BaseModel):
    active: Literal["sage_local", "bitnet", "gemma"] = Field(default="sage_local")
    endpoints: ModelEndpoints = Field(default_factory=ModelEndpoints)
    names: ModelNames = Field(default_factory=ModelNames)
    sage_local_num_ctx: int = Field(default=4096)
    sage_local_num_thread: int = Field(default=4)
    gemma_cpu_cap_percent: int = Field(default=45)
    ollama_keep_alive: int = Field(default=0)


class KokoroConfig(BaseModel):
    engine: Literal["kokoro"] = Field(default="kokoro")
    default_voice: Literal["af_sky", "bm_george"] = Field(default="af_sky")
    allowed_voices: tuple[str, str] = ("af_sky", "bm_george")
    # Absolute path to the folder containing kokoro-v1.0.onnx and voices-v1.0.bin.
    # Empty string means auto-discover from project VOICE/kokoro/ or sibling workspaces.
    model_dir: str = Field(default="")


class SidecarsConfig(BaseModel):
    # Keep only sidecars in active use.
    light_ocr_enabled: bool = True
    stt_enabled: bool = True
    tts_enabled: bool = True
    kokoro_api_url: str = "http://127.0.0.1:5003"
    research_enabled: bool = True
    research_max_results: int = 8


class RelativePaths(BaseModel):
    # All paths stay relative to project root for T7 portability.
    data_dir: str = "data"
    logs_dir: str = "data/logs"
    memory_dir: str = "data/memory"
    ocr_inbox_dir: str = "data/ocr_inbox"
    models_ocr_dir: str = "models/ocr"
    sidecar_ocr_dir: str = "OCR/rapidocr"
    sidecar_stt_dir: str = "VOICE/stt"
    sidecar_tts_dir: str = "VOICE/kokoro"
    tools_audacity_dir: str = "tools/audacity"
    tools_foobar_dir: str = "tools/foobar"

    @field_validator("*")
    @classmethod
    def ensure_relative(cls, value: str) -> str:
        path = Path(str(value or "").strip())
        if path.is_absolute():
            raise ValueError(f"Path must be relative, got absolute path: {value}")
        return path.as_posix()


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAGE5_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "Sage v5"
    environment: Literal["dev", "prod"] = "dev"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)
    sidecars: SidecarsConfig = Field(default_factory=SidecarsConfig)
    paths: RelativePaths = Field(default_factory=RelativePaths)


def resolve_path(root: Path, relative_path: str) -> Path:
    return (root / relative_path).resolve()


def load_config() -> AppConfig:
    return AppConfig()
