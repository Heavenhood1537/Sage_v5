from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import os
import threading as _threading
import time
from typing import Any, AsyncIterator, Literal
from concurrent.futures import Future

import httpx

from core.config import AppConfig

# Persistent background event loop so the httpx.AsyncClient pool is never
# invalidated by asyncio.run() closing its loop between GUI calls.
_BG_LOOP: asyncio.AbstractEventLoop | None = None
_BG_LOOP_LOCK = _threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, creating it if needed."""
    global _BG_LOOP
    with _BG_LOOP_LOCK:
        if _BG_LOOP is None or _BG_LOOP.is_closed():
            _BG_LOOP = asyncio.new_event_loop()
            t = _threading.Thread(target=_BG_LOOP.run_forever, daemon=True)
            t.start()
    return _BG_LOOP


@dataclass
class LlmProvider:
    cfg: AppConfig
    _http_client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _gemma_candidates_cache: list[str] = field(default_factory=list, init=False, repr=False)
    _gemma_candidates_cache_ts: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        transport = self._build_retry_transport()
        client_kwargs: dict[str, Any] = {
            "limits": httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30,
            ),
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http_client = httpx.AsyncClient(**client_kwargs)

    BITNEY_IDENTITY = "Bitney'"
    SAGE_IDENTITY = "Sage | Culinary-Financial-Local Specialist"
    GEMMA_IDENTITY = "Gemma | Culinary Translation-Linguist Specialist"
    LOCAL_CONNECT_TIMEOUT = 5.0
    LOCAL_READ_TIMEOUT = 180.0   # long enough for heavy OCR/chat on slow hardware
    LOCAL_MAX_ATTEMPTS = 3

    def _sage_cpu_cap_percent(self) -> int:
        raw = os.environ.get("SAGE5_SAGE_CPU_CAP_PERCENT", "45")
        try:
            value = int(str(raw).strip())
        except Exception:
            value = 45
        return max(10, min(100, value))

    def _sage_thread_cap(self) -> int:
        logical_cpus = int(os.cpu_count() or 4)
        cap_threads = max(1, int(logical_cpus * (self._sage_cpu_cap_percent() / 100.0)))
        configured_threads = int(getattr(self.cfg.models, "sage_local_num_thread", cap_threads) or cap_threads)
        return max(1, min(configured_threads, cap_threads))

    def _gemma_cpu_cap_percent(self) -> int:
        default_pct = int(getattr(self.cfg.models, "gemma_cpu_cap_percent", 45) or 45)
        raw = os.environ.get("SAGE5_GEMMA_CPU_CAP_PERCENT", str(default_pct))
        try:
            value = int(str(raw).strip())
        except Exception:
            value = default_pct
        return max(10, min(100, value))

    def _ollama_keep_alive(self) -> int:
        default_keep_alive = int(getattr(self.cfg.models, "ollama_keep_alive", 0) or 0)
        raw = os.environ.get("SAGE5_OLLAMA_KEEP_ALIVE", str(default_keep_alive))
        try:
            value = int(str(raw).strip())
        except Exception:
            value = default_keep_alive
        return max(0, value)

    def _gemma_thread_cap(self) -> int:
        logical_cpus = int(os.cpu_count() or 4)
        return max(1, int(logical_cpus * (self._gemma_cpu_cap_percent() / 100.0)))

    @staticmethod
    def _looks_like_noisy_output(text: str) -> bool:
        value = str(text or "").strip()
        if len(value) < 20:
            return False
        alnum = sum(1 for ch in value if ch.isalnum())
        punct = sum(1 for ch in value if ch in "!@#$%^&*()[]{}|\\/:;,.?~`+-=_")
        if alnum <= 2 and punct >= 6:
            return True
        if punct >= max(8, int(len(value) * 0.55)) and alnum <= int(len(value) * 0.25):
            return True
        return False

    @staticmethod
    def _run_sync(coro):
        return asyncio.run_coroutine_threadsafe(coro, _get_bg_loop()).result()

    def run_coroutine(self, coro: Any) -> Any:
        """Run a coroutine on the persistent background loop (blocking)."""
        return asyncio.run_coroutine_threadsafe(coro, _get_bg_loop()).result()

    def submit_coroutine(self, coro: Any) -> Future:
        """Submit a coroutine to the persistent loop and return a cancelable Future."""
        return asyncio.run_coroutine_threadsafe(coro, _get_bg_loop())

    def with_historical_context(self, user_text: str, historical_context: str) -> str:
        prompt = str(user_text or "").strip()
        history = str(historical_context or "").strip()
        if not history:
            return prompt
        return (
            "Historical Context (last 10 turns):\n"
            f"{history}\n\n"
            "Current Request:\n"
            f"{prompt}"
        )

    @staticmethod
    def _join_url(base: str, suffix: str) -> str:
        return base.rstrip("/") + suffix

    @staticmethod
    def _normalize_ollama_base(endpoint: str) -> str:
        base = str(endpoint or "").strip().rstrip("/")
        if base.endswith("/v1"):
            return base[:-3]
        return base

    @staticmethod
    def _prefer_openai_for_local(endpoint: str) -> bool:
        base = str(endpoint or "").strip().lower()
        # OpenAI-compatible local endpoints usually expose /v1.
        return "/v1" in base or ":8080" in base

    async def unload_models_best_effort(self) -> None:
        """Ask Ollama to unload active Sage/Gemma models immediately."""
        assert self._http_client is not None
        keep_alive = self._ollama_keep_alive()

        async def _unload_one(base_endpoint: str, model_name: str) -> None:
            base = self._normalize_ollama_base(base_endpoint)
            endpoint = self._join_url(base, "/api/generate")
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": "",
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_predict": 1},
            }
            try:
                resp = await self._http_client.post(endpoint, json=payload, timeout=httpx.Timeout(2.0, connect=1.0))
                _ = resp.status_code
            except Exception:
                pass

        tasks = [
            _unload_one(self.cfg.models.endpoints.sage_local, self.cfg.models.names.sage_local),
            _unload_one(self.cfg.models.endpoints.gemma, self.cfg.models.names.gemma),
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        try:
            await self.unload_models_best_effort()
        except Exception:
            pass
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

    def _sage_local_system_prompt(self) -> str:
        return (
            "You are Sage, a Financial Strategist and Culinary Specialist.\n"
            "Rung 1: Financial Strategist. Give practical, risk-aware, numbers-first guidance.\n"
            "Rung 2: Culinary Specialist. Give precise cooking guidance with clear steps and substitutions.\n"
            "Rung 3: General Assistant. Be concise, accurate, and actionable.\n"
            "Rules: Prefer bullets, avoid fluff, and state assumptions briefly."
        )

    def _gemma_system_prompt(self) -> str:
        return (
            "You are Gemma, a Culinary Translation-Linguist Specialist. "
            "Translate clearly, preserve meaning, and keep ingredient and quantity details exact."
        )

    async def _fetch_ollama_tag_names(self) -> set[str]:
        base = self._normalize_ollama_base(self.cfg.models.endpoints.gemma)
        endpoint = self._join_url(base, "/api/tags")
        assert self._http_client is not None

        names: set[str] = set()
        try:
            resp = await self._http_client.get(endpoint, timeout=httpx.Timeout(6.0, connect=3.0))
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models") if isinstance(data, dict) else None
            if isinstance(models, list):
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip().lower()
                    if name:
                        names.add(name)
        except Exception:
            # Keep fallback behavior resilient when /api/tags is unavailable.
            return set()
        return names

    async def _gemma_model_candidates(self) -> list[str]:
        now = time.time()
        # Small TTL to avoid repeated /api/tags lookups across warmup + chat bursts.
        if self._gemma_candidates_cache and (now - self._gemma_candidates_cache_ts) < 45.0:
            return list(self._gemma_candidates_cache)

        configured = str(getattr(self.cfg.models.names, "gemma", "") or "").strip()
        preferred = [
            "gemma3n:e4b",
            "gemma3n:e2b",
            "gemma:4b",
            "gemma:2b",
        ]
        installed = await self._fetch_ollama_tag_names()

        candidates: list[str] = []
        if configured:
            candidates.append(configured)

        if installed:
            preferred_present = [name for name in preferred if name.lower() in installed]
            candidates.extend(preferred_present)

            # Include other installed Gemma-family tags as fallback safety net.
            others = sorted(
                [
                    name
                    for name in installed
                    if name.startswith("gemma")
                    and name not in {p.lower() for p in preferred_present}
                ],
            )
            candidates.extend(others)
        else:
            # If tags endpoint is unavailable, use static local fallback order.
            candidates.extend(preferred)

        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item)

        self._gemma_candidates_cache = out
        self._gemma_candidates_cache_ts = now
        return list(out)

    def _fallback_notice(self, local_error: Exception) -> str:
        return (
            "[MODEL NOTICE] Sage local model is unavailable right now. "
            "Switching to Gemma (Local) for this reply.\n"
        )

    def _local_timeout(self) -> httpx.Timeout:
        # GGUF models can take longer to emit first token.
        return httpx.Timeout(
            connect=self.LOCAL_CONNECT_TIMEOUT,
            read=self.LOCAL_READ_TIMEOUT,
            write=self.LOCAL_READ_TIMEOUT,
            pool=self.LOCAL_READ_TIMEOUT,
        )

    def _build_retry_transport(self) -> Any | None:
        """
        Build a best-effort httpx retry transport.
        Uses httpx.Retries when available, otherwise numeric retries.
        """
        retries = max(0, self.LOCAL_MAX_ATTEMPTS - 1)
        try:
            retries_cls = getattr(httpx, "Retries", None)
            if retries_cls is not None:
                retry_cfg = retries_cls(max_retries=retries, backoff_factor=0.3)
                return httpx.AsyncHTTPTransport(retries=retry_cfg)
        except Exception:
            pass

        try:
            return httpx.AsyncHTTPTransport(retries=retries)
        except Exception:
            return None

    async def _stream_sage_local_ollama(self, user_text: str) -> AsyncIterator[str]:
        base = self._normalize_ollama_base(self.cfg.models.endpoints.sage_local)
        endpoint = self._join_url(base, "/api/chat")
        payload: dict[str, Any] = {
            "model": self.cfg.models.names.sage_local,
            "messages": [
                {"role": "system", "content": self._sage_local_system_prompt()},
                {"role": "user", "content": user_text},
            ],
            "stream": True,
            "keep_alive": self._ollama_keep_alive(),
            "options": {
                "num_predict": 1024,
                "num_ctx": int(getattr(self.cfg.models, "sage_local_num_ctx", 4096) or 4096),
                "num_thread": self._sage_thread_cap(),
            },
        }

        timeout = self._local_timeout()

        assert self._http_client is not None
        async with self._http_client.stream("POST", endpoint, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = str(raw_line or "").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                piece = ""
                if isinstance(event, dict):
                    msg = event.get("message")
                    if isinstance(msg, dict):
                        piece = str(msg.get("content") or "")
                    if not piece:
                        piece = str(event.get("response") or "")
                    if bool(event.get("done")) and not piece:
                        break
                if piece:
                    yield piece

    async def _stream_openai_chat(
        self,
        endpoint_base: str,
        model_name: str,
        system_text: str,
        user_text: str,
        timeout: httpx.Timeout | None = None,
        use_retry_transport: bool = False,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        base = str(endpoint_base or "").strip().rstrip("/")
        endpoint = base if base.endswith("/chat/completions") else self._join_url(base, "/chat/completions")
        if "/v1/" not in endpoint and not endpoint.endswith("/v1/chat/completions"):
            endpoint = self._join_url(base, "/v1/chat/completions")

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.2,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        client_timeout = timeout or httpx.Timeout(75.0, connect=8.0)

        assert self._http_client is not None
        async with self._http_client.stream("POST", endpoint, json=payload, timeout=client_timeout) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                line = str(raw_line or "").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line in {"[DONE]", "done"}:
                    break

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                token = ""
                if isinstance(event, dict):
                    choices = event.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        token_key = "de" + "lta"
                        patch_data = first.get(token_key) if isinstance(first, dict) else {}
                        if isinstance(patch_data, dict):
                            token = str(patch_data.get("content") or "")
                        if not token:
                            message = first.get("message") if isinstance(first, dict) else {}
                            if isinstance(message, dict):
                                token = str(message.get("content") or "")
                if token:
                    yield token

    async def _call_sage_local(self, user_text: str) -> AsyncIterator[str]:
        errors: list[str] = []

        if self._prefer_openai_for_local(self.cfg.models.endpoints.sage_local):
            try:
                async for tok in self._stream_openai_chat(
                    endpoint_base=self.cfg.models.endpoints.sage_local,
                    model_name=self.cfg.models.names.sage_local,
                    system_text=self._sage_local_system_prompt(),
                    user_text=user_text,
                    timeout=self._local_timeout(),
                    use_retry_transport=True,
                    max_tokens=1024,
                ):
                    yield tok
                return
            except Exception as e:
                errors.append(f"openai={e}")

        try:
            async for tok in self._stream_sage_local_ollama(user_text):
                yield tok
            return
        except Exception as e:
            errors.append(f"ollama={e}")

        try:
            async for tok in self._stream_openai_chat(
                endpoint_base=self.cfg.models.endpoints.sage_local,
                model_name=self.cfg.models.names.sage_local,
                system_text=self._sage_local_system_prompt(),
                user_text=user_text,
                timeout=self._local_timeout(),
                use_retry_transport=True,
                max_tokens=1024,
            ):
                yield tok
            return
        except Exception as e:
            errors.append(f"openai={e}")

        raise RuntimeError("; ".join(errors) if errors else "sage_local_unavailable")

    async def _stream_sage_local(self, user_text: str) -> AsyncIterator[str]:
        errors: list[str] = []
        for attempt in range(1, self.LOCAL_MAX_ATTEMPTS + 1):
            try:
                async for tok in self._call_sage_local(user_text):
                    yield tok
                return
            except Exception as e:
                errors.append(f"attempt{attempt}={e}")
                if attempt < self.LOCAL_MAX_ATTEMPTS:
                    await asyncio.sleep(0.5 * attempt)

        raise RuntimeError("sage_local_retries_exhausted: " + "; ".join(errors))

    async def ping_gemma_warmup(self, timeout: float = 60.0) -> str:
        """
        Lightweight non-streaming warmup ping for the Gemma/Ollama model.

        Uses POST /api/generate with stream=false so Ollama returns a single JSON
        response object instead of an SSE stream.  This means the httpx read timeout
        applies to the whole response, not per-chunk — giving an accurate wall-clock
        ceiling and avoiding the silent hang that SSE streaming causes during cold loads.
        """
        base = self._normalize_ollama_base(self.cfg.models.endpoints.gemma)
        endpoint = self._join_url(base, "/api/generate")
        req_timeout = httpx.Timeout(timeout, connect=8.0)
        assert self._http_client is not None
        errors: list[str] = []
        for model_name in await self._gemma_model_candidates():
            payload: dict[str, Any] = {
                "model": model_name,
                "prompt": "hi",
                "stream": False,
                "keep_alive": self._ollama_keep_alive(),
                "options": {
                    "num_predict": 16,
                    "num_thread": self._gemma_thread_cap(),
                },
            }
            try:
                resp = await self._http_client.post(endpoint, json=payload, timeout=req_timeout)
                resp.raise_for_status()
                data = resp.json()
                text = str(data.get("response") or "").strip()
                if text:
                    return text[:80]
            except Exception as exc:
                errors.append(f"{model_name}={exc}")
        raise RuntimeError("gemma_warmup_failed: " + "; ".join(errors))

    async def _stream_gemma_local(self, user_text: str) -> AsyncIterator[str]:
        base = self._normalize_ollama_base(self.cfg.models.endpoints.gemma)
        endpoint = self._join_url(base, "/api/chat")
        assert self._http_client is not None

        errors: list[str] = []
        for model_name in await self._gemma_model_candidates():
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": self._gemma_system_prompt()},
                    {"role": "user", "content": user_text},
                ],
                "stream": True,
                "keep_alive": self._ollama_keep_alive(),
                "options": {
                    "num_predict": 768,
                    "num_thread": self._gemma_thread_cap(),
                },
            }
            try:
                async with self._http_client.stream("POST", endpoint, json=payload, timeout=httpx.Timeout(90.0, connect=8.0)) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        line = str(raw_line or "").strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except Exception:
                            continue

                        piece = ""
                        if isinstance(event, dict):
                            msg = event.get("message")
                            if isinstance(msg, dict):
                                piece = str(msg.get("content") or "")
                            if not piece:
                                piece = str(event.get("response") or "")
                            if bool(event.get("done")) and not piece:
                                break
                        if piece:
                            yield piece
                return
            except Exception as exc:
                errors.append(f"{model_name}={exc}")

        raise RuntimeError("gemma_local_unavailable: " + "; ".join(errors))

    async def chat(
        self,
        user_text: str,
        target: Literal["sage_local", "bitnet", "gemma"] | None = None,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Streaming-first provider API.

        - Streams by default so TTS/UI can speak on first token.
        - Silent fallback: Sage Local -> Gemma Local when local lane fails.
        """
        lane = target or self.cfg.models.active
        if lane == "bitnet":
            lane = "sage_local"
        prompt = str(user_text or "").strip()
        if not prompt:
            return

        if lane == "gemma":
            async for tok in self._stream_gemma_local(prompt):
                yield tok
            return

        try:
            buffered_tokens: list[str] = []
            buffered_text = ""
            deciding = True

            async for tok in self._stream_sage_local(prompt):
                if not tok:
                    continue

                if deciding:
                    buffered_tokens.append(tok)
                    buffered_text += tok

                    # Decide early if stream looks corrupted before emitting to UI/TTS.
                    if len(buffered_text) < 48:
                        continue

                    if self._looks_like_noisy_output(buffered_text):
                        raise RuntimeError("sage_local_noisy_output")

                    deciding = False
                    for piece in buffered_tokens:
                        yield piece
                    buffered_tokens = []
                    buffered_text = ""
                    continue

                yield tok

            if deciding and buffered_tokens:
                if self._looks_like_noisy_output(buffered_text):
                    raise RuntimeError("sage_local_noisy_output")
                for piece in buffered_tokens:
                    yield piece
            return
        except Exception as local_error:
            # Silent fallback path: announce and continue on Gemma local lane.
            yield self._fallback_notice(local_error)
            async for tok in self._stream_gemma_local(prompt):
                yield tok

    async def chat_text(
        self,
        user_text: str,
        target: Literal["sage_local", "bitnet", "gemma"] | None = None,
        stream: bool = True,
    ) -> str:
        parts: list[str] = []
        async for piece in self.chat(user_text=user_text, target=target, stream=stream):
            parts.append(piece)
        return "".join(parts).strip()

    # Compatibility wrappers for initial shell/UI integration.
    def chat_sage_local(self, user_text: str) -> str:
        return self._run_sync(self.chat_text(user_text=user_text, target="sage_local", stream=False))

    def chat_bitnet(self, user_text: str) -> str:
        return self.chat_sage_local(user_text)

    def chat_gemma(self, user_text: str) -> str:
        return self._run_sync(self.chat_text(user_text=user_text, target="gemma", stream=False))


# Backward-compatible alias for existing imports.
LLMProvider = LlmProvider
