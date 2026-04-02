from __future__ import annotations

import asyncio

from core.config import load_config
from interface.shell import run_shell


async def _amain() -> int:
    cfg = load_config()
    print(f"{cfg.app_name} booting in {cfg.environment} mode")
    print(f"models: sage_local={cfg.models.names.sage_local}, gemma={cfg.models.names.gemma}")
    print(f"voices: {', '.join(cfg.kokoro.allowed_voices)}")
    await run_shell()
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
