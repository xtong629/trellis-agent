"""Helpers shared by the examples. Not part of the library API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from trellis_agent.cost import CostMeter
from trellis_agent.llm import LLMClient, LLMError
from trellis_agent.llm.openai_compat import OpenAICompatClient


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="use a scripted fake model (no key, no network); else read LLM_* from env/.env",
    )
    return parser.parse_args()


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env reader (KEY=VALUE lines); never overrides variables already set."""
    file = Path(path)
    if not file.is_file():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def check_llm_env() -> None:
    """Exit with a clear message unless LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are available."""
    load_dotenv()
    missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
    if missing:
        print(
            f"error: missing {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in, or pass --mock.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def real_client(meter: CostMeter | None = None) -> OpenAICompatClient:
    check_llm_env()
    try:
        return OpenAICompatClient.from_env(meter=meter)
    except LLMError as exc:  # pragma: no cover - check_llm_env already covers this
        raise SystemExit(f"error: {exc}") from exc


def pick(mock: bool, mock_llm: LLMClient, meter: CostMeter | None = None) -> LLMClient:
    return mock_llm if mock else real_client(meter)
