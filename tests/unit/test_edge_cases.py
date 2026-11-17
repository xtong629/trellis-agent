"""Boundary behaviour that is easy to get wrong silently."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from trellis_agent.agent import AgentConfig
from trellis_agent.cost import CostMeter, PriceTable
from trellis_agent.llm import HashEmbedder
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.trace import summarize
from trellis_agent.types import ToolCall, TraceEvent


def test_agent_config_rejects_zero_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        AgentConfig(max_steps=0)


def test_negative_budget_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        CostMeter(PriceTable(), budget_usd=-1)


def test_price_file_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        PriceTable.from_yaml(path)


def test_hash_embedder_rejects_bad_dim() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_summarize_tolerates_a_malformed_usage_payload() -> None:
    event = TraceEvent("model_request", 0.0, {"usage": "oops", "latency_ms": None}, "root", None)
    s = summarize([event])
    assert s.usage.total_tokens == 0 and s.model_request_count == 1


def test_bare_container_and_any_annotations() -> None:
    @tool
    def f(a: list, b: dict, c: Any) -> None:  # type: ignore[type-arg]
        """Bare containers."""

    props = f.schema().parameters["properties"]
    assert props == {"a": {"type": "array"}, "b": {"type": "object"}, "c": {}}


async def test_sync_tool_that_returns_an_awaitable_is_awaited() -> None:
    @tool
    def deferred(a: int) -> Any:
        """Returns a coroutine from a plain function."""
        return asyncio.sleep(0, result=a * 2)

    result = await ToolRegistry([deferred]).call(ToolCall("c", "deferred", {"a": 4}))
    assert result.content == "8" and not result.is_error
