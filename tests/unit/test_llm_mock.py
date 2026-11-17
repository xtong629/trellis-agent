import math

import pytest

from trellis_agent.cost import BudgetExceeded, CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import (
    Embedder,
    HashEmbedder,
    LLMClient,
    LLMResponse,
    RuleLLM,
    ScriptedLLM,
    ScriptExhausted,
)
from trellis_agent.types import Message, ToolCall, ToolSchema, Usage

SCHEMA = ToolSchema("add", "Add.", {"type": "object", "properties": {}})


async def test_scripted_llm_replays_in_order_and_records_calls() -> None:
    call = ToolCall("c1", "add", {"a": 1})
    llm = ScriptedLLM([Message.assistant(None, (call,)), Message.assistant("done")])
    first = await llm.chat([Message.user("hi")], tools=[SCHEMA])
    second = await llm.chat([Message.user("again")])
    assert isinstance(first, LLMResponse)
    assert first.message.tool_calls == (call,) and first.finish_reason == "tool_calls"
    assert second.message.content == "done" and second.finish_reason == "stop"
    assert first.usage == Usage(10, 5) and first.latency_ms == 0.0 and first.model == "mock"
    assert len(llm.calls) == 2
    assert llm.calls[0].messages == (Message.user("hi"),) and llm.calls[0].tools == (SCHEMA,)
    assert llm.calls[1].tools is None


async def test_scripted_llm_supports_callables_and_reports_exhaustion() -> None:
    llm = ScriptedLLM([lambda msgs: Message.assistant(f"saw {len(msgs)}")])
    assert (await llm.chat([Message.user("a"), Message.user("b")])).message.content == "saw 2"
    with pytest.raises(ScriptExhausted, match="1"):
        await llm.chat([Message.user("c")])


async def test_rule_llm_decides_from_messages_and_tools() -> None:
    def rule(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
        return Message.assistant(f"{messages[-1].content}|{len(tools or [])}")

    llm = RuleLLM(rule, usage_per_call=Usage(3, 4))
    response = await llm.chat([Message.user("q")], tools=[SCHEMA])
    assert response.message.content == "q|1" and response.usage == Usage(3, 4)


async def test_mock_llms_record_cost_in_the_meter_with_tag() -> None:
    prices = PriceTable({"mock": ModelPrice(1.0, 2.0)})
    meter = CostMeter(prices)
    llm = ScriptedLLM(
        [Message.assistant("x")], usage_per_call=Usage(1_000_000, 500_000), meter=meter
    )
    response = await llm.chat([Message.user("q")], tag="plan")
    assert response.cost_usd == pytest.approx(2.0)
    assert meter.by_tag == {"plan": pytest.approx(2.0)}


async def test_mock_llm_respects_budget() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 1.0)}), budget_usd=0.5)
    llm = RuleLLM(
        lambda m, t: Message.assistant("x"), usage_per_call=Usage(1_000_000, 0), meter=meter
    )
    with pytest.raises(BudgetExceeded):
        await llm.chat([Message.user("q")])


def test_mocks_satisfy_the_protocols() -> None:
    llm: LLMClient = ScriptedLLM([])
    emb: Embedder = HashEmbedder()
    assert llm.model == "mock" and emb.dim == 64


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_hash_embedder_is_deterministic_and_normalised() -> None:
    (v1,) = await HashEmbedder(dim=32).embed(["refund process takes three days"])
    (v2,) = await HashEmbedder(dim=32).embed(["refund process takes three days"])
    assert v1 == v2 and len(v1) == 32
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0)


async def test_hash_embedder_shared_words_are_more_similar() -> None:
    emb = HashEmbedder(dim=256)
    q, near, far = await emb.embed(
        ["how long does a refund take", "refund takes three days", "the weather is sunny today"]
    )
    assert _cos(q, near) > _cos(q, far)


async def test_hash_embedder_handles_empty_text() -> None:
    (v,) = await HashEmbedder(dim=8).embed([""])
    assert v == [0.0] * 8
