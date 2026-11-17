from __future__ import annotations

import pytest

from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import RuleLLM, ScriptedLLM
from trellis_agent.memory.extract import LLMExtractor, RuleExtractor
from trellis_agent.memory.manager import MemoryConfig, MemoryManager
from trellis_agent.memory.store import MemoryStore
from trellis_agent.trace import summarize
from trellis_agent.types import Message, Usage


def memory(extractor=None, **cfg: object) -> MemoryManager:
    return MemoryManager(MemoryStore(), extractor or RuleExtractor(), MemoryConfig(**cfg))  # type: ignore[arg-type]


def echo_llm() -> RuleLLM:
    return RuleLLM(lambda m, t: Message.assistant("ok"))


async def test_without_memory_the_agent_behaves_exactly_as_before() -> None:
    llm = echo_llm()
    agent = Agent(llm, config=AgentConfig(system_prompt="sys"))
    result = await agent.arun("hello")
    assert [m.role for m in result.messages] == ["system", "user", "assistant"]
    assert [e.type for e in result.trace] == ["run_start", "model_request", "run_end"]
    assert result.usage == Usage(10, 5) and agent.memory is None
    assert [m.role for m in llm.calls[0].messages] == ["system", "user"]


async def test_a_fact_from_an_early_turn_is_recalled_after_it_left_the_working_window() -> None:
    llm = echo_llm()
    agent = Agent(
        llm, memory=memory(working_token_budget=40), config=AgentConfig(system_prompt="sys")
    )
    await agent.arun("我的截止日期是周五", session_id="s")
    for i in range(6):
        await agent.arun(f"闲聊第{i}句 " + "填充 " * 12, session_id="s")
    await agent.arun("截止日期是哪天?", session_id="s")
    sent = llm.calls[-1].messages
    assert sent[0].content == "sys"  # the system prompt is still first
    notes = [m.content or "" for m in sent if m.role == "system" and "周五" in (m.content or "")]
    assert notes, "the old fact must come back through memory"
    assert not any(
        (m.content or "") == "我的截止日期是周五" for m in sent if m.role == "user"
    )  # trimmed
    assert len(agent._sessions["s"]) == 2 * 8  # ...while the full history is still stored


async def test_the_memory_note_comes_after_the_system_prompt_and_before_the_conversation() -> None:
    llm = echo_llm()
    agent = Agent(llm, memory=memory(), config=AgentConfig(system_prompt="sys"))
    await agent.arun("我的截止日期是周五", session_id="s")
    await agent.arun("截止日期呢", session_id="s")
    roles = [m.role for m in llm.calls[-1].messages]
    assert roles[:2] == ["system", "system"] and "周五" in (llm.calls[-1].messages[1].content or "")


async def test_memory_overhead_is_counted_in_usage_and_cost_and_visible_in_the_trace() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1_000_000.0, 1_000_000.0)}))  # cost == tokens
    extractor_llm = ScriptedLLM(
        [Message.assistant('{"facts": [{"text": "User likes tea", "key": "drink"}]}')],
        usage_per_call=Usage(100, 20),
        meter=meter,
    )
    answer_llm = ScriptedLLM([Message.assistant("ok")], usage_per_call=Usage(10, 5), meter=meter)
    agent = Agent(answer_llm, memory=memory(LLMExtractor(extractor_llm)))
    result = await agent.arun("I like tea", session_id="s")
    assert result.usage == Usage(110, 25)  # the answer call plus the extraction call
    assert result.cost_usd == pytest.approx(135.0) == pytest.approx(meter.total_usd)
    phases = [e.data.get("phase") for e in result.trace if e.type == "model_request"]
    assert "memory_extract" in phases
    s = summarize(result.trace)
    assert s.usage == result.usage and s.cost_usd == pytest.approx(result.cost_usd)
    assert any(e.type == "memory_retrieve" for e in result.trace)


async def test_a_failing_memory_never_fails_the_run() -> None:
    class Broken:
        async def extract(self, turn: list[Message]):  # type: ignore[no-untyped-def]
            raise RuntimeError("extractor bug")

    agent = Agent(echo_llm(), memory=memory(Broken()))
    result = await agent.arun("hello", session_id="s")
    assert result.error is None and result.content == "ok"
    assert [e.data["category"] for e in result.trace if e.type == "error"] == ["memory_error"]
