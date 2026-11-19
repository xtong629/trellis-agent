"""Reflection: a checker reviews a finished answer; on rejection the agent retries once with the
reason. Written before the implementation; mock models only."""

from __future__ import annotations

from typing import Any

import pytest

from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.cost import BudgetExceeded, CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import RuleLLM, ScriptedLLM
from trellis_agent.reflect import (
    LLMReflector,
    ReflectingAgent,
    ReflectionRecord,
    Verdict,
)
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.types import Message, RunResult, ToolCall, Usage


@tool
def lookup(key: str) -> int:
    """Look a number up."""
    return {"a": 7, "b": 5}[key]


def agent_answering(*answers: str) -> Agent:
    """An agent whose model calls ``lookup`` once per turn, then gives the next answer."""
    replies = iter(answers)

    def fn(messages: list[Message], tools: Any) -> Message:
        if messages[-1].role == "user":
            return Message.assistant(None, (ToolCall(f"c{len(messages)}", "lookup", {"key": "a"}),))
        return Message.assistant(next(replies))

    return Agent(RuleLLM(fn), tools=ToolRegistry([lookup]))


class FixedReflector:
    def __init__(self, *verdicts: Verdict | Exception) -> None:
        self.verdicts = list(verdicts)
        self.seen: list[tuple[str, str]] = []

    async def check(self, query: str, result: RunResult) -> Verdict:
        self.seen.append((query, result.content))
        verdict = self.verdicts.pop(0)
        if isinstance(verdict, Exception):
            raise verdict
        return verdict


CHECK_USAGE = Usage(prompt_tokens=50, completion_tokens=10)


def requests(result: RunResult) -> int:
    return sum(1 for e in result.trace if e.type == "model_request")


async def test_an_accepted_answer_is_returned_unchanged_plus_the_cost_of_the_check() -> None:
    reflector = FixedReflector(Verdict(True, usage=CHECK_USAGE, cost_usd=0.5, calls=1))
    result = await ReflectingAgent(agent_answering("ANSWER: 7"), reflector).arun("q?")
    assert result.content == "ANSWER: 7" and result.stop_reason == "end_turn"
    assert reflector.seen == [("q?", "ANSWER: 7")]
    assert requests(result) == 3  # tool step, answer, one check
    assert result.usage.prompt_tokens >= 50
    assert [e.data["ok"] for e in result.trace if e.type == "reflection"] == [True]


async def test_a_rejected_answer_gets_exactly_one_retry_with_the_reason() -> None:
    inner = agent_answering("ANSWER: 5", "ANSWER: 7")
    reflector = FixedReflector(Verdict(False, "used the wrong key", usage=CHECK_USAGE, calls=1))
    result = await ReflectingAgent(inner, reflector).arun("q?", session_id="s")
    assert result.content == "ANSWER: 7"
    assert len(reflector.seen) == 1  # the retry's answer is final and not checked again
    retry_turn = [m.content for m in result.messages if m.role == "user"][-1] or ""
    assert "used the wrong key" in retry_turn and "ANSWER" not in retry_turn.split(":")[0]
    assert requests(result) == 2 + 1 + 2  # first attempt, check, retry (tool step + answer)
    assert [e.data["name"] for e in result.trace if e.type == "tool_call"] == ["lookup", "lookup"]


async def test_more_retries_are_checked_until_the_last_one() -> None:
    inner = agent_answering("1", "2", "3")
    no = Verdict(False, "wrong")
    reflector = FixedReflector(no, no)
    result = await ReflectingAgent(inner, reflector, max_retries=2).arun("q?")
    assert result.content == "3" and len(reflector.seen) == 2


async def test_a_run_without_a_final_answer_is_not_checked() -> None:
    def loops(messages: list[Message], tools: Any) -> Message:
        return Message.assistant(None, (ToolCall(f"c{len(messages)}", "lookup", {"key": "a"}),))

    inner = Agent(RuleLLM(loops), tools=ToolRegistry([lookup]), config=AgentConfig(max_steps=2))
    reflector = FixedReflector()
    result = await ReflectingAgent(inner, reflector).arun("q?")
    assert result.stop_reason == "max_steps" and reflector.seen == []


async def test_a_failing_checker_never_fails_the_run() -> None:
    reflector = FixedReflector(RuntimeError("checker down"))
    result = await ReflectingAgent(agent_answering("ANSWER: 7"), reflector).arun("q?")
    assert result.content == "ANSWER: 7" and result.error is None
    errors = [e for e in result.trace if e.type == "error"]
    assert errors and errors[0].data["category"] == "reflect_error"


async def test_a_check_that_crosses_the_budget_stops_and_is_counted() -> None:
    exc = BudgetExceeded(1.2, 1.0, call_cost=0.3, currency="CNY")
    exc.response = type("R", (), {"usage": CHECK_USAGE, "cost_usd": 0.3})()  # type: ignore[assignment]
    reflector = FixedReflector(exc)
    result = await ReflectingAgent(agent_answering("ANSWER: 7"), reflector).arun("q?")
    assert result.stop_reason == "budget_exceeded" and "budget" in (result.error or "")
    assert result.usage.prompt_tokens >= CHECK_USAGE.prompt_tokens
    assert requests(result) == 3


async def test_records_carry_first_and_final_answers() -> None:
    records: list[ReflectionRecord] = []
    inner = agent_answering("ANSWER: 5", "ANSWER: 7")
    reflector = FixedReflector(Verdict(False, "check b"))
    agent = ReflectingAgent(inner, reflector, on_record=records.append)
    await agent.arun("q?", session_id="t-01#0")
    (rec,) = records
    assert (rec.session_id, rec.first_answer) == ("t-01#0", "ANSWER: 5")
    assert rec.final_answer == "ANSWER: 7"
    assert rec.checked and rec.verdict_ok is False and rec.retried and rec.reason == "check b"


def test_max_retries_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ReflectingAgent(agent_answering(), FixedReflector(), max_retries=0)


# ------------------------------------------------------------------ LLMReflector
def finished_run() -> RunResult:
    call = ToolCall("c1", "lookup", {"key": "a"})
    messages = [
        Message.system("system prompt"),
        Message.user("What is a? End with ANSWER."),
        Message.assistant(None, (call,)),
        Message.tool("c1", "7"),
        Message.assistant("ANSWER: 7"),
    ]
    return RunResult("ANSWER: 7", messages, Usage(), 0.0)


async def test_the_checker_sees_task_answer_and_tool_steps_only() -> None:
    llm = ScriptedLLM([Message.assistant('{"ok": true, "reason": ""}')])
    verdict = await LLMReflector(llm).check("What is a? End with ANSWER.", finished_run())
    assert verdict.ok and verdict.parsed and verdict.calls == 1
    sent = llm.calls[0]
    text = "\n".join(m.content or "" for m in sent.messages)
    assert "What is a?" in text and "ANSWER: 7" in text
    assert 'lookup({"key": "a"})' in text and "-> 7" in text
    assert "system prompt" not in text  # the agent's own system prompt is not shown
    assert sent.temperature == 0.0 and sent.tag == "reflect"


@pytest.mark.parametrize(
    ("reply", "ok", "reason", "parsed"),
    [
        ('```json\n{"ok": false, "reason": "no discount"}\n```', False, "no discount", True),
        ('Verdict: {"ok": true}', True, "", True),
        ('{"ok": "false", "reason": "x"}', True, "", False),
        ("looks fine to me", True, "", False),
    ],
)
async def test_verdict_parsing(reply: str, ok: bool, reason: str, parsed: bool) -> None:
    llm = ScriptedLLM([Message.assistant(reply)])
    verdict = await LLMReflector(llm).check("q", finished_run())
    assert (verdict.ok, verdict.reason, verdict.parsed) == (ok, reason, parsed)


async def test_the_checker_cost_is_metered_under_its_tag() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 2.0, currency="CNY")}))
    llm = ScriptedLLM([Message.assistant('{"ok": true}')], meter=meter)
    verdict = await LLMReflector(llm).check("q", finished_run())
    assert verdict.cost_usd is not None and verdict.cost_usd > 0
    assert set(meter.by_tag) == {"reflect"}


async def test_reflection_with_a_real_agent_and_runner_counts_everything() -> None:
    from trellis_agent.eval import EvalRunner, parse_taskset

    doc = {"name": "r", "version": "1", "tasks": [
        {"id": "t1", "input": "q?", "scorers": [{"type": "contains", "value": "7"}]}]}  # fmt: skip
    taskset = parse_taskset(doc)
    checker = ScriptedLLM([Message.assistant('{"ok": false, "reason": "use key a"}')])

    def make() -> ReflectingAgent:
        return ReflectingAgent(agent_answering("ANSWER: 5", "ANSWER: 7"), LLMReflector(checker))

    report = await EvalRunner(make).run(taskset)
    (case,) = report.cases
    assert case.status == "passed" and case.model_requests == 5
    assert case.tool_calls == ["lookup", "lookup"]
