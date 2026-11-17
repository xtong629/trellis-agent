"""PlannedAgent end to end, and the multi-agent collaboration acceptance test."""

from __future__ import annotations

import json

import pytest

from tests.unit.vclock import FakeWorker
from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.eval import EvalRunner, parse_taskset
from trellis_agent.llm import LLMError, RuleLLM, ScriptedLLM
from trellis_agent.planner import Planner, RetryPolicy, Scheduler, WorkerSpec
from trellis_agent.planner.planned import PlannedAgent
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.trace import summarize
from trellis_agent.types import Message, ToolCall, Usage

CALL_USAGE = Usage(10, 5)
PRICES = PriceTable({"mock": ModelPrice(1_000_000.0, 1_000_000.0)})  # cost == tokens, easy to check

PLAN = {
    "tasks": [
        {"id": "t1", "goal": "look up the unit price of a widget", "kind": "research"},
        {
            "id": "t2",
            "goal": "multiply the unit price by 3",
            "kind": "compute",
            "depends_on": ["t1"],
        },
    ]
}


@tool
def lookup_price(item: str) -> str:
    """Look up the unit price of an item."""
    return "14"


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def llm(*replies: Message, meter: CostMeter | None = None) -> ScriptedLLM:
    return ScriptedLLM(list(replies), usage_per_call=CALL_USAGE, meter=meter)


def call(name: str, **args: object) -> Message:
    return Message.assistant(None, (ToolCall(f"c-{name}", name, dict(args)),))


def build(meter: CostMeter | None = None):
    planner_llm = llm(Message.assistant(json.dumps(PLAN)), meter=meter)
    researcher_llm = llm(
        call("lookup_price", item="widget"), Message.assistant("price is 14"), meter=meter
    )
    calculator_llm = llm(
        call("multiply", a=14, b=3), Message.assistant("the total is 42"), meter=meter
    )
    synth_llm = llm(Message.assistant("Three widgets cost 42."), meter=meter)
    researcher = Agent(
        researcher_llm,
        tools=ToolRegistry([lookup_price]),
        config=AgentConfig(system_prompt="You are a RESEARCHER. Find facts with your tools."),
    )
    calculator = Agent(
        calculator_llm,
        tools=ToolRegistry([multiply]),
        config=AgentConfig(system_prompt="You are a CALCULATOR. Do arithmetic precisely."),
    )
    specs = [
        WorkerSpec("research", "finds facts and prices"),
        WorkerSpec("compute", "does arithmetic"),
    ]
    planned = PlannedAgent(
        Planner(planner_llm, specs),
        Scheduler({"research": researcher, "compute": calculator}),
        synth_llm,
    )
    return planned, planner_llm, researcher_llm, calculator_llm, synth_llm


# ----------------------------------------------------------------- multi-agent collaboration
async def test_two_differently_configured_workers_each_handle_only_their_own_task_type() -> None:
    planned, planner_llm, researcher_llm, calculator_llm, _ = build()
    result = await planned.arun("What do 3 widgets cost?")

    assert result.content == "Three widgets cost 42." and result.error is None
    assert result.stop_reason == "end_turn"

    # The planner was told what each worker is good at.
    planner_text = "\n".join(m.content or "" for m in planner_llm.calls[0].messages)
    assert "research: finds facts" in planner_text and "compute: does arithmetic" in planner_text

    # Researcher: its own system prompt and tool set, and only the research task.
    r_first = researcher_llm.calls[0].messages
    assert "RESEARCHER" in (r_first[0].content or "") and "CALCULATOR" not in (
        r_first[0].content or ""
    )
    assert [s.name for s in researcher_llm.calls[0].tools or ()] == ["lookup_price"]
    r_user = r_first[-1].content or ""
    assert "unit price of a widget" in r_user and "multiply" not in r_user
    assert len(researcher_llm.calls) == 2  # one tool call + one answer: it never saw task t2

    # Calculator: different system prompt and tools, and it received the researcher's output.
    c_first = calculator_llm.calls[0].messages
    assert "CALCULATOR" in (c_first[0].content or "") and "RESEARCHER" not in (
        c_first[0].content or ""
    )
    assert [s.name for s in calculator_llm.calls[0].tools or ()] == ["multiply"]
    c_user = c_first[-1].content or ""
    assert "multiply the unit price by 3" in c_user and "[t1] price is 14" in c_user
    assert "unit price of a widget" not in c_user.split("Your task")[1]  # only its own task
    assert len(calculator_llm.calls) == 2

    # Tool use of both workers is visible in the merged trace (so tool_called scorers work).
    called = [e.data["name"] for e in result.trace if e.type == "tool_call"]
    assert sorted(called) == ["lookup_price", "multiply"]


async def test_usage_and_cost_cover_planning_workers_and_synthesis_and_match_the_trace() -> None:
    meter = CostMeter(PRICES)
    planned, *_ = build(meter)
    result = await planned.arun("What do 3 widgets cost?")
    assert result.usage == Usage(60, 30)  # six model calls: plan, 2 + 2 worker calls, synthesis
    assert result.cost_usd == pytest.approx(meter.total_usd) == pytest.approx(90.0)
    s = summarize(result.trace)
    assert s.usage == result.usage and s.cost_usd == pytest.approx(result.cost_usd)
    assert meter.by_tag["plan"] == pytest.approx(15.0) and meter.by_tag[
        "synthesize"
    ] == pytest.approx(15.0)


async def test_it_is_a_drop_in_target_for_the_evaluation_harness() -> None:
    planned, *_ = build()
    taskset = parse_taskset(
        {
            "name": "planned",
            "version": "1",
            "tasks": [
                {
                    "id": "cost-of-three",
                    "input": "What do 3 widgets cost?",
                    "scorers": [
                        {"type": "number_match", "value": 42},
                        {"type": "tool_called", "name": "multiply"},
                        {"type": "tool_called", "name": "lookup_price"},
                    ],
                }
            ],
        }
    )
    report = await EvalRunner(lambda: planned).run(taskset)
    assert report.cases[0].status == "passed", [s.reason for s in report.cases[0].scores]


# --- failures are reported, not raised
async def test_a_plan_that_cannot_be_repaired_fails_cleanly_and_runs_no_workers() -> None:
    planner_llm = llm(*[Message.assistant("not json")] * 3)
    worker = FakeWorker("w")
    planned = PlannedAgent(
        Planner(planner_llm, [WorkerSpec("research")]), Scheduler({"research": worker}), llm()
    )
    result = await planned.arun("goal")
    assert result.stop_reason == "plan_error" and "JSON" in (result.error or "")
    assert result.content == "" and worker.calls == []
    assert result.usage == Usage(30, 15)  # three planning attempts were paid for


async def test_a_failed_task_is_reported_and_synthesis_is_skipped() -> None:
    plan = {"tasks": [{"id": "t1", "goal": "do it", "kind": "research"}]}
    broken = RuleLLM(lambda m, t: (_ for _ in ()).throw(LLMError("provider down")))
    worker = Agent(broken)
    synth = llm(Message.assistant("never"))
    planned = PlannedAgent(
        Planner(llm(Message.assistant(json.dumps(plan))), [WorkerSpec("research")]),
        Scheduler({"research": worker}, retry=RetryPolicy(base_delay=0.0, jitter=0.0)),
        synth,
    )
    result = await planned.arun("goal")
    assert result.stop_reason == "tasks_failed" and result.content == ""
    assert "t1" in (result.error or "") and "provider down" in (result.error or "")
    assert synth.calls == []


async def test_running_out_of_budget_during_synthesis_still_counts_the_final_call() -> None:
    meter = CostMeter(PRICES, budget_usd=80.0)  # plan 15 + workers 60 = 75, synthesis crosses 80
    planned, *_ = build(meter)
    result = await planned.arun("What do 3 widgets cost?")
    assert result.stop_reason == "budget_exceeded" and result.content == ""
    assert result.usage == Usage(60, 30)
    assert result.cost_usd == pytest.approx(meter.total_usd)


async def test_budget_exhausted_during_planning_is_reported() -> None:
    meter = CostMeter(PRICES, budget_usd=1.0)
    planned, *_ = build(meter)
    result = await planned.arun("goal")
    assert result.stop_reason == "budget_exceeded"
    assert result.usage == CALL_USAGE and result.cost_usd == pytest.approx(meter.total_usd)


async def test_replanning_flows_through_the_real_planner() -> None:
    first = {"tasks": [{"id": "t1", "goal": "try the broken tool", "kind": "research"}]}
    second = {"tasks": [{"id": "r1", "goal": "do it another way", "kind": "research"}]}
    planner_llm = llm(Message.assistant(json.dumps(first)), Message.assistant(json.dumps(second)))
    worker = FakeWorker("w", script=[ValueError("tool broke"), None])
    planner = Planner(planner_llm, [WorkerSpec("research")])
    scheduler = Scheduler({"research": worker}, on_failure="replan", planner=planner)
    planned = PlannedAgent(planner, scheduler, llm(Message.assistant("recovered answer")))
    result = await planned.arun("goal")
    assert result.content == "recovered answer" and result.stop_reason == "end_turn"
    assert [c.tag for c in planner_llm.calls] == ["plan", "replan"]
    assert "tool broke" in "\n".join(m.content or "" for m in planner_llm.calls[1].messages)
    assert any(e.type == "replan" for e in result.trace)
    assert result.usage.prompt_tokens >= 30  # planning twice + the worker + synthesis
