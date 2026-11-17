from __future__ import annotations

import json

import pytest

from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import LLMError, RuleLLM, ScriptedLLM
from trellis_agent.planner import TaskNode, WorkerSpec
from trellis_agent.planner.graph import PlanError
from trellis_agent.planner.planner import Planner
from trellis_agent.types import Message, Usage

WORKERS = [
    WorkerSpec("research", "looks facts up with search tools"),
    WorkerSpec("compute", "does arithmetic"),
]
GOOD = {
    "tasks": [
        {"id": "t1", "goal": "find the price", "kind": "research"},
        {"id": "t2", "goal": "multiply by 3", "kind": "compute", "depends_on": ["t1"]},
    ]
}


def reply(payload: object) -> Message:
    return Message.assistant(payload if isinstance(payload, str) else json.dumps(payload))


DEFAULT_USAGE = Usage(100, 20)


def scripted(
    *payloads: object, usage: Usage = DEFAULT_USAGE, meter: CostMeter | None = None
) -> ScriptedLLM:
    return ScriptedLLM([reply(p) for p in payloads], usage_per_call=usage, meter=meter)


def prompt_text(llm: ScriptedLLM, call: int = 0) -> str:
    return "\n".join(m.content or "" for m in llm.calls[call].messages)


# ----------------------------------------------------------------- planning
async def test_a_valid_plan_becomes_a_graph_with_accounting() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 1.0)}))
    llm = scripted(GOOD, usage=Usage(1_000_000, 0), meter=meter)
    plan = await Planner(llm, WORKERS).plan("compute the total")
    assert [(n.id, n.kind, n.depends_on) for n in plan.graph.nodes] == [
        ("t1", "research", ()),
        ("t2", "compute", ("t1",)),
    ]
    assert plan.attempts == 1 and plan.usage == Usage(1_000_000, 0)
    assert plan.cost_usd == pytest.approx(1.0) and meter.by_tag == {"plan": pytest.approx(1.0)}


async def test_the_prompt_names_the_goal_the_workers_and_the_output_format() -> None:
    llm = scripted(GOOD)
    await Planner(llm, WORKERS, max_nodes=7).plan("compute the total")
    text = prompt_text(llm)
    assert "compute the total" in text
    assert "research" in text and "looks facts up" in text and "compute" in text
    assert "JSON" in text and "depends_on" in text and "7" in text
    assert llm.calls[0].temperature == 0.0 and llm.calls[0].tag == "plan"


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here is the plan: " + json.dumps(GOOD) + " Good luck!",
        json.dumps(GOOD["tasks"]),  # a bare list is accepted too
    ],
)
async def test_tolerates_fences_chatter_and_bare_lists(raw: str) -> None:
    plan = await Planner(scripted(raw), WORKERS).plan("g")
    assert plan.graph.ids() == ["t1", "t2"]


# ----------------------------------------------------------------- repair loop
async def test_an_invalid_plan_is_sent_back_with_the_problems_and_repaired() -> None:
    cyclic = {
        "tasks": [
            {"id": "a", "goal": "x", "kind": "research", "depends_on": ["b"]},
            {"id": "b", "goal": "y", "kind": "research", "depends_on": ["a"]},
        ]
    }
    llm = scripted(cyclic, GOOD, usage=Usage(100, 20))
    plan = await Planner(llm, WORKERS).plan("g")
    assert plan.attempts == 2 and plan.graph.ids() == ["t1", "t2"]
    assert plan.usage == Usage(200, 40)  # the failed attempt was paid for too
    second = llm.calls[1].messages
    assert second[-2].role == "assistant"
    feedback = second[-1].content or ""
    assert second[-1].role == "user" and "cycle" in feedback and "JSON" in feedback
    assert json.dumps(cyclic) in (second[-2].content or "")  # it sees its own rejected answer


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ("I cannot do that", "JSON"),
        ({"tasks": []}, "at least one"),
        ({"tasks": [{"id": "a", "goal": "x", "kind": "wizard"}]}, "wizard"),
        (
            {"tasks": [{"id": "a", "goal": "x", "kind": "compute", "depends_on": ["ghost"]}]},
            "ghost",
        ),
        ({"plan": "no tasks key"}, "tasks"),
    ],
)
async def test_each_kind_of_bad_plan_is_explained_to_the_model(bad: object, fragment: str) -> None:
    llm = scripted(bad, GOOD)
    plan = await Planner(llm, WORKERS).plan("g")
    assert plan.attempts == 2
    assert fragment in (llm.calls[1].messages[-1].content or "")


async def test_too_many_tasks_is_a_repairable_problem() -> None:
    many = {"tasks": [{"id": f"t{i}", "goal": "x", "kind": "compute"} for i in range(6)]}
    llm = scripted(many, GOOD)
    plan = await Planner(llm, WORKERS, max_nodes=4).plan("g")
    assert "at most 4" in (llm.calls[1].messages[-1].content or "") and plan.attempts == 2


async def test_gives_up_after_the_repair_limit_and_reports_what_was_spent() -> None:
    llm = scripted("nope", "still nope", "never", GOOD, usage=Usage(100, 20))
    with pytest.raises(PlanError) as info:
        await Planner(llm, WORKERS, max_repair_attempts=2).plan("g")
    err = info.value
    assert len(llm.calls) == 3 and err.attempts == 3  # 1 try + 2 repairs
    assert err.usage == Usage(300, 60) and err.budget_exceeded is False
    assert err.problems and "JSON" in " ".join(err.problems)


async def test_no_repairs_when_the_limit_is_zero() -> None:
    llm = scripted("nope", GOOD)
    with pytest.raises(PlanError):
        await Planner(llm, WORKERS, max_repair_attempts=0).plan("g")
    assert len(llm.calls) == 1


# ----------------------------------------------------------------- model failures
async def test_a_failing_planner_model_is_a_plan_error_without_a_repair_loop() -> None:
    def fail(messages: list[Message], tools: object) -> Message:
        raise LLMError("provider down")

    llm = RuleLLM(fail)
    with pytest.raises(PlanError, match="provider down") as info:
        await Planner(llm, WORKERS).plan("g")
    assert info.value.usage == Usage() and info.value.budget_exceeded is False


async def test_budget_exceeded_while_planning_keeps_all_the_spend() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 1.0)}), budget_usd=1.5)
    llm = scripted("bad json", GOOD, usage=Usage(1_000_000, 0), meter=meter)
    with pytest.raises(PlanError) as info:
        await Planner(llm, WORKERS).plan("g")
    err = info.value
    assert err.budget_exceeded is True and len(llm.calls) == 2
    assert err.usage == Usage(2_000_000, 0)  # includes the call that crossed the budget
    assert err.cost_usd == pytest.approx(2.0) == meter.total_usd


# ----------------------------------------------------------------- replanning
async def test_replan_prompt_carries_the_failure_the_finished_work_and_what_is_left() -> None:
    llm = scripted(
        {"tasks": [{"id": "r1", "goal": "retry", "kind": "research", "depends_on": ["t1"]}]}
    )
    failed = TaskNode("t2", "multiply by 3", "compute", ("t1",))
    remaining = [failed, TaskNode("t3", "report", "research", ("t2",))]
    plan = await Planner(llm, WORKERS).replan(
        "the goal", failed, "ValueError: calculator exploded", {"t1": "price is 14"}, remaining
    )
    text = prompt_text(llm)
    for needle in ("the goal", "t2", "calculator exploded", "price is 14", "t3"):
        assert needle in text
    assert llm.calls[0].tag == "replan"
    assert plan.graph.ids() == ["r1"] and plan.graph.get("r1").depends_on == ("t1",)


async def test_replan_rejects_reusing_finished_ids_or_unknown_dependencies() -> None:
    def one(task_id: str, dep: str | None = None) -> dict[str, object]:
        task: dict[str, object] = {"id": task_id, "goal": "x", "kind": "compute"}
        if dep:
            task["depends_on"] = [dep]
        return {"tasks": [task]}

    llm = scripted(one("t1"), one("r1", "zzz"), one("r2", "t1"))
    failed = TaskNode("t2", "g", "compute")
    plan = await Planner(llm, WORKERS).replan("g", failed, "err", {"t1": "out"}, [failed])
    assert plan.attempts == 3 and plan.graph.ids() == ["r2"]
    assert "duplicate" in (llm.calls[1].messages[-1].content or "")
    assert "zzz" in (llm.calls[2].messages[-1].content or "")


async def test_long_outputs_are_truncated_in_the_replan_prompt() -> None:
    llm = scripted({"tasks": [{"id": "r1", "goal": "x", "kind": "compute"}]})
    failed = TaskNode("t2", "g", "compute")
    await Planner(llm, WORKERS).replan("g", failed, "err", {"t1": "Z" * 5000}, [failed])
    text = prompt_text(llm)
    assert "Z" * 100 in text and "Z" * 2000 not in text and "truncated" in text


def test_a_planner_needs_workers_to_assign_tasks_to() -> None:
    with pytest.raises(ValueError, match="worker"):
        Planner(scripted(GOOD), [])
