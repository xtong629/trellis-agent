"""Task set C2 (v0.2 latency/fault experiment): data, answers, scoring of reply shapes, the fault
schedule, the slow tools, and offline wiring of both systems (mock models; machinery only)."""

from __future__ import annotations

import time
from typing import Any

import pytest
from experiments import v02_run
from experiments.c2_tasks import CELLS, MAX_FAILS_IN_A_ROW, Cell, fails, make_c2_tools, taskset_c2
from experiments.targets import c2_planned, c2_single

from trellis_agent.eval import EvalRunner, ScorerSpec, build_scorer, parse_taskset, scorer_types
from trellis_agent.llm import RuleLLM
from trellis_agent.tools import ToolRegistry
from trellis_agent.types import Message, RunResult, ToolCall, Usage


def test_shape_answers_and_validation() -> None:
    doc = taskset_c2()
    assert (doc["name"], doc["version"], len(doc["tasks"])) == ("exp-c2-latency", "1", 8)
    answers = {t["id"]: t["metadata"]["answer"] for t in doc["tasks"]}
    assert answers["latency-01"] == 14 + 9 + 0  # lantern at Arden, Bexley, Corrin
    assert answers["latency-02"] == 6 + 18 + 11 + 4  # kettle at all four depots
    assert {t["metadata"]["lookups"] for t in doc["tasks"]} == {3, 4}
    for t in doc["tasks"]:
        assert t["input"].endswith("ANSWER: <number>'.")
        assert {s["type"] for s in t["scorers"]} == {"number_match", "tool_called"}
    parse_taskset(doc, known_scorers=scorer_types())


def _result(text: str) -> RunResult:
    return RunResult(text, [Message.assistant(text)], Usage(), 0.0)


async def test_golden_reply_shapes() -> None:
    for task in taskset_c2()["tasks"]:
        spec = next(s for s in task["scorers"] if s["type"] == "number_match")
        scorer = build_scorer(
            ScorerSpec("number_match", {k: v for k, v in spec.items() if k != "type"})
        )
        v = task["metadata"]["answer"]
        for text in (f"ANSWER: {v}", f"Arden 14, Bexley 9.\nTotal {v}.\nANSWER: {v}",
                     f"**ANSWER: {v}**", f"answer: {v} units", f"ANSWER: {v}.00"):  # fmt: skip
            assert (await scorer.score(None, _result(text))).passed, (task["id"], text)
        for text in (f"ANSWER: {v + 1}", f"Total {v}\nANSWER: {v - 3}", f"The total is {v}."):
            assert not (await scorer.score(None, _result(text))).passed, (task["id"], text)


def test_the_fault_schedule_is_deterministic_bounded_and_near_its_rate() -> None:
    cell = Cell(0.0, 0.3)
    args = [{"depot": d, "product": f"p{i}"} for d in "ABCD" for i in range(100)]
    first = [fails(cell, "get_units", a, 1) for a in args]
    assert first == [fails(cell, "get_units", a, 1) for a in reversed(args)][::-1]
    assert 0.2 < sum(first) / len(first) < 0.4
    assert not any(fails(cell, "get_units", a, MAX_FAILS_IN_A_ROW + 1) for a in args)
    assert fails(cell, "get_units", {"depot": " arden", "product": "Kettle"}, 1) == fails(
        cell, "get_units", {"depot": "Arden", "product": "kettle"}, 1
    )
    assert not any(fails(Cell(0.0, 0.0), "get_units", a, 1) for a in args)


def test_the_four_cells() -> None:
    assert [c.label for c in CELLS] == ["l1", "l4", "l10", "l4f0.3"]
    assert sorted(v02_run.EXPERIMENTS["c2"].systems) == sorted(
        f"c2_{s}_{c.label}" for c in CELLS for s in ("single", "planned")
    )
    assert v02_run.EXPERIMENTS["c2"].concurrency == 1 and v02_run.EXPERIMENTS["c2"].timing


async def test_slow_tools_wait_and_fail_as_scheduled() -> None:
    cell = Cell(0.05, 1.0)  # every first and second identical call fails
    registry = ToolRegistry(make_c2_tools(cell).all)
    call = ToolCall("x", "get_units", {"depot": "Arden", "product": "kettle"})
    started = time.perf_counter()
    outcomes = [await registry.call(call) for _ in range(3)]
    assert time.perf_counter() - started >= 0.15
    assert [o.is_error for o in outcomes] == [True, True, False]
    assert "busy" in outcomes[0].content and outcomes[2].content == "6"


def _mock() -> Any:
    return RuleLLM(lambda m, t: Message.assistant("unused"), model="mock")


@pytest.mark.parametrize("make", [c2_single, c2_planned])
async def test_both_systems_solve_the_set_offline_under_faults(make: Any) -> None:
    taskset = parse_taskset(taskset_c2(), known_scorers=scorer_types())
    report = await EvalRunner(lambda: make(_mock(), Cell(0.0, 0.3))).run(taskset)
    assert [c.status for c in report.cases] == ["passed"] * 8, [c.content for c in report.cases]


async def test_the_planner_overlaps_tool_latency_and_the_single_agent_does_not() -> None:
    doc = taskset_c2()
    taskset = parse_taskset({**doc, "tasks": doc["tasks"][1:2]})  # four lookups
    cell = Cell(0.2, 0.0)
    timings = {}
    for make in (c2_single, c2_planned):
        report = await EvalRunner(lambda make=make: make(_mock(), cell), include_timing=True).run(
            taskset
        )
        timings[make.__name__] = next(iter((report.timing or {}).values()))
    assert timings["c2_single"] >= 4 * 200
    assert timings["c2_planned"] < 0.6 * timings["c2_single"]


def test_merged_timing_renumbers_repeats() -> None:
    class Chunk:
        def __init__(self, timing: dict[str, float] | None) -> None:
            self.timing = timing

    chunks = [Chunk({"t1#0": 5.0, "t2#0": 6.0}), Chunk({"t1#0": 7.0, "t2#0": 8.0})]
    merged = v02_run.merged_timing(chunks)  # type: ignore[arg-type]
    assert merged == {"t1#0": 5.0, "t2#0": 6.0, "t1#1": 7.0, "t2#1": 8.0}
    assert v02_run.merged_timing([Chunk(None)]) is None  # type: ignore[list-item]
