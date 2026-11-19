"""The v0.2 driver offline: chunk order, stop rules, the freeze check, and the C1 wiring with a
mock model (proves the machinery only; mock results say nothing about real models)."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
from typing import Any

import pytest
from experiments import v02_run
from experiments.c1_tasks import taskset_c1
from experiments.targets import c1_baseline

from trellis_agent.eval import EvalReport, EvalRunner, parse_taskset, scorer_types
from trellis_agent.llm import RuleLLM
from trellis_agent.types import Message


def test_chunks_are_repeat_major_and_alternate_systems() -> None:
    plan = v02_run.chunk_plan(2, ["s1", "s2"], {(0, "s1")})
    assert plan == [(0, "s2"), (1, "s1"), (1, "s2")]


def _report(**case: Any) -> EvalReport:
    """A one-case report from a real offline run, with the case fields replaced."""
    llm = RuleLLM(lambda m, t: Message.assistant("unused"), model="mock")
    doc = taskset_c1(1)
    taskset = parse_taskset({**doc, "tasks": doc["tasks"][:1]}, known_scorers=scorer_types())
    report = asyncio.run(EvalRunner(lambda: c1_baseline(llm)).run(taskset))
    return dataclasses.replace(report, cases=[dataclasses.replace(report.cases[0], **case)])


OK = [{"status": 200, "latency_s": 1.0}]


def test_a_failed_case_is_a_result_but_a_provider_error_stops_the_run() -> None:
    wrong = _report(status="failed")
    steps = _report(error="max_steps (12) reached", stop_reason="max_steps")
    provider = _report(error="LLMError: boom", stop_reason="llm_error")
    assert v02_run.stop_reason(OK, 0.1, 5, 10, 100, wrong) is None
    assert v02_run.stop_reason(OK, 0.1, 5, 10, 100, steps) is None
    assert "provider" in (v02_run.stop_reason(OK, 0.1, 5, 10, 100, provider) or "")


def test_http_budget_and_token_stops() -> None:
    report = _report()

    def why(log: list[dict[str, Any]], spent: float, tokens: int) -> str:
        return v02_run.stop_reason(log, spent, 5, tokens, 100, report) or ""

    assert "429" in why([{"status": 429, "latency_s": 1}], 0, 0)
    assert "budget" in why(OK, 4.0, 0)
    assert "token" in why(OK, 0.0, 80)
    assert "slower" in why([{"status": 200, "latency_s": 61}], 0, 0)


def test_full_runs_need_the_frozen_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    exp = v02_run.EXPERIMENTS["c1"]
    args = argparse.Namespace(trial=False, level=1)
    monkeypatch.setattr(v02_run, "frozen_entry", lambda key: None)
    with pytest.raises(SystemExit, match="not frozen"):
        v02_run.load_taskset(exp, args)
    monkeypatch.setattr(v02_run, "frozen_entry", lambda key: {"level": 1, "content_hash": "x"})
    with pytest.raises(SystemExit, match="differs"):
        v02_run.load_taskset(exp, args)
    good = parse_taskset(taskset_c1(1)).content_hash
    monkeypatch.setattr(v02_run, "frozen_entry", lambda key: {"level": 1, "content_hash": good})
    assert v02_run.load_taskset(exp, args)[1] == 1
    trial = v02_run.load_taskset(exp, argparse.Namespace(trial=True, level=2))
    assert trial[1] == 2 and trial[0].version == "1-level2"


async def test_c1_baseline_runs_offline_with_the_mock() -> None:
    llm = RuleLLM(lambda m, t: Message.assistant("unused"), model="mock")
    taskset = parse_taskset(taskset_c1(1), known_scorers=scorer_types())
    report = await EvalRunner(lambda: c1_baseline(llm), concurrency=2).run(taskset)
    assert len(report.cases) == 24 and not report.aborted_by_budget
    rates = v02_run.family_rates(report)
    assert rates["easy"] == "6/6"  # the wiring mock only knows how to answer the control family
    assert all(c.tool_calls == ["get_item"] for c in report.cases if c.task_id.startswith("easy"))


async def test_c1_reflect_runs_offline_and_records_every_run() -> None:
    from dataclasses import asdict

    from experiments.targets import c1_reflect

    llm = RuleLLM(lambda m, t: Message.assistant("unused"), model="mock")
    taskset = parse_taskset(taskset_c1(2), known_scorers=scorer_types())
    records: list[Any] = []
    report = await EvalRunner(lambda: c1_reflect(llm, records.append), concurrency=2).run(taskset)
    assert v02_run.family_rates(report)["easy"] == "6/6"
    assert all(c.model_requests == 3 for c in report.cases)  # lookup, answer, one check
    sidecar = await v02_run.attempts_report(taskset, [[asdict(r) for r in records]])
    totals = sidecar["totals"]
    assert (totals["runs"], totals["checked"], totals["retried"]) == (24, 24, 0)
    assert sidecar["by_family"]["easy"]["first_correct"] == 6 and totals["harm"] == 0


def _record(task_id: str, first: str, final: str, ok: bool | None) -> dict[str, Any]:
    return {"session_id": f"{task_id}#0", "query": "q", "first_answer": first,
            "first_stop_reason": "end_turn", "checked": ok is not None, "verdict_ok": ok,
            "reason": "", "parsed": None if ok is None else True, "retried": ok is False,
            "final_answer": final, "final_stop_reason": "end_turn"}  # fmt: skip


async def test_attempt_counts_follow_the_pre_registered_definitions() -> None:
    taskset = parse_taskset(taskset_c1(2), known_scorers=scorer_types())
    right = {t.id: t.metadata["answer"] for t in taskset.tasks}
    records = [
        _record("easy-01", "ANSWER: 1", f"ANSWER: {right['easy-01']}", False),  # fix
        _record("easy-02", f"ANSWER: {right['easy-02']}", "ANSWER: 1", False),  # harm, false reject
        _record("easy-03", "ANSWER: 1", "ANSWER: 1", True),  # false accept
        _record("easy-04", f"ANSWER: {right['easy-04']}", f"ANSWER: {right['easy-04']}", True),
        _record("deep-01", "", "", None),  # not checked (no final answer)
    ]
    totals = (await v02_run.attempts_report(taskset, [records]))["totals"]
    assert totals == {"runs": 5, "checked": 4, "rejected": 2, "retried": 2,
                      "unparsed_verdicts": 0, "first_correct": 2, "final_correct": 2, "fix": 1,
                      "harm": 1, "false_accept": 1, "false_reject": 1}  # fmt: skip
