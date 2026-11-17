import asyncio
import itertools
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from trellis_agent.agent import Agent
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.eval.runner import EvalRunner
from trellis_agent.eval.scorers import ScorerConfigError
from trellis_agent.eval.taskset import EvalTask, ScorerSpec, TaskSet, parse_taskset
from trellis_agent.llm import RuleLLM, ScriptedLLM
from trellis_agent.types import Message, RunResult, TraceEvent, Usage


def taskset(*tasks: dict[str, Any]) -> TaskSet:
    return parse_taskset({"name": "demo", "version": "1", "tasks": list(tasks)})


def task(id_: str, input_: str = "q", *scorers: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"id": id_, "input": input_, "scorers": list(scorers), **extra}


def result(content: str = "ok", **kw: Any) -> RunResult:
    kw.setdefault("usage", Usage(10, 5))
    kw.setdefault("cost", None)
    trace = [TraceEvent("model_request", 0.0, {}, "root", None)]
    return RunResult(
        content,
        [],
        kw["usage"],
        kw["cost"],
        trace,
        kw.get("error"),
        kw.get("stop_reason", "end_turn"),
    )


class FakeTarget:
    def __init__(self, fn: Callable[[str, str], Awaitable[RunResult] | RunResult]) -> None:
        self.fn = fn
        self.sessions: list[str] = []

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        self.sessions.append(session_id)
        out = self.fn(query, session_id)
        return await out if asyncio.iscoroutine(out) else out  # type: ignore[no-any-return]


def factory_of(fn: Callable[[str, str], Any]) -> tuple[Callable[[], FakeTarget], list[FakeTarget]]:
    made: list[FakeTarget] = []

    def make() -> FakeTarget:
        made.append(FakeTarget(fn))
        return made[-1]

    return make, made


# ----------------------------------------------------------------- basic scoring
async def test_passing_and_failing_cases_are_scored() -> None:
    ts = taskset(
        task("good", "say hi", {"type": "contains", "value": "hi"}),
        task("bad", "say bye", {"type": "contains", "value": "bye"}),
    )
    make, _ = factory_of(lambda q, s: result("hi there"))
    report = await EvalRunner(make).run(ts)
    good, bad = report.cases
    assert (good.task_id, good.status) == ("good", "passed")
    assert (bad.task_id, bad.status) == ("bad", "failed")
    assert [s.name for s in good.scores] == ["run_succeeded", "contains('hi')"]
    failed_score = next(s for s in bad.scores if not s.passed)
    assert "bye" in failed_score.reason
    s = report.summary()
    assert (s["passed"], s["failed"], s["pass_rate"]) == (1, 1, 0.5)
    assert report.taskset_hash == ts.content_hash and report.taskset_name == "demo"


async def test_run_error_fails_the_case_unless_an_error_is_expected() -> None:
    ts = taskset(task("boom"), task("refusal", expect_success=False))
    make, _ = factory_of(
        lambda q, s: result("", error="max_steps reached", stop_reason="max_steps")
    )
    report = await EvalRunner(make).run(ts)
    boom, refusal = report.cases
    assert boom.status == "failed" and "max_steps reached" in boom.scores[0].reason
    assert refusal.status == "passed" and refusal.scores[0].name == "run_fails_as_expected"
    assert boom.error == "max_steps reached" and boom.stop_reason == "max_steps"


async def test_a_crashing_target_is_a_failed_case_not_a_crashed_evaluation() -> None:
    def fn(q: str, s: str) -> RunResult:
        if q == "explode":
            raise RuntimeError("kaput")
        return result("fine")

    make, _ = factory_of(fn)
    report = await EvalRunner(make).run(taskset(task("a", "explode"), task("b", "ok")))
    a, b = report.cases
    assert a.status == "failed" and a.stop_reason == "target_error" and "kaput" in (a.error or "")
    assert b.status == "passed"


async def test_trace_derived_facts_are_recorded() -> None:
    trace = [
        TraceEvent("model_request", 0.0, {}, "root", None),
        TraceEvent("tool_call", 0.0, {"name": "add"}, "root", None),
        TraceEvent("model_request", 0.0, {}, "root", None),
    ]
    make, _ = factory_of(lambda q, s: RunResult("x", [], Usage(1, 1), None, trace))
    (case,) = (await EvalRunner(make).run(taskset(task("t")))).cases
    assert case.model_requests == 2 and case.tool_calls == ["add"]


async def test_long_content_is_capped_in_the_report() -> None:
    make, _ = factory_of(lambda q, s: result("x" * 5000))
    (case,) = (await EvalRunner(make).run(taskset(task("t")))).cases
    assert len(case.content) < 2100 and case.content.endswith("[truncated]")


# ----------------------------------------------------------------- repeats, isolation, determinism
async def test_repeats_use_a_fresh_target_and_unique_sessions_per_job() -> None:
    make, made = factory_of(lambda q, s: result("ok"))
    report = await EvalRunner(make, repeats=3, concurrency=1).run(taskset(task("a"), task("b")))
    assert len(made) == 6
    assert [(c.task_id, c.repeat) for c in report.cases] == [
        ("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("b", 2),
    ]  # fmt: skip
    sessions = [s for t in made for s in t.sessions]
    assert len(set(sessions)) == 6 and sessions[0] == "a#0"


async def test_pass_rate_mean_and_spread_across_repeats() -> None:
    counter = itertools.count()
    # repeat 0: both pass; repeat 1: one fails; repeat 2: both fail (jobs are issued repeat-major)
    outcomes = [True, True, True, False, False, False]
    make, _ = factory_of(lambda q, s: result("yes" if outcomes[next(counter)] else "no"))
    ts = taskset(task("a", "q", {"type": "contains", "value": "yes"}),
                 task("b", "q", {"type": "contains", "value": "yes"}))  # fmt: skip
    summary = (await EvalRunner(make, repeats=3, concurrency=1).run(ts)).summary()
    assert summary["pass_rate_by_repeat"] == [1.0, 0.5, 0.0]
    assert summary["pass_rate_mean"] == pytest.approx(0.5)
    assert summary["pass_rate_std"] == pytest.approx(0.5)  # sample std (n-1)
    assert summary["pass_rate"] == pytest.approx(0.5)
    assert {t["task_id"]: t["pass_rate"] for t in summary["tasks"]} == {
        "a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3),
    }  # fmt: skip


async def test_single_repeat_has_no_spread() -> None:
    make, _ = factory_of(lambda q, s: result("ok"))
    summary = (await EvalRunner(make).run(taskset(task("a")))).summary()
    assert summary["pass_rate_mean"] == 1.0 and summary["pass_rate_std"] is None


async def test_jobs_really_run_concurrently_but_the_report_order_is_stable() -> None:
    arrived = 0
    gate = asyncio.Event()

    async def fn(q: str, s: str) -> RunResult:
        nonlocal arrived
        arrived += 1
        if arrived == 3:
            gate.set()
        await asyncio.wait_for(gate.wait(), timeout=2)  # deadlocks if jobs were serial
        if q == "a":
            await asyncio.sleep(0.02)  # the first job finishes last
        return result(q)

    make, _ = factory_of(fn)
    ts = taskset(task("a", "a"), task("b", "b"), task("c", "c"))
    report = await EvalRunner(make, concurrency=3).run(ts)
    assert [c.task_id for c in report.cases] == ["a", "b", "c"]
    assert [c.content for c in report.cases] == ["a", "b", "c"]


async def test_concurrency_limit_is_respected() -> None:
    running = peak = 0

    async def fn(q: str, s: str) -> RunResult:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.005)
        running -= 1
        return result()

    make, _ = factory_of(fn)
    await EvalRunner(make, concurrency=2).run(taskset(*(task(f"t{i}") for i in range(6))))
    assert peak == 2


async def test_two_runs_of_the_same_taskset_are_identical() -> None:
    ts = taskset(task("a", "q", {"type": "contains", "value": "ok"}), task("b"))
    make, _ = factory_of(lambda q, s: result("ok"))
    first = (await EvalRunner(make, repeats=2, concurrency=4).run(ts)).to_dict()
    second = (await EvalRunner(make, repeats=2, concurrency=4).run(ts)).to_dict()
    assert first == second


async def test_timing_is_opt_in_and_uses_the_injected_clock() -> None:
    make, _ = factory_of(lambda q, s: result())
    ts = taskset(task("a"))
    assert (await EvalRunner(make).run(ts)).timing is None
    ticks = itertools.count(0, 5)  # each clock() call advances 5 seconds
    report = await EvalRunner(make, include_timing=True, clock=lambda: float(next(ticks))).run(ts)
    assert report.timing == {"a#0": 5000.0}


# ----------------------------------------------------------------- configuration
async def test_bad_scorer_config_fails_before_any_job_runs() -> None:
    make, made = factory_of(lambda q, s: result())
    ts = taskset(task("a", "q", {"type": "contains"}))  # missing value
    with pytest.raises(ScorerConfigError, match="value"):
        await EvalRunner(make).run(ts)
    ts2 = taskset(task("a", "q", {"type": "llm_judge", "rubric": "r"}))
    with pytest.raises(ScorerConfigError, match="judge"):
        await EvalRunner(make).run(ts2)  # no judge model configured
    assert made == []


def test_invalid_runner_arguments() -> None:
    make, _ = factory_of(lambda q, s: result())
    for kw in ({"repeats": 0}, {"concurrency": 0}, {"budget_usd": -1.0}):
        with pytest.raises(ValueError):
            EvalRunner(make, **kw)


async def test_config_is_recorded_in_the_report() -> None:
    make, _ = factory_of(lambda q, s: result())
    runner = EvalRunner(make, repeats=2, concurrency=3, budget_usd=1.5, config={"model": "m"})
    report = await runner.run(taskset(task("a")))
    assert report.config == {
        "repeats": 2,
        "concurrency": 3,
        "budget": {"amount": 1.5, "currency": None},  # no meter: the currency is not known
        "model": "m",
    }


# --- cost: agent vs judge, reconciliation
PRICES = PriceTable({"agent-m": ModelPrice(1.0, 1.0), "judge-m": ModelPrice(2.0, 2.0)})


def agent_and_judge(meter: CostMeter, *, judge_reply: str = '{"score": 0.9, "reason": "good"}'):
    def make() -> Agent:
        llm = ScriptedLLM(
            [Message.assistant("an answer")],
            model="agent-m",
            usage_per_call=Usage(1_000_000, 0),
            meter=meter,
        )
        return Agent(llm)

    judge = ScriptedLLM(
        [Message.assistant(judge_reply)] * 10,
        model="judge-m",
        usage_per_call=Usage(500_000, 0),
        meter=meter,
    )
    return make, judge


async def test_judge_tokens_and_cost_are_reported_separately_and_reconcile_with_the_meter() -> None:
    meter = CostMeter(PRICES)
    make, judge = agent_and_judge(meter)
    ts = taskset(task("a", "q", {"type": "llm_judge", "rubric": "good answer"}))
    report = await EvalRunner(make, judge_llm=judge, meter=meter).run(ts)
    (case,) = report.cases
    assert case.agent_cost_usd == pytest.approx(1.0)
    assert case.judge_cost_usd == pytest.approx(1.0)  # 500k tokens at $2/M
    assert case.usage == Usage(1_000_000, 0) and case.judge_usage == Usage(500_000, 0)
    assert case.cost_usd == pytest.approx(2.0)
    s = report.summary()
    assert s["agent_cost"] == pytest.approx(1.0) and s["judge_cost"] == pytest.approx(1.0)
    assert s["cost"] == pytest.approx(2.0) and s["cost_complete"] is True
    assert s["usage"]["prompt_tokens"] == 1_000_000 and s["judge_usage"]["prompt_tokens"] == 500_000
    assert s["cost_per_success"] == pytest.approx(2.0)
    assert report.meter_total_usd == pytest.approx(meter.total_usd) == pytest.approx(2.0)
    assert s["meter_discrepancy"] == pytest.approx(0.0)
    assert meter.by_tag["judge"] == pytest.approx(1.0)


async def test_cost_per_success_divides_total_cost_by_passes() -> None:
    meter = CostMeter(PRICES)
    make, _ = agent_and_judge(meter)
    ts = taskset(
        task("ok", "q", {"type": "contains", "value": "answer"}),
        task("no", "q", {"type": "contains", "value": "zzz"}),
    )
    s = (await EvalRunner(make, meter=meter).run(ts)).summary()
    assert s["cost"] == pytest.approx(2.0) and s["passed"] == 1
    assert s["cost_per_success"] == pytest.approx(2.0)


async def test_no_passes_means_no_cost_per_success() -> None:
    meter = CostMeter(PRICES)
    make, _ = agent_and_judge(meter)
    s = (
        await EvalRunner(make).run(taskset(task("a", "q", {"type": "contains", "value": "zzz"})))
    ).summary()
    assert s["passed"] == 0 and s["cost_per_success"] is None


async def test_unpriced_models_are_flagged_not_counted_as_free() -> None:
    llm = RuleLLM(lambda m, t: Message.assistant("x"))  # no meter, so no cost
    s = (await EvalRunner(lambda: Agent(llm)).run(taskset(task("a")))).summary()
    assert s["cost"] is None and s["cost_complete"] is False


async def test_judge_is_skipped_when_the_run_itself_failed() -> None:
    meter = CostMeter(PRICES)
    _, judge = agent_and_judge(meter)
    make, _ = factory_of(lambda q, s: result("", error="boom", stop_reason="llm_error"))
    ts = taskset(task("a", "q", {"type": "llm_judge", "rubric": "r"}))
    (case,) = (await EvalRunner(make, judge_llm=judge, meter=meter).run(ts)).cases
    assert case.status == "failed" and len(judge.calls) == 0
    skipped = case.scores[-1]
    assert skipped.judge and not skipped.passed and "skipped" in skipped.reason
    assert case.judge_cost_usd == 0.0 and meter.total_usd == 0.0


async def test_an_unparsable_judgement_fails_the_case_but_keeps_the_cost() -> None:
    meter = CostMeter(PRICES)
    make, judge = agent_and_judge(meter, judge_reply="no json here")
    ts = taskset(task("a", "q", {"type": "llm_judge", "rubric": "r"}))
    (case,) = (await EvalRunner(make, judge_llm=judge, meter=meter).run(ts)).cases
    assert case.status == "failed" and case.scores[-1].error
    assert case.judge_cost_usd == pytest.approx(1.0)


# ----------------------------------------------------------------- budgets
async def test_runner_budget_stops_starting_new_jobs() -> None:
    meter = CostMeter(PRICES)
    make, _ = agent_and_judge(meter)  # every job costs exactly $1.00
    ts = taskset(*(task(f"t{i}") for i in range(6)))
    report = await EvalRunner(make, concurrency=1, budget_usd=2.5).run(ts)
    assert [c.status for c in report.cases] == ["passed"] * 3 + ["not_run"] * 3
    s = report.summary()
    assert report.aborted_by_budget is True and s["aborted_by_budget"] is True
    assert (s["completed_jobs"], s["not_run_jobs"], s["planned_jobs"]) == (3, 3, 6)
    assert s["pass_rate"] == 1.0  # rates cover completed jobs only; the gap is explicit above
    assert s["cost"] == pytest.approx(3.0)


async def test_budget_hit_inside_the_agent_aborts_the_case_and_counts_the_final_call() -> None:
    meter = CostMeter(PRICES, budget_usd=1.5)
    make, _ = agent_and_judge(meter)  # each agent call costs $1.00
    ts = taskset(task("a"), task("b"), task("c"))
    report = await EvalRunner(make, concurrency=1, meter=meter).run(ts)
    assert [c.status for c in report.cases] == ["passed", "aborted", "not_run"]
    aborted = report.cases[1]
    assert aborted.stop_reason == "budget_exceeded"
    assert aborted.agent_cost_usd == pytest.approx(1.0)  # the call that crossed the line is counted
    assert report.summary()["cost"] == pytest.approx(meter.total_usd) == pytest.approx(2.0)
    assert report.aborted_by_budget and report.summary()["aborted_jobs"] == 1


async def test_budget_hit_inside_the_judge_counts_the_judges_final_call() -> None:
    meter = CostMeter(PRICES, budget_usd=1.5)
    make, judge = agent_and_judge(meter)  # agent $1.00, then the judge's $1.00 crosses the budget
    ts = taskset(task("a", "q", {"type": "llm_judge", "rubric": "r"}), task("b"))
    report = await EvalRunner(make, judge_llm=judge, concurrency=1, meter=meter).run(ts)
    assert [c.status for c in report.cases] == ["aborted", "not_run"]
    case = report.cases[0]
    assert case.agent_cost_usd == pytest.approx(1.0) and case.judge_cost_usd == pytest.approx(1.0)
    assert case.scores[-1].error and "budget" in case.scores[-1].reason
    assert report.summary()["cost"] == pytest.approx(meter.total_usd) == pytest.approx(2.0)
    assert report.summary()["meter_discrepancy"] == pytest.approx(0.0)


def test_unused_imports_guard() -> None:
    # keeps EvalTask / ScorerSpec importable from the public module path used in docs
    assert EvalTask and ScorerSpec


# --- two currencies through the runner
async def test_an_agent_in_one_currency_and_a_judge_in_another_need_a_declared_rate() -> None:
    def entry(cur: str, rate: float) -> dict[str, object]:
        return {"currency": cur, "per": "1M tokens", "input": rate, "output": rate}

    models = {"agent-m": entry("CNY", 1.0), "judge-m": entry("USD", 2.0)}
    fx = [
        {"from": "USD", "to": "CNY", "rate": 7.0, "source": "https://example.test/fx", "as_of": "d"}
    ]
    meter = CostMeter(
        PriceTable.from_dict({"models": models, "fx": fx}), budget=100, currency="CNY"
    )
    make, judge = agent_and_judge(meter)
    ts = taskset(task("a", "q", {"type": "llm_judge", "rubric": "good answer"}))
    report = await EvalRunner(make, judge_llm=judge, meter=meter).run(ts)
    s = report.summary()
    assert s["currency"] == "CNY" and [r["rate"] for r in s["fx_applied"]] == [7.0]
    assert s["cost_by_currency"] == {"CNY": pytest.approx(1.0), "USD": pytest.approx(1.0)}
    assert s["agent_cost"] == pytest.approx(1.0) and s["judge_cost"] == pytest.approx(
        7.0
    )  # converted
    assert s["cost"] == pytest.approx(8.0) and s["meter_discrepancy"] == pytest.approx(0.0)
