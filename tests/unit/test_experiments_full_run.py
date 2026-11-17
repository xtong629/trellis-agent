from experiments.full_run import chunk_plan, merge_reports, stop_reason

from trellis_agent.eval import EvalReport
from trellis_agent.eval.report import CaseResult


def case(task: str, status: str = "passed", cost: float = 0.0, **kw: object) -> CaseResult:
    return CaseResult(task, 0, status, agent_cost_usd=cost, **kw)  # type: ignore[arg-type]


def report(cases: list[CaseResult], total: float = 1.0, **kw: object) -> EvalReport:
    return EvalReport(
        "exp", "2", "h", {"model": "m", "meta": {"k": "v"}}, cases, total, currency="CNY",
        cost_by_currency={"CNY": total}, **kw,
    )  # type: ignore[arg-type]  # fmt: skip


def test_the_plan_is_repeat_major_over_the_four_systems_and_skips_finished_chunks() -> None:
    plan = chunk_plan(repeats=2, done=set())
    assert [(r, s) for r, s in plan][:5] == [(0, "a_single"), (0, "a_planned"), (0, "b_window"),
                                             (0, "b_memory"), (1, "a_single")]  # fmt: skip
    assert len(plan) == 8
    assert chunk_plan(repeats=2, done={(0, "a_single"), (1, "b_memory")}) == [
        p for p in plan if p not in {(0, "a_single"), (1, "b_memory")}
    ]


def test_merging_chunks_renumbers_repeats_and_sums_costs_exactly() -> None:
    chunks = [
        report([case("t1", cost=0.05), case("t2", "failed", cost=0.05)], 0.1),
        report([case("t1", cost=0.2)], 0.2),
    ]
    merged = merge_reports(
        chunks, concurrency={"jobs": 2}, spent=[0.1, 0.2], extra_meta={"repeats": "2"}
    )
    assert [(c.task_id, c.repeat) for c in merged.cases] == [("t1", 0), ("t2", 0), ("t1", 1)]
    assert merged.meter_total_usd == 0.1 + 0.2 and merged.cost_by_currency == {"CNY": 0.1 + 0.2}
    assert merged.config["concurrency"] == {"jobs": 2} and merged.config["meta"]["repeats"] == "2"
    assert merged.config["meta"]["k"] == "v" and merged.currency == "CNY"
    assert (
        merged.summary()["meter_discrepancy"] == 0
        or abs(merged.summary()["meter_discrepancy"]) < 1e-9
    )


def test_stop_reasons_cover_429_slow_calls_budget_and_aborts() -> None:
    ok = [{"status": 200, "latency_s": 3.0}]
    clean = report([case("t")])
    assert stop_reason(ok, 1.0, 5.0, [clean]) is None
    assert "429" in (stop_reason([*ok, {"status": 429, "latency_s": 0.1}], 1.0, 5.0, [clean]) or "")
    assert "slow" in (stop_reason([{"status": 200, "latency_s": 75.0}], 1.0, 5.0, [clean]) or "")
    assert "budget" in (stop_reason(ok, 4.0, 5.0, [clean]) or "")  # 80% of the cap
    aborted = report([case("t", "aborted")], aborted_by_budget=True)
    assert stop_reason(ok, 1.0, 5.0, [aborted]) is not None
    crashed = report([case("t", "failed", error="target_error: boom")])
    assert "error" in (stop_reason(ok, 1.0, 5.0, [crashed]) or "")


def test_the_recorded_scheduler_concurrency_is_the_real_default() -> None:
    import inspect

    from experiments.full_run import SCHEDULER_MAX

    from trellis_agent.planner import Scheduler

    assert (
        inspect.signature(Scheduler.__init__).parameters["max_concurrency"].default == SCHEDULER_MAX
    )
