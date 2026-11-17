"""The scheduler, on virtual time. Timelines are exact because time only moves when all is idle."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence

import pytest

from tests.unit.vclock import FakeWorker, VirtualClock
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.planner import GraphError, RetryPolicy, TaskGraph, TaskNode, TransientError
from trellis_agent.planner.graph import Plan
from trellis_agent.planner.scheduler import NodeResult, NodeStatus, Scheduler
from trellis_agent.trace import summarize
from trellis_agent.types import Message, RunResult, Usage


def n(id_: str, *deps: str, kind: str = "default") -> TaskNode:
    return TaskNode(id_, f"goal of {id_}", kind, tuple(deps))


def failing(reason: str = "target_error", error: str = "boom") -> RunResult:
    return RunResult("", [], Usage(10, 5), None, [], error, reason)


def make(clock: VirtualClock, workers: Mapping[str, FakeWorker], **kw: object) -> Scheduler:
    kw.setdefault("retry", RetryPolicy(jitter=0.0))
    return Scheduler(workers, clock=clock.time, sleep=clock.sleep, **kw)  # type: ignore[arg-type]


def status_of(result_nodes: Sequence[NodeResult]) -> dict[str, str]:
    return {r.node.id: r.status.value for r in result_nodes}


# ----------------------------------------------------------------- event-driven backfill
@pytest.mark.parametrize("limit", [2, 3, 4])
async def test_a_freed_slot_is_refilled_immediately_not_at_the_end_of_a_batch(limit: int) -> None:
    # A takes 3, B takes 1, C needs B and takes 1. A batch scheduler would start C only after the
    # whole first batch (A and B) is done, at t=3. Event-driven: C starts the moment B finishes.
    clock = VirtualClock()
    a, b, c = (FakeWorker(x, clock, d) for x, d in (("a", 3.0), ("b", 1.0), ("c", 1.0)))
    scheduler = make(clock, {"a": a, "b": b, "c": c}, max_concurrency=limit)
    graph = TaskGraph([n("A", kind="a"), n("B", kind="b"), n("C", "B", kind="c")])
    result = await clock.run(scheduler.run(graph, "goal"))
    by = result.by_id()
    assert (by["A"].started_at, by["A"].finished_at) == (0.0, 3.0)
    assert (by["B"].started_at, by["B"].finished_at) == (0.0, 1.0)
    assert (by["C"].started_at, by["C"].finished_at) == (1.0, 2.0)
    assert by["C"].started_at < by["A"].finished_at  # started while A was still running
    assert result.makespan == 3.0  # a batch scheduler needs 4.0
    assert result.status == "succeeded"


async def test_concurrency_limit_is_respected_and_slots_never_idle() -> None:
    clock = VirtualClock()
    w = FakeWorker("w", clock, 1.0)
    graph = TaskGraph([n(f"t{i}") for i in range(6)])
    result = await clock.run(make(clock, {"default": w}, max_concurrency=2).run(graph, "g"))
    assert w.peak == 2 and result.makespan == 3.0  # 6 one-second tasks / 2 slots, no gaps


async def test_dependencies_are_respected_and_outputs_flow_downstream() -> None:
    clock = VirtualClock()
    workers = {k: FakeWorker(k, clock, 1.0) for k in ("a", "b", "c", "d")}
    graph = TaskGraph(
        [n("a", kind="a"), n("b", "a", kind="b"), n("c", "a", kind="c"), n("d", "b", "c", kind="d")]
    )
    result = await clock.run(make(clock, workers).run(graph, "THE OVERALL GOAL"))
    by = result.by_id()
    assert by["b"].started_at == by["c"].started_at == 1.0  # parallel branches
    assert by["d"].started_at == 2.0 and result.makespan == 3.0
    prompt = workers["d"].calls[0].prompt
    assert "THE OVERALL GOAL" in prompt and "goal of d" in prompt
    assert "b output" in prompt and "c output" in prompt and "a output" not in prompt
    assert "a output" in workers["b"].calls[0].prompt


async def test_results_follow_graph_order_whatever_finishes_first() -> None:
    clock = VirtualClock()
    slow, fast = FakeWorker("slow", clock, 5.0), FakeWorker("fast", clock, 1.0)
    graph = TaskGraph([n("first", kind="slow"), n("second", kind="fast")])
    result = await clock.run(make(clock, {"slow": slow, "fast": fast}).run(graph, "g"))
    assert [r.node.id for r in result.nodes] == ["first", "second"]


# ----------------------------------------------------------------- retries and backoff
async def test_transient_errors_are_retried_with_exponential_backoff() -> None:
    clock = VirtualClock()
    w = FakeWorker("w", clock, 0.0, script=[TransientError("t1"), TransientError("t2"), None])
    result = await clock.run(make(clock, {"default": w}).run(TaskGraph([n("a")]), "g"))
    (r,) = result.nodes
    assert r.status is NodeStatus.DONE and r.attempts == 3
    assert clock.sleeps == [0.5, 1.0]  # 0.5 then 1.0, as the policy says
    assert r.finished_at == 1.5
    assert [c.session_id for c in w.calls] == ["a#1", "a#2", "a#3"]  # a fresh session per attempt
    s = summarize(result.trace)
    assert s.retry_count == 2


async def test_jitter_keeps_delays_inside_the_band_and_is_seedable() -> None:
    async def delays(seed: int) -> list[float]:
        clock = VirtualClock()
        w = FakeWorker("w", clock, script=[TransientError(), TransientError(), None])
        sched = Scheduler(
            {"default": w},
            clock=clock.time,
            sleep=clock.sleep,
            retry=RetryPolicy(jitter=0.2),
            rng=random.Random(seed),
        )
        await clock.run(sched.run(TaskGraph([n("a")]), "g"))
        return clock.sleeps

    first = await delays(1)
    assert 0.4 <= first[0] <= 0.6 and 0.8 <= first[1] <= 1.2
    assert await delays(1) == first and await delays(2) != first


async def test_retry_gives_up_after_max_attempts_and_reports_the_last_error() -> None:
    clock = VirtualClock()
    w = FakeWorker("w", clock, script=[TransientError(f"try {i}") for i in range(5)])
    result = await clock.run(make(clock, {"default": w}).run(TaskGraph([n("a")]), "g"))
    (r,) = result.nodes
    assert r.status is NodeStatus.FAILED and r.attempts == 3
    assert "try 2" in (r.error or "") and "TransientError" in (r.error or "")
    assert len(clock.sleeps) == 2  # no pointless sleep after the final attempt


async def test_non_retryable_errors_fail_immediately() -> None:
    clock = VirtualClock()
    w = FakeWorker("w", clock, script=[ValueError("bad input")])
    result = await clock.run(make(clock, {"default": w}).run(TaskGraph([n("a")]), "g"))
    (r,) = result.nodes
    assert r.status is NodeStatus.FAILED and r.attempts == 1 and clock.sleeps == []
    assert "ValueError" in (r.error or "")


async def test_provider_failures_in_a_run_result_are_retried_but_step_limits_are_not() -> None:
    clock = VirtualClock()
    flaky = FakeWorker("w", clock, script=[failing("llm_error"), None])
    result = await clock.run(make(clock, {"default": flaky}).run(TaskGraph([n("a")]), "g"))
    assert result.nodes[0].status is NodeStatus.DONE and result.nodes[0].attempts == 2

    clock2 = VirtualClock()
    stuck = FakeWorker("w", clock2, script=[failing("max_steps", "max_steps reached")])
    result2 = await clock2.run(make(clock2, {"default": stuck}).run(TaskGraph([n("a")]), "g"))
    assert result2.nodes[0].status is NodeStatus.FAILED and result2.nodes[0].attempts == 1


async def test_failed_attempts_still_count_toward_usage() -> None:
    clock = VirtualClock()
    first = RunResult("", [], Usage(100, 10), None, [], "down", "llm_error")
    w = FakeWorker("w", clock, script=[first, None], usage=Usage(100, 10))
    (r,) = (await clock.run(make(clock, {"default": w}).run(TaskGraph([n("a")]), "g"))).nodes
    assert r.usage == Usage(200, 20)  # both attempts were paid for


# ----------------------------------------------------------------- failure policies
async def test_failure_skips_only_the_dependents_and_independent_work_continues() -> None:
    clock = VirtualClock()
    bad = FakeWorker("bad", clock, script=[ValueError("nope")])
    ok = FakeWorker("ok", clock, 1.0)
    graph = TaskGraph(
        [n("a", kind="bad"), n("b", "a", kind="ok"), n("c", "b", kind="ok"), n("x", kind="ok")]
    )
    result = await clock.run(make(clock, {"bad": bad, "ok": ok}).run(graph, "g"))
    assert status_of(result.nodes) == {"a": "failed", "b": "skipped", "c": "skipped", "x": "done"}
    by = result.by_id()
    assert "'a'" in (by["b"].skipped_because or "") and "'a'" in (by["c"].skipped_because or "")
    assert result.status == "failed"
    assert [c.prompt for c in ok.calls if "goal of b" in c.prompt] == []  # never ran


async def test_fail_fast_cancels_running_work_and_skips_the_rest() -> None:
    clock = VirtualClock()
    bad = FakeWorker("bad", clock, 1.0, script=[ValueError("nope")])
    slow = FakeWorker("slow", clock, 10.0)
    graph = TaskGraph([n("a", kind="bad"), n("x", kind="slow"), n("y", "x", kind="slow")])
    result = await clock.run(
        make(clock, {"bad": bad, "slow": slow}, on_failure="fail_fast").run(graph, "g")
    )
    assert status_of(result.nodes) == {"a": "failed", "x": "skipped", "y": "skipped"}
    assert slow.calls[0].cancelled and result.makespan == 1.0  # did not wait for x
    assert "fail_fast" in (result.by_id()["x"].skipped_because or "")


# ----------------------------------------------------------------- replanning
class FakePlanner:
    def __init__(self, replacement: TaskGraph | Exception) -> None:
        self.replacement = replacement
        self.calls: list[dict[str, object]] = []

    async def replan(
        self,
        goal: str,
        failed: TaskNode,
        error: str,
        done: Mapping[str, str],
        remaining: Sequence[TaskNode],
    ) -> Plan:
        self.calls.append(
            {
                "goal": goal,
                "failed": failed.id,
                "error": error,
                "done": dict(done),
                "remaining": [r.id for r in remaining],
            }
        )
        if isinstance(self.replacement, Exception):
            raise self.replacement
        return Plan(self.replacement, Usage(7, 3), 0.5, 1)


async def test_replan_replaces_the_unfinished_part_of_the_plan() -> None:
    clock = VirtualClock()
    ok = FakeWorker("ok", clock, 1.0, cost=0.0)
    bad = FakeWorker("bad", clock, 1.0, cost=0.0, script=[ValueError("tool broke")])
    planner = FakePlanner(TaskGraph([n("a2", "root", kind="ok"), n("final", "a2", kind="ok")]))
    graph = TaskGraph([n("root", kind="ok"), n("a", "root", kind="bad"), n("b", "a", kind="ok")])
    sched = make(clock, {"ok": ok, "bad": bad}, on_failure="replan", planner=planner)
    result = await clock.run(sched.run(graph, "the goal"))
    assert status_of(result.nodes) == {"root": "done", "a2": "done", "final": "done"}
    assert result.status == "succeeded" and result.replans == 1
    (call,) = planner.calls
    assert call["failed"] == "a" and "tool broke" in str(call["error"])
    assert call["done"] == {"root": "ok output"} and call["remaining"] == ["a", "b"]
    assert [r.node.id for r in result.replaced] == ["a", "b"]  # history is kept, not erased
    assert (
        result.replaced[0].status is NodeStatus.FAILED
        and result.replaced[1].status is NodeStatus.SKIPPED
    )
    assert (
        result.usage.prompt_tokens >= 7 and result.cost_usd is not None and result.cost_usd >= 0.5
    )
    assert summarize(result.trace).event_count > 0
    assert any(e.type == "replan" for e in result.trace)


async def test_replan_waits_for_running_tasks_to_finish_first() -> None:
    clock = VirtualClock()
    bad = FakeWorker("bad", clock, 1.0, script=[ValueError("x")])
    slow = FakeWorker("slow", clock, 5.0)
    planner = FakePlanner(TaskGraph([n("z", "slow_done", kind="slow")]))
    graph = TaskGraph([n("a", kind="bad"), n("slow_done", kind="slow")])
    result = await clock.run(
        make(clock, {"bad": bad, "slow": slow}, on_failure="replan", planner=planner).run(
            graph, "g"
        )
    )
    assert planner.calls[0]["done"] == {"slow_done": "slow output"}  # it was allowed to finish
    assert result.by_id()["z"].started_at == 5.0


async def test_replans_are_limited() -> None:
    clock = VirtualClock()
    bad = FakeWorker("bad", clock, script=[ValueError("1"), ValueError("2"), ValueError("3")])
    planner = FakePlanner(TaskGraph([n("again", kind="bad")]))
    sched = make(clock, {"bad": bad}, on_failure="replan", planner=planner, max_replans=1)
    result = await clock.run(sched.run(TaskGraph([n("a", kind="bad")]), "g"))
    assert result.replans == 1 and len(planner.calls) == 1
    assert result.status == "failed" and result.by_id()["again"].status is NodeStatus.FAILED


@pytest.mark.parametrize(
    "bad_replacement",
    [
        TaskGraph([n("x", "ghost")]),
        TaskGraph([n("p", "q"), n("q", "p")]),
        RuntimeError("planner down"),
    ],
)
async def test_an_unusable_replan_falls_back_to_skipping_dependents(
    bad_replacement: TaskGraph | Exception,
) -> None:
    clock = VirtualClock()
    bad = FakeWorker("bad", clock, script=[ValueError("x")])
    ok = FakeWorker("ok", clock, 1.0)
    planner = FakePlanner(bad_replacement)
    graph = TaskGraph([n("a", kind="bad"), n("b", "a", kind="ok"), n("free", kind="ok")])
    sched = make(clock, {"bad": bad, "ok": ok}, on_failure="replan", planner=planner)
    result = await clock.run(sched.run(graph, "g"))
    assert status_of(result.nodes) == {"a": "failed", "b": "skipped", "free": "done"}
    assert result.replans == 0
    assert any(
        e.type == "error" and e.data.get("category") == "replan_failed" for e in result.trace
    )


def test_replan_policy_needs_a_planner() -> None:
    with pytest.raises(ValueError, match="planner"):
        Scheduler({"default": FakeWorker()}, on_failure="replan")


# ----------------------------------------------------------------- budget
PRICES = PriceTable({"m": ModelPrice(1.0, 1.0)})


async def test_exhausted_budget_stops_new_work_but_lets_running_work_finish() -> None:
    clock = VirtualClock()
    meter = CostMeter(PRICES, budget_usd=1.0)
    meter.record("m", Usage(prompt_tokens=1_000_000))  # budget already spent
    w = FakeWorker("w", clock, 1.0)
    result = await clock.run(make(clock, {"default": w}, meter=meter).run(TaskGraph([n("a")]), "g"))
    assert w.calls == [] and result.nodes[0].status is NodeStatus.SKIPPED
    assert result.status == "budget_exceeded" and "budget" in (
        result.nodes[0].skipped_because or ""
    )


async def test_a_worker_hitting_the_budget_stops_the_schedule() -> None:
    clock = VirtualClock()
    over = failing("budget_exceeded", "budget exceeded: spent $2 of $1")
    w = FakeWorker("w", clock, 1.0, script=[over])
    graph = TaskGraph([n("a"), n("b", "a")])
    result = await clock.run(make(clock, {"default": w}).run(graph, "g"))
    assert status_of(result.nodes) == {"a": "failed", "b": "skipped"}
    assert result.status == "budget_exceeded" and w.calls[0].prompt  # a ran once, no retry


async def test_no_new_attempt_is_started_once_the_budget_is_exhausted() -> None:
    clock = VirtualClock()
    meter = CostMeter(PRICES, budget_usd=1.0)

    class SpendThenFail(FakeWorker):
        async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
            meter.record("m", Usage(prompt_tokens=1_000_000))
            raise TransientError("flaky")

    w = SpendThenFail("w", clock)
    result = await clock.run(make(clock, {"default": w}, meter=meter).run(TaskGraph([n("a")]), "g"))
    assert result.nodes[0].attempts == 1 and result.status == "budget_exceeded"


# ----------------------------------------------------------------- dispatch, accounting, trace
async def test_unknown_kinds_are_rejected_before_anything_runs() -> None:
    clock = VirtualClock()
    w = FakeWorker("w", clock)
    sched = make(clock, {"research": w})
    with pytest.raises(GraphError, match="kind 'wizard'"):
        await sched.run(TaskGraph([n("a", kind="wizard")]), "g")
    assert w.calls == []


async def test_a_default_kind_catches_unregistered_kinds() -> None:
    clock = VirtualClock()
    w = FakeWorker("general", clock)
    result = await clock.run(
        make(clock, {"general": w}, default_kind="general").run(
            TaskGraph([n("a", kind="odd")]), "g"
        )
    )
    assert result.status == "succeeded" and len(w.calls) == 1


async def test_usage_and_cost_are_summed_and_unknown_costs_stay_unknown() -> None:
    clock = VirtualClock()
    priced = FakeWorker("p", clock, usage=Usage(100, 10), cost=0.25)
    graph = TaskGraph([n("a", kind="p"), n("b", kind="p")])
    result = await clock.run(make(clock, {"p": priced}).run(graph, "g"))
    assert result.usage == Usage(200, 20) and result.cost_usd == pytest.approx(0.5)

    clock2 = VirtualClock()
    unpriced = FakeWorker("u", clock2, usage=Usage(100, 10), cost=None)
    result2 = await clock2.run(
        make(clock2, {"u": unpriced}).run(TaskGraph([n("a", kind="u")]), "g")
    )
    assert result2.cost_usd is None and result2.nodes[0].cost_usd is None


async def test_trace_records_the_schedule() -> None:
    clock = VirtualClock()
    w = FakeWorker("w", clock, script=[ValueError("x")])
    graph = TaskGraph([n("a"), n("b", "a")])
    result = await clock.run(make(clock, {"default": w}).run(graph, "g"))
    kinds = [e.type for e in result.trace]
    assert kinds.count("task_start") == 1 and kinds.count("task_end") == 1
    assert "task_skipped" in kinds
    ends = [e for e in result.trace if e.type == "task_end"]
    assert ends[0].data["status"] == "failed" and ends[0].data["id"] == "a"


async def test_worker_traces_are_kept_on_the_node() -> None:
    clock = VirtualClock()
    from trellis_agent.types import TraceEvent

    event = TraceEvent("tool_call", 0.0, {"name": "add"}, "root", None)
    w = FakeWorker(
        "w", clock, script=[RunResult("done", [], Usage(1, 1), None, [event], None, "end_turn")]
    )
    result = await clock.run(make(clock, {"default": w}).run(TaskGraph([n("a")]), "g"))
    assert result.nodes[0].trace == [event] and result.nodes[0].output == "done"


async def test_cancelling_the_run_cancels_running_workers() -> None:
    w = FakeWorker("w", None, 30.0)
    sched = Scheduler({"default": w}, retry=RetryPolicy(jitter=0.0))
    task = asyncio.ensure_future(sched.run(TaskGraph([n("a"), n("b")]), "g"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(c.cancelled for c in w.calls) and len(w.calls) == 2


# ----------------------------------------------------------------- planner errors keep their cost
async def test_a_failed_replan_still_counts_what_it_cost() -> None:
    from trellis_agent.planner.graph import PlanError

    clock = VirtualClock()
    bad = FakeWorker("bad", clock, cost=0.0, script=[ValueError("x")])
    err = PlanError(["could not plan"], usage=Usage(50, 5), cost_usd=0.25, attempts=3)
    planner = FakePlanner(err)
    sched = make(clock, {"bad": bad}, on_failure="replan", planner=planner)
    result = await clock.run(sched.run(TaskGraph([n("a", kind="bad")]), "g"))
    assert result.replans == 0 and result.status == "failed"
    assert result.usage.prompt_tokens >= 50 and result.cost_usd == pytest.approx(0.25)


async def test_a_replan_that_runs_out_of_budget_stops_the_schedule() -> None:
    from trellis_agent.planner.graph import PlanError

    clock = VirtualClock()
    bad = FakeWorker("bad", clock, cost=0.0, script=[ValueError("x")])
    ok = FakeWorker("ok", clock, 1.0, cost=0.0)
    err = PlanError(["stopped"], usage=Usage(9, 1), cost_usd=2.0, budget_exceeded=True)
    sched = make(clock, {"bad": bad, "ok": ok}, on_failure="replan", planner=FakePlanner(err))
    graph = TaskGraph([n("a", kind="bad"), n("b", "a", kind="ok")])
    result = await clock.run(sched.run(graph, "g"))
    assert result.status == "budget_exceeded" and ok.calls == []
    assert result.cost_usd == pytest.approx(2.0)
    assert result.by_id()["b"].status is NodeStatus.SKIPPED


# --- configuration and raw budget errors
@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": {}},
        {"default_kind": "ghost"},
        {"max_concurrency": 0},
        {"max_replans": -1},
        {"on_failure": "explode"},
    ],
)
def test_invalid_scheduler_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    args: dict[str, object] = {"workers": {"default": FakeWorker()}}
    args.update(kwargs)
    with pytest.raises(ValueError):
        Scheduler(**args)  # type: ignore[arg-type]


async def test_a_worker_raising_budget_exceeded_is_counted_and_stops_the_schedule() -> None:
    from trellis_agent.cost import BudgetExceeded
    from trellis_agent.llm import LLMResponse

    clock = VirtualClock()
    exc = BudgetExceeded(2.0, 1.0, call_cost=0.7)
    exc.response = LLMResponse(Message.assistant("x"), Usage(40, 4), 0.0, "stop", "m", 0.7)
    w = FakeWorker("w", clock, script=[exc])
    result = await clock.run(make(clock, {"default": w}).run(TaskGraph([n("a"), n("b", "a")]), "g"))
    assert result.status == "budget_exceeded" and result.nodes[0].attempts == 1
    assert result.nodes[0].usage == Usage(40, 4) and result.nodes[0].cost_usd == pytest.approx(0.7)
    assert result.nodes[1].status is NodeStatus.SKIPPED


async def test_a_planner_raising_budget_exceeded_directly_stops_and_counts_the_call() -> None:
    from trellis_agent.cost import BudgetExceeded
    from trellis_agent.llm import LLMResponse

    clock = VirtualClock()
    exc = BudgetExceeded(2.0, 1.0, call_cost=0.4)
    exc.response = LLMResponse(Message.assistant("x"), Usage(9, 1), 0.0, "stop", "m", 0.4)
    bad = FakeWorker("bad", clock, cost=0.0, script=[ValueError("x")])
    sched = make(clock, {"bad": bad}, on_failure="replan", planner=FakePlanner(exc))
    result = await clock.run(sched.run(TaskGraph([n("a", kind="bad")]), "g"))
    assert result.status == "budget_exceeded"
    assert result.cost_usd == pytest.approx(0.4) and result.usage.prompt_tokens >= 9
