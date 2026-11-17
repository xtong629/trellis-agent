"""Event-driven task scheduler.

The core idea is in ``_Run.execute``: wait for *any one* running task to finish
(``asyncio.wait(..., FIRST_COMPLETED)``), record its outcome, then immediately start whatever has
become ready. A slot freed by a short task is refilled at once; nothing waits for a "batch" of
slower tasks to finish. Failures follow a policy (skip dependents, fail fast, or ask a planner to
replan), transient failures are retried with backoff, and a budget can stop new work.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

from ..cost import BudgetExceeded, CostMeter
from ..trace import Tracer
from ..types import RunResult, TraceEvent, Usage
from .graph import Plan, PlanError, TaskGraph, TaskNode
from .retry import RetryPolicy

OnFailure = Literal["skip_dependents", "fail_fast", "replan"]


class NodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Worker(Protocol):
    async def arun(self, query: str, *, session_id: str = "default") -> RunResult: ...


class Replanner(Protocol):
    async def replan(
        self,
        goal: str,
        failed: TaskNode,
        error: str,
        done: Mapping[str, str],
        remaining: Sequence[TaskNode],
    ) -> Plan: ...


@dataclass
class NodeResult:
    node: TaskNode
    status: NodeStatus = NodeStatus.PENDING
    output: str = ""
    attempts: int = 0
    error: str | None = None
    skipped_because: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = 0.0  # None once any attempt had an unknown cost
    started_at: float | None = None
    finished_at: float | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    budget_hit: bool = False


@dataclass
class ScheduleResult:
    nodes: list[NodeResult]
    status: str  # succeeded | failed | budget_exceeded
    replans: int
    replaced: list[NodeResult]  # unfinished nodes that a replan swapped out (kept for the record)
    usage: Usage  # everything: nodes (incl. replaced ones) plus replanning
    cost_usd: float | None
    trace: list[TraceEvent]
    started_at: float
    finished_at: float
    planning_usage: Usage = field(default_factory=Usage)  # the replanning part of ``usage``
    planning_cost_usd: float | None = 0.0

    @property
    def makespan(self) -> float:
        return self.finished_at - self.started_at

    def by_id(self) -> dict[str, NodeResult]:
        return {r.node.id: r for r in self.nodes}


def build_task_prompt(goal: str, node: TaskNode, upstream: Mapping[str, str]) -> str:
    lines = [f"Overall goal: {goal}", f"Your task ({node.id}): {node.goal}"]
    if upstream:
        lines.append("Results from prerequisite tasks:")
        lines += [f"[{dep}] {output}" for dep, output in upstream.items()]
    return "\n".join(lines)


class Scheduler:
    def __init__(
        self,
        workers: Mapping[str, Worker],
        *,
        default_kind: str | None = None,
        max_concurrency: int = 4,
        retry: RetryPolicy | None = None,
        on_failure: OnFailure = "skip_dependents",
        planner: Replanner | None = None,
        max_replans: int = 1,
        meter: CostMeter | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not workers:
            raise ValueError("at least one worker is required")
        if default_kind is not None and default_kind not in workers:
            raise ValueError(f"default_kind {default_kind!r} is not a registered worker")
        if max_concurrency < 1 or max_replans < 0:
            raise ValueError("max_concurrency must be >= 1 and max_replans >= 0")
        if on_failure not in ("skip_dependents", "fail_fast", "replan"):
            raise ValueError(f"unknown on_failure policy {on_failure!r}")
        if on_failure == "replan" and planner is None:
            raise ValueError("on_failure='replan' needs a planner")
        self.workers = dict(workers)
        self.default_kind = default_kind
        self.max_concurrency = max_concurrency
        self.retry = retry or RetryPolicy()
        self.on_failure = on_failure
        self.planner = planner
        self.max_replans = max_replans
        self.meter = meter
        self.clock = clock
        self.sleep = sleep
        self.rng = rng or random.Random(0)

    def known_kinds(self) -> set[str] | None:
        return None if self.default_kind else set(self.workers)

    def worker_for(self, node: TaskNode) -> Worker:
        return self.workers.get(node.kind) or self.workers[self.default_kind or ""]

    async def run(self, graph: TaskGraph, goal: str) -> ScheduleResult:
        graph.validate(self.known_kinds())  # raises GraphError before anything is started
        return await _Run(self, graph, goal).execute()


class _Run:
    """State of one ``Scheduler.run`` call."""

    def __init__(self, sched: Scheduler, graph: TaskGraph, goal: str) -> None:
        self.s = sched
        self.goal = goal
        self.max_nodes = graph.max_nodes
        self.tracer = Tracer(clock=sched.clock)
        self.nodes: dict[str, TaskNode] = {n.id: n for n in graph.nodes}
        self.order: list[str] = [n.id for n in graph.nodes]
        self.results: dict[str, NodeResult] = {n.id: NodeResult(n) for n in graph.nodes}
        self.running: dict[asyncio.Task[None], str] = {}
        self.halted: str | None = None  # "budget" | "fail_fast": no new work is started
        self.trigger: str | None = None  # the task whose failure caused fail_fast
        self.pending_replan: str | None = None
        self.replans = 0
        self.replan_disabled = False
        self.replaced: list[NodeResult] = []
        self.plan_usage = Usage()
        self.plan_cost: float | None = 0.0

    # ------------------------------------------------------------ the event loop
    async def execute(self) -> ScheduleResult:
        started = self.s.clock()
        try:
            while True:
                self._launch_ready()
                if not self.running:
                    if self.pending_replan is not None:
                        await self._replan()
                        continue
                    break
                finished, _ = await asyncio.wait(
                    set(self.running), return_when=asyncio.FIRST_COMPLETED
                )
                for task in finished:
                    self._on_finished(task)
                if self.halted == "fail_fast":
                    await self._cancel_running()
        finally:
            await self._cancel_running(note="the run was cancelled")
        self._skip_unstarted()
        return self._result(started)

    def _launch_ready(self) -> None:
        """Start every ready task that fits in a free slot (called after each completion)."""
        if self.halted or self.pending_replan is not None:
            return
        for task_id in self.order:
            if len(self.running) >= self.s.max_concurrency:
                return
            entry = self.results[task_id]
            if entry.status is not NodeStatus.PENDING:
                continue
            if not all(
                self.results[d].status is NodeStatus.DONE for d in self.nodes[task_id].depends_on
            ):
                continue
            if self.s.meter is not None and self.s.meter.exhausted:
                self.halted = "budget"
                return
            entry.status = NodeStatus.RUNNING
            entry.started_at = self.s.clock()
            self.running[asyncio.create_task(self._run_node(entry))] = task_id

    def _on_finished(self, task: asyncio.Task[None]) -> None:
        task_id = self.running.pop(task)
        if task.cancelled():
            return
        task.result()  # a bug inside the scheduler must surface, not vanish
        entry = self.results[task_id]
        self.tracer.emit("task_end", id=task_id, status=entry.status.value, attempts=entry.attempts)
        if entry.status is NodeStatus.FAILED:
            self._on_failure(entry)

    # ------------------------------------------------------------ failure handling
    def _on_failure(self, entry: NodeResult) -> None:
        if entry.budget_hit:
            self.halted = self.halted or "budget"
        elif self.pending_replan is not None:
            return  # a replan is already waiting for running tasks; it will cover this one too
        elif self.s.on_failure == "fail_fast":
            self.halted, self.trigger = "fail_fast", entry.node.id
        elif (
            self.s.on_failure == "replan"
            and not self.replan_disabled
            and self.replans < self.s.max_replans
        ):
            self.pending_replan = entry.node.id
        else:
            self._skip_dependents(entry.node.id)

    def _skip_dependents(self, failed_id: str) -> None:
        blocked = TaskGraph(list(self.nodes.values())).dependents(failed_id)
        for task_id in self.order:
            if task_id in blocked and self.results[task_id].status is NodeStatus.PENDING:
                self._skip(task_id, f"depends on task {failed_id!r}, which failed")

    def _skip(self, task_id: str, reason: str) -> None:
        entry = self.results[task_id]
        entry.status, entry.skipped_because = NodeStatus.SKIPPED, reason
        self.tracer.emit("task_skipped", id=task_id, reason=reason)

    async def _cancel_running(self, note: str | None = None) -> None:
        if not self.running:
            return
        tasks = list(self.running)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        reason = note or f"cancelled: fail_fast after task {self.trigger!r} failed"
        for task in tasks:
            self._skip(self.running.pop(task), reason)

    def _skip_unstarted(self) -> None:
        reason = {
            "budget": "budget exhausted",
            "fail_fast": f"fail_fast after task {self.trigger!r} failed",
        }.get(self.halted or "", "an upstream task did not complete")
        for task_id in self.order:
            if self.results[task_id].status is NodeStatus.PENDING:
                self._skip(task_id, reason)

    # ------------------------------------------------------------ replanning
    async def _replan(self) -> None:
        trigger, self.pending_replan = self.pending_replan or "", None
        planner = self.s.planner
        assert planner is not None
        failed = self.results[trigger]
        done = {t: r.output for t, r in self.results.items() if r.status is NodeStatus.DONE}
        remaining = [
            self.nodes[t] for t in self.order if self.results[t].status is not NodeStatus.DONE
        ]
        try:
            plan = await planner.replan(self.goal, failed.node, failed.error or "", done, remaining)
            kept = [self.nodes[t] for t in self.order if self.results[t].status is NodeStatus.DONE]
            TaskGraph([*kept, *plan.graph.nodes], max_nodes=self.max_nodes).validate(
                self.s.known_kinds()
            )
        except BudgetExceeded as exc:
            self._account_plan(
                exc.response.usage if exc.response else None,
                exc.response.cost_usd if exc.response else None,
            )
            self.halted = "budget"
            return
        except Exception as exc:  # includes GraphError: an unusable replan is not fatal
            if isinstance(exc, PlanError):
                self._account_plan(exc.usage, exc.cost_usd)  # failed planning was still paid for
                if exc.budget_exceeded:
                    self.halted = "budget"
                    return
            self.tracer.emit(
                "error", category="replan_failed", message=f"{type(exc).__name__}: {exc}"
            )
            for entry in [r for r in self.results.values() if r.status is NodeStatus.FAILED]:
                self._skip_dependents(entry.node.id)
            self.replan_disabled = True  # do not keep asking a planner that cannot help
            return
        self._account_plan(plan.usage, plan.cost_usd)
        for task_id in [t for t in self.order if self.results[t].status is not NodeStatus.DONE]:
            if self.results[task_id].status is NodeStatus.PENDING:
                self._skip(task_id, f"replaced by a new plan after task {trigger!r} failed")
            self.replaced.append(self.results.pop(task_id))
            del self.nodes[task_id]
            self.order.remove(task_id)
        for node in plan.graph.nodes:
            self.nodes[node.id] = node
            self.order.append(node.id)
            self.results[node.id] = NodeResult(node)
        self.replans += 1
        self.tracer.emit(
            "replan", failed=trigger, new_tasks=[n.id for n in plan.graph.nodes], count=self.replans
        )

    def _account_plan(self, usage: Usage | None, cost: float | None) -> None:
        if usage is not None:
            self.plan_usage = self.plan_usage + usage
        self.plan_cost = None if self.plan_cost is None or cost is None else self.plan_cost + cost

    # ------------------------------------------------------------ one task, with retries
    async def _run_node(self, entry: NodeResult) -> None:
        node, policy = entry.node, self.s.retry
        upstream = {d: self.results[d].output for d in node.depends_on}
        prompt = build_task_prompt(self.goal, node, upstream)
        worker = self.s.worker_for(node)
        for attempt in range(1, policy.max_attempts + 1):
            entry.attempts = attempt
            self.tracer.emit("task_start", id=node.id, kind=node.kind, attempt=attempt)
            retryable = False
            try:
                run = await worker.arun(prompt, session_id=f"{node.id}#{attempt}")
            except asyncio.CancelledError:
                raise
            except BudgetExceeded as exc:
                entry.budget_hit = True
                if exc.response is not None:
                    self._add_usage(entry, exc.response.usage, exc.response.cost_usd)
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                error, retryable = f"{type(exc).__name__}: {exc}", policy.is_retryable_error(exc)
            else:
                self._add_usage(entry, run.usage, run.cost_usd)
                entry.trace.extend(run.trace)
                if run.error is None:
                    entry.status, entry.output, entry.error = NodeStatus.DONE, run.content, None
                    break
                error, retryable = run.error, policy.is_retryable_result(run)
                entry.budget_hit = entry.budget_hit or run.stop_reason == "budget_exceeded"
            entry.error = error
            self.tracer.emit("error", category="task_attempt_failed", id=node.id, message=error)
            if self.s.meter is not None and self.s.meter.exhausted:
                entry.budget_hit = True
            if entry.budget_hit or not retryable or attempt == policy.max_attempts:
                entry.status = NodeStatus.FAILED
                break
            delay = policy.delay(attempt, self.s.rng)
            self.tracer.emit("retry", id=node.id, attempt=attempt, delay=delay, reason=error)
            await self.s.sleep(delay)
        entry.finished_at = self.s.clock()

    @staticmethod
    def _add_usage(entry: NodeResult, usage: Usage, cost: float | None) -> None:
        entry.usage = entry.usage + usage
        entry.cost_usd = None if entry.cost_usd is None or cost is None else entry.cost_usd + cost

    # ------------------------------------------------------------ result
    def _result(self, started: float) -> ScheduleResult:
        final = [self.results[t] for t in self.order]
        everyone = [*self.replaced, *final]
        usage = self.plan_usage
        costs: list[float | None] = [self.plan_cost]
        for r in everyone:
            usage = usage + r.usage
            costs.append(r.cost_usd)
        total: float | None = (
            None if any(c is None for c in costs) else sum(c for c in costs if c is not None)
        )
        if self.halted == "budget":
            status = "budget_exceeded"
        elif all(r.status is NodeStatus.DONE for r in final):
            status = "succeeded"
        else:
            status = "failed"
        return ScheduleResult(
            nodes=final,
            status=status,
            replans=self.replans,
            replaced=self.replaced,
            usage=usage,
            cost_usd=total,
            trace=self.tracer.events,
            started_at=started,
            finished_at=self.s.clock(),
            planning_usage=self.plan_usage,
            planning_cost_usd=self.plan_cost,
        )
