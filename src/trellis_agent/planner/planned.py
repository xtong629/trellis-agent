"""PlannedAgent: plan -> schedule role-specific workers -> synthesise one answer.

It satisfies the evaluation harness's ``Runnable`` protocol, so "with the planner" and "without"
(a plain ``Agent``) can be compared on the same task set. Failures are reported in the
``RunResult`` (never raised) with the same accounting rules as everywhere else: every model call is
counted, including failed planning attempts and the call that crossed a budget.
"""

from __future__ import annotations

import time

from ..cost import BudgetExceeded
from ..llm import LLMClient, LLMError
from ..types import Message, RunResult, TraceEvent, Usage
from .graph import PlanError
from .planner import Planner
from .scheduler import NodeResult, NodeStatus, Scheduler, ScheduleResult

_SYSTEM = (
    "You write the final answer to the GOAL using only the TASK RESULTS provided. "
    "Be concise and do not invent facts that are not in the results."
)
_RESULT_CAP = 4000


def _event(kind: str, **data: object) -> TraceEvent:
    return TraceEvent(kind, time.time(), dict(data), "root", None)


def _model_event(
    phase: str, usage: Usage, cost: float | None, latency_ms: float = 0.0, **extra: object
) -> TraceEvent:
    return _event(
        "model_request",
        phase=phase,
        usage=usage.to_dict(),
        cost_usd=cost,
        latency_ms=latency_ms,
        **extra,
    )


def _sum_cost(parts: list[float | None]) -> float | None:
    known = [p for p in parts if p is not None]
    return sum(known) if len(known) == len(parts) else None


class PlannedAgent:
    def __init__(self, planner: Planner, scheduler: Scheduler, synthesizer: LLMClient) -> None:
        self.planner = planner
        self.scheduler = scheduler
        self.synthesizer = synthesizer

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        trace: list[TraceEvent] = []
        usage = Usage()
        costs: list[float | None] = []

        def finish(content: str, stop: str, error: str | None = None) -> RunResult:
            messages = [Message.user(query), Message.assistant(content)]
            return RunResult(content, messages, usage, _sum_cost(costs), trace, error, stop)

        try:
            plan = await self.planner.plan(query)
        except PlanError as exc:
            usage, costs = usage + exc.usage, [exc.cost_usd]
            trace.append(_model_event("plan", exc.usage, exc.cost_usd))
            trace.append(_event("error", category="plan_error", message=str(exc)))
            stop = "budget_exceeded" if exc.budget_exceeded else "plan_error"
            return finish("", stop, f"planning failed: {exc}")
        usage, costs = usage + plan.usage, [plan.cost_usd]
        trace.append(_model_event("plan", plan.usage, plan.cost_usd, attempts=plan.attempts))
        trace.append(_event("plan", tasks=plan.graph.ids(), attempts=plan.attempts))

        schedule = await self.scheduler.run(plan.graph, query)
        usage, costs = usage + schedule.usage, [*costs, schedule.cost_usd]
        trace += self._schedule_events(schedule)
        if schedule.status != "succeeded":
            stop = "budget_exceeded" if schedule.status == "budget_exceeded" else "tasks_failed"
            return finish("", stop, self._describe_failures(schedule))

        results = "\n".join(
            f"[{r.node.id}] {r.node.goal}\n{r.output[:_RESULT_CAP]}" for r in schedule.nodes
        )
        messages = [
            Message.system(_SYSTEM),
            Message.user(f"GOAL:\n{query}\n\nTASK RESULTS:\n{results}"),
        ]
        try:
            reply = await self.synthesizer.chat(messages, temperature=0.0, tag="synthesize")
        except BudgetExceeded as exc:
            paid = exc.response
            if paid is not None:
                usage, costs = usage + paid.usage, [*costs, paid.cost_usd]
                trace.append(
                    _model_event("synthesize", paid.usage, paid.cost_usd, over_budget=True)
                )
            trace.append(_event("error", category="budget_exceeded", message=str(exc)))
            return finish("", "budget_exceeded", str(exc))
        except LLMError as exc:
            trace.append(_event("error", category="llm_error", message=str(exc)))
            return finish("", "llm_error", f"synthesis failed: {exc}")
        usage, costs = usage + reply.usage, [*costs, reply.cost_usd]
        trace.append(_model_event("synthesize", reply.usage, reply.cost_usd, reply.latency_ms))
        return finish(reply.message.content or "", "end_turn")

    @staticmethod
    def _schedule_events(schedule: ScheduleResult) -> list[TraceEvent]:
        """Scheduler events, the replanning cost, and every worker's own trace (tool calls etc.)."""
        events = list(schedule.trace)
        if schedule.planning_usage.total_tokens:
            events.append(
                _model_event("replan", schedule.planning_usage, schedule.planning_cost_usd)
            )
        for node in [*schedule.replaced, *schedule.nodes]:
            events += node.trace
        return events

    @staticmethod
    def _describe_failures(schedule: ScheduleResult) -> str:
        def line(r: NodeResult) -> str:
            return f"{r.node.id} ({r.status.value}: {r.error or r.skipped_because})"

        bad = [r for r in schedule.nodes if r.status is not NodeStatus.DONE]
        label = (
            "budget exhausted" if schedule.status == "budget_exceeded" else "tasks did not complete"
        )
        return f"{label}: " + "; ".join(line(r) for r in bad)
