"""Run a task set against a target, score every run, and account for every cent.

Each (task, repeat) is an independent job with a fresh target and its own session, so runs cannot
contaminate each other. Jobs are issued repeat-major (all tasks once, then all tasks again, ...), so
a budget abort leaves the most even coverage; the report keeps that same stable order regardless of
which job finished first.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from ..cost import BudgetExceeded, CostMeter
from ..llm import LLMClient
from ..types import RunResult, Usage
from .report import ABORTED, FAILED, NOT_RUN, PASSED, CaseResult, EvalReport, cap_content
from .scorers import Scorer, ScoreResult, build_scorer, called_tools, check_success
from .taskset import EvalTask, TaskSet


class Runnable(Protocol):
    """Anything that answers a query: an ``Agent``, a planner-driven agent, a test fake."""

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult: ...


TargetFactory = Callable[[], Runnable]


class EvalRunner:
    def __init__(
        self,
        target_factory: TargetFactory,
        *,
        repeats: int = 1,
        concurrency: int = 4,
        budget_usd: float | None = None,
        meter: CostMeter | None = None,
        judge_llm: LLMClient | None = None,
        include_timing: bool = False,
        clock: Callable[[], float] = time.perf_counter,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if budget_usd is not None and budget_usd < 0:
            raise ValueError("budget_usd must be non-negative")
        self.target_factory = target_factory
        self.repeats = repeats
        self.concurrency = concurrency
        self.budget_usd = budget_usd
        self.meter = meter
        self.judge_llm = judge_llm
        self.include_timing = include_timing
        self.clock = clock
        self.extra_config = dict(config or {})

    async def run(self, taskset: TaskSet) -> EvalReport:
        # Build every scorer first: a typo in a task file fails fast, before any money is spent.
        scorers = {
            t.id: [build_scorer(spec, judge_llm=self.judge_llm) for spec in t.scorers]
            for t in taskset.tasks
        }
        jobs = [(r, t) for r in range(self.repeats) for t in taskset.tasks]
        cases: list[CaseResult | None] = [None] * len(jobs)
        timing: dict[str, float] = {}
        gate = asyncio.Semaphore(self.concurrency)
        spent = 0.0
        stopped = False  # set once the budget is hit; later jobs are not started

        async def one(index: int, repeat: int, task: EvalTask) -> None:
            nonlocal spent, stopped
            async with gate:
                over = self.budget_usd is not None and spent >= self.budget_usd
                if stopped or over:
                    stopped = True
                    cases[index] = CaseResult(task.id, repeat, NOT_RUN)
                    return
                started = self.clock()
                case = await self._execute(task, repeat, scorers[task.id])
                if self.include_timing:
                    timing[case.key] = (self.clock() - started) * 1000
                spent += (case.agent_cost_usd or 0.0) + (case.judge_cost_usd or 0.0)
                if case.status == ABORTED:
                    stopped = True
                cases[index] = case

        await asyncio.gather(*(one(i, r, t) for i, (r, t) in enumerate(jobs)))
        ordered = [c for c in cases if c is not None]
        return EvalReport(
            taskset_name=taskset.name,
            taskset_version=taskset.version,
            taskset_hash=taskset.content_hash,
            config={
                "repeats": self.repeats,
                "concurrency": self.concurrency,
                "budget_usd": self.budget_usd,
                **self.extra_config,
            },
            cases=ordered,
            meter_total_usd=None if self.meter is None else self.meter.total_usd,
            aborted_by_budget=stopped,
            timing=timing if self.include_timing else None,
        )

    async def _execute(self, task: EvalTask, repeat: int, scorers: list[Scorer]) -> CaseResult:
        session_id = f"{task.id}#{repeat}"
        try:
            result = await self.target_factory().arun(task.input, session_id=session_id)
        except BudgetExceeded as exc:
            paid = exc.response
            result = RunResult(
                "",
                [],
                paid.usage if paid else Usage(),
                paid.cost_usd if paid else None,
                [],
                str(exc),
                "budget_exceeded",
            )
        except Exception as exc:  # a crashing target is a failed case, not a crashed evaluation
            result = RunResult(
                "", [], Usage(), None, [], f"{type(exc).__name__}: {exc}", "target_error"
            )

        scores: list[ScoreResult] = []
        aborted = result.stop_reason == "budget_exceeded"
        if not aborted:
            scores.append(check_success(task, result))
            for scorer in scorers:
                is_judge = getattr(scorer, "judge", False)
                if is_judge and result.error is not None:
                    scores.append(
                        ScoreResult(
                            scorer.name, False, 0.0, "skipped: the run itself failed", judge=True
                        )
                    )
                    continue
                try:
                    scores.append(await scorer.score(task, result))
                except BudgetExceeded as exc:
                    paid = exc.response
                    scores.append(
                        ScoreResult(
                            scorer.name,
                            False,
                            0.0,
                            f"judge call exceeded the budget: {exc}",
                            judge=True,
                            error=True,
                            usage=paid.usage if paid else None,
                            cost_usd=paid.cost_usd if paid else None,
                        )
                    )
                    aborted = True
                    break
                except Exception as exc:
                    scores.append(
                        ScoreResult(
                            scorer.name,
                            False,
                            0.0,
                            f"scorer raised {type(exc).__name__}: {exc}",
                            judge=is_judge,
                            error=True,
                        )
                    )

        judged = [s for s in scores if s.judge and s.usage is not None]
        judge_usage = Usage()
        for s in judged:
            assert s.usage is not None
            judge_usage = judge_usage + s.usage
        known = [s.cost_usd for s in judged if s.cost_usd is not None]
        if not judged:
            judge_cost: float | None = 0.0
        else:
            judge_cost = sum(known) if len(known) == len(judged) else None

        all_passed = all(s.passed for s in scores)
        status = ABORTED if aborted else (PASSED if all_passed else FAILED)
        return CaseResult(
            task_id=task.id,
            repeat=repeat,
            status=status,
            content=cap_content(result.content),
            error=result.error,
            stop_reason=result.stop_reason,
            scores=scores,
            usage=result.usage,
            judge_usage=judge_usage,
            agent_cost_usd=result.cost_usd,
            judge_cost_usd=judge_cost,
            judge_calls=len(judged),
            model_requests=sum(1 for e in result.trace if e.type == "model_request"),
            tool_calls=called_tools(result),
        )
