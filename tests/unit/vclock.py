"""A virtual clock for scheduler tests: no real waiting, exact timelines.

``VirtualClock.sleep`` parks a coroutine until the driver advances time. ``run`` lets every other
task run until all are blocked in ``sleep``, then jumps to the earliest wake-up time. Because time
only moves when everything is idle, a test can assert exact start/end times.
"""

from __future__ import annotations

import asyncio
import heapq
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

from trellis_agent.types import RunResult, Usage

T = TypeVar("T")


class VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self._waiters: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = 0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._seq += 1
        heapq.heappush(self._waiters, (self.now + delay, self._seq, future))
        await future

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        task = asyncio.ensure_future(coro)
        idle_rounds = 0
        while not task.done():
            for _ in range(60):
                await asyncio.sleep(0)
            while self._waiters and self._waiters[0][2].done():
                heapq.heappop(self._waiters)  # cancelled sleepers
            if task.done():
                break
            if not self._waiters:
                idle_rounds += 1
                if idle_rounds > 3:
                    task.cancel()
                    raise RuntimeError("deadlock: nothing is sleeping and the task is not done")
                continue
            idle_rounds = 0
            wake = self._waiters[0][0]
            self.now = max(self.now, wake)
            while self._waiters and self._waiters[0][0] <= self.now:
                _, _, future = heapq.heappop(self._waiters)
                if not future.done():
                    future.set_result(None)
        return task.result()


@dataclass
class Call:
    prompt: str
    session_id: str
    start: float
    end: float = -1.0
    cancelled: bool = False


@dataclass
class FakeWorker:
    """A worker that takes ``duration`` virtual seconds. ``script`` lists per-call outcomes:
    None/"ok" = succeed, an Exception = raise it, a RunResult = return it."""

    name: str = "w"
    clock: VirtualClock | None = None
    duration: float = 0.0
    script: list[Any] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(10, 5))
    cost: float | None = None
    calls: list[Call] = field(default_factory=list)
    running: int = 0
    peak: int = 0

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        start = self.clock.now if self.clock else 0.0
        call = Call(query, session_id, start)
        self.calls.append(call)
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if self.duration and self.clock:
                await self.clock.sleep(self.duration)
            elif self.duration:
                await asyncio.sleep(self.duration)
        except asyncio.CancelledError:
            call.cancelled = True
            raise
        finally:
            self.running -= 1
            call.end = self.clock.now if self.clock else 0.0
        outcome = self.script.pop(0) if self.script else None
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, RunResult):
            return outcome
        return RunResult(f"{self.name} output", [], self.usage, self.cost, [], None, "end_turn")
