"""Structured trace events and their aggregation.

Event vocabulary (``TraceEvent.type``): ``run_start``, ``model_request``, ``tool_call``,
``tool_result``, ``retry``, ``error``, ``run_end``, plus ``span_start``/``span_end``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .types import TraceEvent, Usage

ROOT_SPAN = "root"


class Tracer:
    """Collects events; spans nest per asyncio task (via ``contextvars``), not per thread."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        sink: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self._clock = clock
        self._sink = sink
        self._events: list[TraceEvent] = []
        self._parents: dict[str, str | None] = {ROOT_SPAN: None}
        self._counter = 0
        self._current: ContextVar[str] = ContextVar("trellis_span", default=ROOT_SPAN)

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def emit(self, type: str, **data: Any) -> None:
        span = self._current.get()
        event = TraceEvent(type, self._clock(), dict(data), span, self._parents[span])
        self._events.append(event)
        if self._sink is not None:
            self._sink(event)

    @contextmanager
    def span(self, name: str) -> Iterator[str]:
        self._counter += 1
        span_id = f"s{self._counter}"
        self._parents[span_id] = self._current.get()
        token = self._current.set(span_id)
        self.emit("span_start", name=name)
        error: str | None = None
        try:
            yield span_id
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            data: dict[str, Any] = {"name": name}
            if error is not None:
                data["error"] = error
            self.emit("span_end", **data)
            self._current.reset(token)


@dataclass
class TraceSummary:
    event_count: int = 0
    model_request_count: int = 0
    model_latency_ms: float = 0.0
    tool_call_count: int = 0
    tool_success_count: int = 0
    tool_latency_ms: float = 0.0
    error_count: int = 0
    error_categories: dict[str, int] = field(default_factory=dict)
    retry_count: int = 0
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    unpriced_model_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "event_count": self.event_count,
            "model_request_count": self.model_request_count,
            "model_latency_ms": self.model_latency_ms,
            "tool_call_count": self.tool_call_count,
            "tool_success_count": self.tool_success_count,
            "tool_latency_ms": self.tool_latency_ms,
            "error_count": self.error_count,
            "error_categories": dict(self.error_categories),
            "retry_count": self.retry_count,
            "usage": self.usage.to_dict(),
            "cost_usd": self.cost_usd,
            "unpriced_model_requests": self.unpriced_model_requests,
        }
        if self.usage.reasoning_share is not None:  # only when the provider reported it
            data["reasoning_share"] = self.usage.reasoning_share
        return data


def _usage_from_dict(raw: Any) -> Usage:
    return Usage.from_dict(raw) if isinstance(raw, dict) else Usage()


def summarize(events: Iterable[TraceEvent]) -> TraceSummary:
    """Aggregate events. ``cost_usd`` is None unless at least one request carried a price."""
    s = TraceSummary()
    priced = False
    cost = 0.0
    for event in events:
        s.event_count += 1
        data = event.data
        if event.type == "model_request":
            s.model_request_count += 1
            s.model_latency_ms += float(data.get("latency_ms") or 0.0)
            s.usage = s.usage + _usage_from_dict(data.get("usage"))
            if data.get("cost_usd") is None:
                s.unpriced_model_requests += 1
            else:
                priced = True
                cost += float(data["cost_usd"])
        elif event.type == "tool_call":
            s.tool_call_count += 1
        elif event.type == "tool_result":
            s.tool_latency_ms += float(data.get("latency_ms") or 0.0)
            if not data.get("error"):
                s.tool_success_count += 1
        elif event.type == "retry":
            s.retry_count += 1
        elif event.type == "error":
            s.error_count += 1
            category = str(data.get("category") or "unknown")
            s.error_categories[category] = s.error_categories.get(category, 0) + 1
    s.cost_usd = cost if priced else None
    return s
