import asyncio
import itertools

import pytest

from trellis_agent.trace import Tracer, TraceSummary, summarize
from trellis_agent.types import TraceEvent


def make_tracer() -> Tracer:
    counter = itertools.count()
    return Tracer(clock=lambda: float(next(counter)))


def test_emit_records_type_data_and_timestamp() -> None:
    tracer = make_tracer()
    tracer.emit("custom", a=1)
    (event,) = tracer.events
    assert (event.type, event.data) == ("custom", {"a": 1})
    assert event.ts == 0.0
    assert event.span_id == "root" and event.parent_id is None


def test_events_property_returns_a_copy() -> None:
    tracer = make_tracer()
    tracer.emit("x")
    tracer.events.clear()
    assert len(tracer.events) == 1


def test_sink_receives_every_event() -> None:
    seen: list[TraceEvent] = []
    tracer = Tracer(sink=seen.append)
    tracer.emit("a")
    tracer.emit("b")
    assert [e.type for e in seen] == ["a", "b"]


def test_spans_nest_and_restore() -> None:
    tracer = make_tracer()
    with tracer.span("outer") as outer:
        tracer.emit("in_outer")
        with tracer.span("inner") as inner:
            tracer.emit("in_inner")
        tracer.emit("back_in_outer")
    tracer.emit("after")
    by_type = {e.type: e for e in tracer.events}
    assert by_type["in_outer"].span_id == outer
    assert by_type["in_inner"].span_id == inner
    assert by_type["in_inner"].parent_id == outer
    assert by_type["back_in_outer"].span_id == outer
    assert by_type["after"].span_id == "root"


def test_span_start_and_end_events_are_emitted() -> None:
    tracer = make_tracer()
    with tracer.span("work"):
        pass
    assert [e.type for e in tracer.events] == ["span_start", "span_end"]
    assert tracer.events[0].data["name"] == "work"


def test_span_closes_on_exception() -> None:
    tracer = make_tracer()
    with pytest.raises(RuntimeError), tracer.span("boom"):
        raise RuntimeError("x")
    assert tracer.events[-1].type == "span_end"
    assert tracer.events[-1].data["error"] == "RuntimeError"
    tracer.emit("after")
    assert tracer.events[-1].span_id == "root"


async def test_concurrent_tasks_do_not_mix_spans() -> None:
    tracer = make_tracer()

    async def worker(name: str) -> None:
        with tracer.span(name):
            await asyncio.sleep(0)
            tracer.emit("work", who=name)
            await asyncio.sleep(0)

    await asyncio.gather(worker("a"), worker("b"))
    starts = {e.data["name"]: e.span_id for e in tracer.events if e.type == "span_start"}
    for event in tracer.events:
        if event.type == "work":
            assert event.span_id == starts[event.data["who"]]


def test_summarize_aggregates_a_run() -> None:
    tracer = make_tracer()
    tracer.emit(
        "model_request",
        latency_ms=100.0,
        usage={"prompt_tokens": 10, "completion_tokens": 4, "cached_tokens": 0, "estimated": False},
        cost_usd=0.5,
    )
    tracer.emit("tool_call", name="add")
    tracer.emit("tool_result", name="add", latency_ms=7.0, error=False)
    tracer.emit("tool_call", name="bad")
    tracer.emit("tool_result", name="bad", latency_ms=3.0, error=True)
    tracer.emit("retry", reason="timeout")
    tracer.emit("error", category="llm_error", message="x")
    tracer.emit("error", category="llm_error", message="y")
    tracer.emit(
        "model_request",
        latency_ms=50.0,
        usage={"prompt_tokens": 20, "completion_tokens": 6, "cached_tokens": 5, "estimated": True},
        cost_usd=None,
    )
    s = summarize(tracer.events)
    assert isinstance(s, TraceSummary)
    assert s.model_request_count == 2 and s.model_latency_ms == 150.0
    assert s.tool_call_count == 2 and s.tool_success_count == 1 and s.tool_latency_ms == 10.0
    assert s.error_count == 2 and s.error_categories == {"llm_error": 2}
    assert s.retry_count == 1
    assert (s.usage.prompt_tokens, s.usage.completion_tokens, s.usage.cached_tokens) == (30, 10, 5)
    assert s.usage.estimated is True
    assert s.cost_usd == 0.5  # only priced calls contribute
    assert s.unpriced_model_requests == 1
    assert s.event_count == len(tracer.events)


def test_summarize_empty_has_no_cost() -> None:
    s = summarize([])
    assert s.cost_usd is None and s.event_count == 0
    assert s.to_dict()["usage"]["prompt_tokens"] == 0


def test_summarize_aggregates_reasoning_tokens_and_shows_the_share() -> None:
    tracer = make_tracer()
    for reasoning in (30, 10):
        usage = {"prompt_tokens": 5, "completion_tokens": 40, "reasoning_tokens": reasoning}
        tracer.emit("model_request", usage=usage, latency_ms=1.0, cost_usd=None)
    s = summarize(tracer.events)
    assert s.usage.reasoning_tokens == 40 and s.usage.reasoning_share == 0.5
    assert s.to_dict()["reasoning_share"] == 0.5


def test_summary_has_no_share_when_reasoning_is_unknown() -> None:
    tracer = make_tracer()
    tracer.emit(
        "model_request", usage={"prompt_tokens": 5, "completion_tokens": 40}, latency_ms=1.0
    )
    d = summarize(tracer.events).to_dict()
    assert d["usage"].get("reasoning_tokens") is None and "reasoning_share" not in d
