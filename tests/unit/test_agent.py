import asyncio

import pytest

from trellis_agent.agent import Agent, AgentConfig, AgentEvent
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import LLMError, RuleLLM, ScriptedLLM
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.trace import summarize
from trellis_agent.types import Message, RunResult, ToolCall, TraceEvent, Usage


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def calls(*items: ToolCall) -> Message:
    return Message.assistant(None, tuple(items))


def add_call(id_: str = "c1", a: int = 2, b: int = 3) -> ToolCall:
    return ToolCall(id_, "add", {"a": a, "b": b})


def make_agent(script: list[Message], **kw: object) -> tuple[Agent, ScriptedLLM]:
    llm = ScriptedLLM(script)
    return Agent(llm, tools=ToolRegistry([add]), **kw), llm  # type: ignore[arg-type]


# ----------------------------------------------------------------- basic flow
async def test_plain_answer_without_tools() -> None:
    llm = ScriptedLLM([Message.assistant("hello")])
    agent = Agent(llm, config=AgentConfig(system_prompt="Be brief."))
    result = await agent.arun("hi")
    assert result.content == "hello" and result.error is None and result.stop_reason == "end_turn"
    assert [m.role for m in result.messages] == ["system", "user", "assistant"]
    assert result.usage == Usage(10, 5)
    assert llm.calls[0].tools is None  # no registry => no tools offered
    assert [e.type for e in result.trace] == ["run_start", "model_request", "run_end"]


async def test_tool_round_trip_feeds_result_back_to_the_model() -> None:
    agent, llm = make_agent([calls(add_call()), Message.assistant("The sum is 5.")])
    result = await agent.arun("what is 2+3?")
    assert result.content == "The sum is 5."
    second_call_messages = llm.calls[1].messages
    assert second_call_messages[-2].tool_calls == (add_call(),)  # assistant request kept
    assert second_call_messages[-1] == Message.tool("c1", "5")
    assert result.usage == Usage(20, 10)  # two model calls
    assert [s.name for s in (llm.calls[0].tools or ())] == ["add"]
    types = [e.type for e in result.trace]
    assert types == [
        "run_start", "model_request", "tool_call", "tool_result", "model_request", "run_end",
    ]  # fmt: skip


async def test_several_tool_calls_in_one_message_keep_order_and_ids() -> None:
    agent, llm = make_agent(
        [calls(add_call("a", 1, 1), add_call("b", 2, 2)), Message.assistant("ok")]
    )
    await agent.arun("go")
    tool_msgs = [m for m in llm.calls[1].messages if m.role == "tool"]
    assert [(m.tool_call_id, m.content) for m in tool_msgs] == [("a", "2"), ("b", "4")]


# ----------------------------------------------------------------- tool errors reach the model
async def test_invalid_arguments_are_returned_to_the_model_which_can_retry() -> None:
    bad = ToolCall("c1", "add", {"a": "two", "b": 3})
    agent, llm = make_agent([calls(bad), calls(add_call("c2")), Message.assistant("5")])
    result = await agent.arun("add")
    first_result = llm.calls[1].messages[-1]
    assert first_result.role == "tool" and "not of type 'integer'" in (first_result.content or "")
    assert result.content == "5" and result.error is None
    errors = [e for e in result.trace if e.type == "tool_result" and e.data["error"]]
    assert len(errors) == 1 and errors[0].data["error_type"] == "invalid_arguments"


async def test_unknown_tool_and_tool_exception_do_not_crash_the_run() -> None:
    @tool
    def boom(x: int) -> int:
        """Fails."""
        raise RuntimeError("kaput")

    script = [
        calls(ToolCall("u", "missing", {}), ToolCall("b", "boom", {"x": 1})),
        Message.assistant("recovered"),
    ]
    llm = ScriptedLLM(script)
    result = await Agent(llm, tools=ToolRegistry([boom])).arun("x")
    contents = [m.content for m in llm.calls[1].messages if m.role == "tool"]
    assert "unknown tool 'missing'" in (contents[0] or "")
    assert "kaput" in (contents[1] or "")
    assert result.content == "recovered" and result.error is None


# ----------------------------------------------------------------- stopping conditions
async def test_max_steps_stops_a_looping_model() -> None:
    llm = RuleLLM(lambda m, t: calls(add_call()))
    agent = Agent(llm, tools=ToolRegistry([add]), config=AgentConfig(max_steps=3))
    result = await agent.arun("loop forever")
    assert result.stop_reason == "max_steps" and "max_steps" in (result.error or "")
    assert len(llm.calls) == 3
    assert result.messages[-1].role == "tool"


async def test_budget_exceeded_ends_the_run_without_raising() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 1.0)}), budget_usd=0.000001)
    llm = ScriptedLLM(
        [calls(add_call()), Message.assistant("never reached")],
        usage_per_call=Usage(1_000_000, 0),
        meter=meter,
    )
    result = await Agent(llm, tools=ToolRegistry([add])).arun("q")
    assert result.stop_reason == "budget_exceeded" and "budget" in (result.error or "")
    assert summarize(result.trace).error_categories == {"budget_exceeded": 1}
    assert len(llm.calls) == 1


async def test_llm_failure_is_reported_in_the_result() -> None:
    def fail(messages: list[Message], tools: object) -> Message:
        raise LLMError("provider down")

    result = await Agent(RuleLLM(fail)).arun("q")
    assert result.stop_reason == "llm_error" and "provider down" in (result.error or "")
    assert result.content == ""


async def test_cost_is_summed_when_priced_and_none_otherwise() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 2.0)}))
    llm = ScriptedLLM(
        [calls(add_call()), Message.assistant("5")],
        usage_per_call=Usage(1_000_000, 500_000),
        meter=meter,
    )
    priced = await Agent(llm, tools=ToolRegistry([add])).arun("q")
    assert priced.cost_usd == pytest.approx(4.0)
    assert summarize(priced.trace).cost_usd == pytest.approx(4.0)
    free, _ = make_agent([Message.assistant("x")])
    assert (await free.arun("q")).cost_usd is None


# ----------------------------------------------------------------- stream is the only loop
async def test_stream_yields_events_in_order_and_finishes_with_the_result() -> None:
    agent, _ = make_agent([calls(add_call()), Message.assistant("5")])
    events = [e async for e in agent.stream("q")]
    assert [e.type for e in events] == ["tool_call", "tool_result", "text", "done"]
    assert all(isinstance(e, AgentEvent) for e in events)
    assert events[0].data["name"] == "add" and events[1].data["content"] == "5"
    assert events[2].data["content"] == "5"
    done = events[-1].result
    assert isinstance(done, RunResult) and done.content == "5"


async def test_arun_equals_the_result_carried_by_stream() -> None:
    def script() -> list[Message]:
        return [calls(add_call()), Message.assistant("5")]

    streamed = [e async for e in make_agent(script())[0].stream("q")][-1].result
    collected = await make_agent(script())[0].arun("q")
    assert streamed is not None
    assert (streamed.content, streamed.messages, streamed.usage, streamed.stop_reason) == (
        collected.content, collected.messages, collected.usage, collected.stop_reason,
    )  # fmt: skip
    assert [e.type for e in streamed.trace] == [e.type for e in collected.trace]


def test_sync_run_wraps_arun() -> None:
    agent, _ = make_agent([Message.assistant("sync ok")])
    assert agent.run("q").content == "sync ok"


async def test_sync_run_inside_a_running_loop_is_refused() -> None:
    agent, _ = make_agent([Message.assistant("x")])
    with pytest.raises(RuntimeError, match="arun"):
        agent.run("q")


# ----------------------------------------------------------------- sessions, config, tracing
async def test_sessions_keep_history_and_are_isolated() -> None:
    llm = RuleLLM(lambda m, t: Message.assistant(f"seen {len(m)}"))
    agent = Agent(llm, config=AgentConfig(system_prompt="sys"))
    r1 = await agent.arun("one", session_id="a")
    r2 = await agent.arun("two", session_id="a")
    r3 = await agent.arun("three", session_id="b")
    assert r1.content == "seen 2"  # system + user
    assert r2.content == "seen 4"  # system + user + assistant + user
    assert r3.content == "seen 2"  # other session starts fresh
    assert sum(1 for m in r2.messages if m.role == "system") == 1  # system prompt not duplicated


async def test_no_system_message_when_prompt_is_empty() -> None:
    llm = ScriptedLLM([Message.assistant("x")])
    await Agent(llm).arun("q")
    assert [m.role for m in llm.calls[0].messages] == ["user"]


async def test_config_temperature_reaches_the_model() -> None:
    llm = ScriptedLLM([Message.assistant("x")])
    await Agent(llm, config=AgentConfig(temperature=0.7)).arun("q")
    assert llm.calls[0].temperature == 0.7


async def test_trace_sink_receives_run_events() -> None:
    seen: list[TraceEvent] = []
    agent = Agent(ScriptedLLM([Message.assistant("x")]), trace_sink=seen.append)
    await agent.arun("q")
    assert [e.type for e in seen] == ["run_start", "model_request", "run_end"]


async def test_concurrent_runs_get_separate_traces() -> None:
    llm = RuleLLM(lambda m, t: Message.assistant("x"))
    agent = Agent(llm)
    r1, r2 = await asyncio.gather(agent.arun("a", session_id="1"), agent.arun("b", session_id="2"))
    assert [e.type for e in r1.trace] == [e.type for e in r2.trace] == [
        "run_start", "model_request", "run_end",
    ]  # fmt: skip
    assert r1.trace[0].data["query"] == "a" and r2.trace[0].data["query"] == "b"
