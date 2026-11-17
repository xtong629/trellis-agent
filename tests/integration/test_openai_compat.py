"""Contract tests: OpenAICompatClient against a fake OpenAI-compatible HTTP server,
including malformed responses that real providers do produce."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from tests.conftest import FakeOpenAI
from trellis_agent.agent import Agent
from trellis_agent.cost import BudgetExceeded, CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import LLMError
from trellis_agent.llm.openai_compat import OpenAICompatClient
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.types import Message, ToolCall, ToolSchema, Usage

SCHEMA = ToolSchema(
    "add", "Add two ints.", {"type": "object", "properties": {"a": {"type": "integer"}}}
)


@pytest.fixture
async def make_client(fake_openai: FakeOpenAI) -> AsyncIterator[type]:
    created: list[OpenAICompatClient] = []

    def factory(**kw: object) -> OpenAICompatClient:
        http = httpx.AsyncClient(trust_env=False)  # never route loopback through a proxy
        client = OpenAICompatClient(
            base_url=fake_openai.base_url,
            api_key="sk-test",
            model="fake-model",
            max_retries=0,
            http_client=http,
            **kw,  # type: ignore[arg-type]
        )
        created.append(client)
        return client

    yield factory  # type: ignore[misc]
    for client in created:
        await client.aclose()


# ----------------------------------------------------------------- happy path
async def test_plain_completion_is_parsed(fake_openai: FakeOpenAI, make_client: type) -> None:
    fake_openai.reply(
        {"content": "hello"},
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    )
    client = make_client()
    response = await client.chat([Message.user("hi")])
    assert response.message == Message.assistant("hello")
    assert response.usage == Usage(100, 20, 40, estimated=False)
    assert response.finish_reason == "stop" and response.model == "fake-model"
    assert response.latency_ms > 0 and response.cost_usd is None


async def test_request_wire_format(fake_openai: FakeOpenAI, make_client: type) -> None:
    fake_openai.reply({"content": "ok"})
    history = [
        Message.system("sys"),
        Message.user("q"),
        Message.assistant(None, (ToolCall("c1", "add", {"a": 1}),)),
        Message.tool("c1", "1"),
    ]
    await make_client().chat(history, tools=[SCHEMA], temperature=0.3, max_tokens=50)
    body = fake_openai.requests[0]
    assert body["model"] == "fake-model" and body["temperature"] == 0.3 and body["max_tokens"] == 50
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "add", "arguments": '{"a": 1}'}}
        ],
    }
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "1"}
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two ints.",
                "parameters": SCHEMA.parameters,
            },
        }
    ]
    assert fake_openai.headers[0]["authorization"] == "Bearer sk-test"


async def test_optional_fields_are_omitted_when_unset(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply({"content": "ok"})
    await make_client().chat([Message.user("q")])
    assert "tools" not in fake_openai.requests[0] and "max_tokens" not in fake_openai.requests[0]


async def test_tool_calls_are_parsed(fake_openai: FakeOpenAI, make_client: type) -> None:
    fake_openai.reply(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'},
                }
            ],
        }
    )
    response = await make_client().chat([Message.user("q")], tools=[SCHEMA])
    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls == (ToolCall("call_9", "add", {"a": 2, "b": 3}),)


# ----------------------------------------------------------------- malformed responses
async def test_missing_usage_is_estimated_and_flagged(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.raw(
        200,
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "a" * 40},
                    "finish_reason": "stop",
                }
            ],
        },
    )
    response = await make_client().chat([Message.user("b" * 80)])
    assert response.usage.estimated is True
    assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0


async def test_missing_usage_taints_the_meter(fake_openai: FakeOpenAI, make_client: type) -> None:
    fake_openai.raw(
        200,
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        },
    )
    meter = CostMeter(PriceTable({"fake-model": ModelPrice(1.0, 1.0)}))
    await make_client(meter=meter).chat([Message.user("q")])
    assert meter.any_estimated is True


@pytest.mark.parametrize(
    ("raw_args", "fragment"),
    [
        ("{not json", "Expecting"),
        ('{"a": 1', "Expecting"),
        ("[1, 2]", "JSON object"),  # valid JSON, but not an object
        ('"just a string"', "JSON object"),
    ],
)
async def test_invalid_tool_call_arguments_are_surfaced_not_raised(
    fake_openai: FakeOpenAI, make_client: type, raw_args: str, fragment: str
) -> None:
    fake_openai.reply(
        {
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "add", "arguments": raw_args}}
            ],
        }
    )
    response = await make_client().chat([Message.user("q")], tools=[SCHEMA])
    (call,) = response.message.tool_calls
    assert call.arguments == {} and call.raw_arguments == raw_args
    assert call.arguments_error is not None and fragment in call.arguments_error


async def test_empty_arguments_string_means_no_arguments(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply(
        {
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "ping", "arguments": ""}}
            ],
        }
    )
    (call,) = (await make_client().chat([Message.user("q")])).message.tool_calls
    assert call.arguments == {} and call.arguments_error is None


async def test_missing_tool_call_id_is_synthesised(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply(
        {
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "a", "arguments": "{}"}},
                {"type": "function", "function": {"name": "b", "arguments": "{}"}},
            ],
        }
    )
    calls = (await make_client().chat([Message.user("q")])).message.tool_calls
    assert [c.id for c in calls] == ["call_0", "call_1"]


async def test_history_with_unparseable_arguments_is_resent_as_valid_json(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply({"content": "ok"})
    broken = ToolCall("c1", "add", {}, raw_arguments="{oops", arguments_error="bad")
    history = [
        Message.user("q"),
        Message.assistant(None, (broken,)),
        Message.tool("c1", "Error: x"),
    ]
    await make_client().chat(history)
    sent = fake_openai.requests[0]["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {}  # strict providers reject invalid JSON in history


async def test_no_choices_raises_llm_error(fake_openai: FakeOpenAI, make_client: type) -> None:
    fake_openai.raw(
        200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []}
    )
    with pytest.raises(LLMError, match="choices"):
        await make_client().chat([Message.user("q")])


@pytest.mark.parametrize("status", [401, 429, 500])
async def test_http_errors_become_llm_errors(
    fake_openai: FakeOpenAI, make_client: type, status: int
) -> None:
    fake_openai.raw(status, {"error": {"message": "nope"}})
    with pytest.raises(LLMError, match=str(status)):
        await make_client().chat([Message.user("q")])


async def test_non_json_body_becomes_llm_error(fake_openai: FakeOpenAI, make_client: type) -> None:
    fake_openai.raw(200, b"<html>gateway</html>")
    with pytest.raises(LLMError):
        await make_client().chat([Message.user("q")])


async def test_connection_failure_becomes_llm_error(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.stop()  # nothing listening any more
    with pytest.raises(LLMError, match="connect"):
        await make_client().chat([Message.user("q")])


# ----------------------------------------------------------------- cost metering
async def test_spend_is_recorded_with_tag_and_budget_enforced(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply(
        {"content": "a"},
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000},
    )
    fake_openai.reply(
        {"content": "b"},
        usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000},
    )
    meter = CostMeter(PriceTable({"fake-model": ModelPrice(1.0, 1.0)}), budget_usd=1.5)
    client = make_client(meter=meter)
    first = await client.chat([Message.user("q")], tag="plan")
    assert first.cost_usd == pytest.approx(1.0) and meter.by_tag == {"plan": pytest.approx(1.0)}
    with pytest.raises(BudgetExceeded):
        await client.chat([Message.user("q")], tag="plan")
    assert meter.total_usd == pytest.approx(2.0)


# ----------------------------------------------------------------- end to end with the Agent
async def test_agent_round_trip_and_error_feedback_over_http(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def tool_reply(call_id: str, arguments: str) -> None:
        fake_openai.reply(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "add", "arguments": arguments},
                    }
                ],
            }
        )

    tool_reply("c1", "{broken")  # 1st: malformed arguments
    tool_reply("c2", '{"a": 2, "b": 3}')  # 2nd: model corrects itself
    fake_openai.reply({"content": "The sum is 5."})  # 3rd: final answer

    agent = Agent(make_client(), tools=ToolRegistry([add]))
    result = await agent.arun("what is 2+3?")

    assert result.content == "The sum is 5." and result.error is None
    second_request_tool_msg = fake_openai.requests[1]["messages"][-1]
    assert second_request_tool_msg["role"] == "tool"
    assert "not valid JSON" in second_request_tool_msg["content"]  # error reached the model
    assert fake_openai.requests[2]["messages"][-1]["content"] == "5"
    assert result.usage == Usage(36, 21)  # 3 calls x (12, 7) from the server's own usage


async def test_timeout_becomes_llm_error(
    fake_openai: FakeOpenAI, make_client: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openai

    client = make_client()

    async def slow(**_: object) -> None:
        raise openai.APITimeoutError(request=httpx.Request("POST", fake_openai.base_url))

    monkeypatch.setattr(client._client.chat.completions, "create", slow)
    with pytest.raises(LLMError, match="timed out"):
        await client.chat([Message.user("q")])


async def test_budget_exceeded_carries_the_paid_for_response_over_http(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply(
        {"content": "costly"},
        usage={"prompt_tokens": 2_000_000, "completion_tokens": 0, "total_tokens": 2_000_000},
    )
    meter = CostMeter(PriceTable({"fake-model": ModelPrice(1.0, 1.0)}), budget_usd=1.0)
    with pytest.raises(BudgetExceeded) as info:
        await make_client(meter=meter).chat([Message.user("q")], tag="judge")
    response = info.value.response
    assert response is not None and response.message.content == "costly"
    assert response.cost_usd == pytest.approx(2.0) and response.usage.prompt_tokens == 2_000_000


# ----------------------------------------------------------------- extra request parameters
async def test_extra_params_are_sent_as_top_level_body_fields(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply({"content": "ok"})
    client = make_client(extra_params={"reasoning_effort": "low", "thinking": {"type": "enabled"}})
    await client.chat([Message.user("q")])
    body = fake_openai.requests[0]
    assert body["reasoning_effort"] == "low" and body["thinking"] == {"type": "enabled"}
    assert body["model"] == "fake-model" and "extra_body" not in body  # merged, not nested


async def test_per_call_extra_params_override_the_client_defaults(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply({"content": "a"})
    fake_openai.reply({"content": "b"})
    client = make_client(extra_params={"reasoning_effort": "low", "keep": 1})
    await client.chat([Message.user("q")], extra_params={"reasoning_effort": "high"})
    await client.chat([Message.user("q")])
    assert (
        fake_openai.requests[0]["reasoning_effort"] == "high"
        and fake_openai.requests[0]["keep"] == 1
    )
    assert fake_openai.requests[1]["reasoning_effort"] == "low"


async def test_no_extra_params_means_no_extra_fields(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply({"content": "ok"})
    await make_client().chat([Message.user("q")])
    assert set(fake_openai.requests[0]) == {"model", "messages", "temperature"}


@pytest.mark.parametrize(
    "name", ["model", "messages", "tools", "temperature", "max_tokens", "stream"]
)
async def test_extra_params_cannot_override_what_the_client_controls(
    make_client: type, name: str
) -> None:
    with pytest.raises(ValueError, match=name):
        make_client(extra_params={name: 1})
    client = make_client()
    with pytest.raises(ValueError, match=name):
        await client.chat([Message.user("q")], extra_params={name: 1})


# ----------------------------------------------------------------- reasoning tokens
async def test_reasoning_tokens_are_read_from_the_usage_details(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    fake_openai.reply(
        {"content": "ok"},
        usage={
            "prompt_tokens": 20,
            "completion_tokens": 50,
            "total_tokens": 70,
            "completion_tokens_details": {"reasoning_tokens": 40, "audio_tokens": None},
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    )
    usage = (await make_client().chat([Message.user("q")])).usage
    assert usage == Usage(20, 50, 4, reasoning_tokens=40) and usage.reasoning_share == 0.8


@pytest.mark.parametrize(
    "details",
    [None, {}, {"reasoning_tokens": None}],
    ids=["no-details", "empty-details", "null-reasoning"],
)
async def test_missing_reasoning_tokens_stay_none(
    fake_openai: FakeOpenAI, make_client: type, details: object
) -> None:
    usage: dict[str, object] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    if details is not None:
        usage["completion_tokens_details"] = details
    fake_openai.reply({"content": "ok"}, usage=usage)
    assert (await make_client().chat([Message.user("q")])).usage.reasoning_tokens is None


async def test_estimated_usage_never_claims_reasoning_tokens(
    fake_openai: FakeOpenAI, make_client: type
) -> None:
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop",
    }
    fake_openai.raw(
        200,
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [choice],
        },
    )  # no usage object at all
    usage = (await make_client().chat([Message.user("q")])).usage
    assert usage.estimated and usage.reasoning_tokens is None
