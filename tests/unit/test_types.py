import pytest

from trellis_agent.types import Message, ToolCall, Usage


def test_usage_adds_and_totals() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5, cached_tokens=2)
    b = Usage(prompt_tokens=1, completion_tokens=2, estimated=True)
    c = a + b
    assert (c.prompt_tokens, c.completion_tokens, c.cached_tokens) == (11, 7, 2)
    assert c.total_tokens == 18
    assert c.estimated is True  # any estimated component taints the sum
    assert a.estimated is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_tokens": -1},
        {"completion_tokens": -1},
        {"prompt_tokens": 3, "cached_tokens": 4},  # cached is a subset of prompt
    ],
)
def test_usage_rejects_invalid_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Usage(**kwargs)


def test_message_constructors() -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 1})
    assert Message.system("s").role == "system"
    assert Message.user("u").content == "u"
    assistant = Message.assistant(None, tool_calls=(call,))
    assert assistant.tool_calls == (call,)
    tool = Message.tool("c1", "3")
    assert (tool.role, tool.tool_call_id, tool.content) == ("tool", "c1", "3")


def test_tool_message_requires_call_id() -> None:
    with pytest.raises(ValueError):
        Message(role="tool", content="x")


def test_toolcall_defaults_have_no_argument_error() -> None:
    call = ToolCall(id="c1", name="add", arguments={})
    assert call.arguments_error is None and call.raw_arguments is None


# ----------------------------------------------------------------- reasoning tokens
def test_reasoning_tokens_are_optional_and_none_by_default() -> None:
    assert Usage(10, 5).reasoning_tokens is None
    assert (
        "reasoning_tokens" not in Usage(10, 5).to_dict()
    )  # absent when unknown: old reports unchanged


def test_reasoning_tokens_add_up_and_unknown_stays_unknown() -> None:
    assert (
        Usage(1, 10, reasoning_tokens=4) + Usage(1, 10, reasoning_tokens=6)
    ).reasoning_tokens == 10
    assert (Usage(1, 10) + Usage(1, 10)).reasoning_tokens is None  # nothing known: still None
    mixed = Usage(1, 10, reasoning_tokens=4) + Usage(1, 10)
    assert mixed.reasoning_tokens == 4  # the known part is kept (a lower bound), not discarded


def test_reasoning_share_is_a_fraction_of_the_output() -> None:
    assert Usage(1, 40, reasoning_tokens=30).reasoning_share == 0.75
    assert Usage(1, 40).reasoning_share is None
    assert Usage(1, 0, reasoning_tokens=0).reasoning_share is None  # no output: no share


def test_reasoning_tokens_round_trip_and_validate() -> None:
    usage = Usage(5, 9, 2, reasoning_tokens=7)
    assert usage.to_dict()["reasoning_tokens"] == 7 and Usage.from_dict(usage.to_dict()) == usage
    assert Usage.from_dict({"prompt_tokens": 1, "completion_tokens": 2}).reasoning_tokens is None
    for bad in (-1,):
        with pytest.raises(ValueError):
            Usage(1, 2, reasoning_tokens=bad)
    with pytest.raises(ValueError):
        Usage(1, 2, reasoning_tokens=3)  # cannot exceed the completion tokens it is part of
