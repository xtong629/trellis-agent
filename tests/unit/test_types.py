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
