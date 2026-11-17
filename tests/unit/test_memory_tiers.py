import pytest

from trellis_agent.memory.tiers import WorkingMemory, estimate_tokens
from trellis_agent.types import Message, ToolCall

SYS = Message.system("S" * 40)


def chat(n: int, size: int = 40) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out += [Message.user(f"u{i}".ljust(size, ".")), Message.assistant(f"a{i}".ljust(size, "."))]
    return out


def test_history_that_fits_is_untouched() -> None:
    history = [SYS, *chat(2)]
    kept, dropped = WorkingMemory(token_budget=10_000).fit(history)
    assert kept == history and dropped == []


def test_over_budget_drops_the_oldest_messages_and_keeps_the_system_message() -> None:
    history = [SYS, *chat(10)]
    budget = estimate_tokens([SYS, *chat(10)[-4:]]) + 1  # room for the system + last 4 messages
    kept, dropped = WorkingMemory(token_budget=budget).fit(history)
    assert kept[0] == SYS and kept[1:] == chat(10)[-4:]
    assert dropped == chat(10)[:-4] and estimate_tokens(kept) <= budget


def test_the_newest_message_is_always_kept_even_if_it_alone_exceeds_the_budget() -> None:
    huge = Message.user("x" * 4000)
    kept, dropped = WorkingMemory(token_budget=50).fit([SYS, *chat(3), huge])
    assert kept == [SYS, huge] and len(dropped) == 6


def test_a_tool_result_is_never_left_without_its_tool_call() -> None:
    call = ToolCall("c1", "add", {"a": 1, "b": 2})
    history = [
        Message.user("old question " * 20),
        Message.assistant(None, (call,)),
        Message.tool("c1", "3"),
        Message.assistant("it is 3"),
        Message.user("next?"),
    ]
    tail = history[2:]  # budget fits the tool result onward but not the assistant tool call
    kept, dropped = WorkingMemory(token_budget=estimate_tokens(tail) + 1).fit(history)
    assert all(m.role != "tool" for m in kept[:1]) and kept[0].role != "tool"
    assert kept == history[3:] and dropped == history[:3]


def test_estimate_counts_content_and_tool_arguments() -> None:
    small = estimate_tokens([Message.user("hi")])
    assert estimate_tokens([Message.user("hi" * 100)]) > small
    call = ToolCall("c", "n", {"k": "v" * 400})
    assert estimate_tokens([Message.assistant(None, (call,))]) > 100


def test_invalid_budget() -> None:
    with pytest.raises(ValueError):
        WorkingMemory(token_budget=0)
