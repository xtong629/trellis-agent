"""Example 02: tools, argument validation and error feedback.

The mock model first sends a *bad* argument (a string where an int is required). The registry
rejects it, the error text goes back to the model, and the model corrects itself -- the same loop a
real model goes through.

    python examples/02_tools_agent.py --mock
"""

import asyncio

from _common import parse_args, pick

from trellis_agent.agent import Agent
from trellis_agent.llm import RuleLLM
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.trace import summarize
from trellis_agent.types import Message, ToolCall, ToolSchema


@tool
def get_weather(city: str) -> dict[str, str]:
    """Look up the current weather for a city."""
    table = {"Tokyo": {"temp": "18C", "sky": "sunny"}, "Oslo": {"temp": "2C", "sky": "snow"}}
    return table.get(city, {"temp": "unknown", "sky": "unknown"})


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def scripted_rule(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """A tiny rule-based stand-in for a model (decides from what it has seen so far)."""
    tool_results = [m for m in messages if m.role == "tool"]
    if not tool_results:
        return Message.assistant(
            None,
            (
                ToolCall("c1", "get_weather", {"city": "Tokyo"}),
                ToolCall("c2", "add", {"a": "18", "b": 7}),
            ),
        )
    if any((m.content or "").startswith("Error:") for m in tool_results) and len(tool_results) == 2:
        return Message.assistant(None, (ToolCall("c3", "add", {"a": 18, "b": 7}),))
    return Message.assistant("Tokyo is 18C and sunny; 18 + 7 = 25.")


async def main() -> None:
    args = parse_args(__doc__ or "")
    llm = pick(args.mock, RuleLLM(scripted_rule))
    agent = Agent(llm, tools=ToolRegistry([get_weather, add]))
    result = await agent.arun("What's the weather in Tokyo, and what is 18 + 7?")
    summary = summarize(result.trace)
    print(f"Answer: {result.content}")
    print(f"tool calls: {summary.tool_call_count} ({summary.tool_success_count} ok)")
    for event in result.trace:
        if event.type == "tool_result" and event.data["error"]:
            print(f"error: {event.data['error_type']} (tool {event.data['name']})")


if __name__ == "__main__":
    asyncio.run(main())
