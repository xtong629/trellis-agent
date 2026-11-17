"""Example 04: a planner splits a goal into tasks; role-specific worker agents carry them out.

A "researcher" (with a price-lookup tool) and a "calculator" (with an arithmetic tool) have their
own system prompts and tools. The planner assigns each task a role; the scheduler runs independent
tasks in parallel and passes results to the tasks that depend on them.

    python examples/04_planned.py --mock     # offline, deterministic (rule-based fake models)
    python examples/04_planned.py            # real model, reads LLM_* from env or .env
"""

import asyncio
import json
import re

from _common import parse_args, real_client

from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.llm import LLMClient, RuleLLM
from trellis_agent.planner import PlannedAgent, Planner, Scheduler, WorkerSpec
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.types import Message, ToolCall, ToolSchema

GOAL = "What is the total price of 3 widgets and 2 gadgets?"
PRICES = {"widget": 14, "gadget": 9}


@tool
def lookup_price(item: str) -> str:
    """Look up the unit price of an item."""
    return str(PRICES.get(item, "unknown"))


@tool
def linear_total(price_a: int, qty_a: int, price_b: int, qty_b: int) -> int:
    """Return price_a * qty_a + price_b * qty_b."""
    return price_a * qty_a + price_b * qty_b


# ---- rule-based stand-ins for models (only used with --mock) --------------------------------
PLAN = {
    "tasks": [
        {"id": "t1", "goal": "Look up the unit price of a widget", "kind": "research"},
        {"id": "t2", "goal": "Look up the unit price of a gadget", "kind": "research"},
        {
            "id": "t3",
            "goal": "Compute 3 widgets + 2 gadgets from the prices",
            "kind": "compute",
            "depends_on": ["t1", "t2"],
        },
    ]
}


def _last_user(messages: list[Message]) -> str:
    return next(m.content or "" for m in reversed(messages) if m.role == "user")


def mock_planner(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    return Message.assistant(json.dumps(PLAN))


def mock_researcher(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    seen = [m for m in messages if m.role == "tool"]
    own_task = _last_user(messages).split("Your task", 1)[-1]  # not the shared overall goal
    item = "widget" if "widget" in own_task else "gadget"
    if seen:
        return Message.assistant(f"{item} costs {seen[-1].content}")
    return Message.assistant(None, (ToolCall(f"c-{item}", "lookup_price", {"item": item}),))


def mock_calculator(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    seen = [m for m in messages if m.role == "tool"]
    if seen:
        return Message.assistant(f"total is {seen[-1].content}")
    text = _last_user(messages)
    price_a = int(re.search(r"\[t1\][^\d]*(\d+)", text).group(1))  # type: ignore[union-attr]
    price_b = int(re.search(r"\[t2\][^\d]*(\d+)", text).group(1))  # type: ignore[union-attr]
    args = {"price_a": price_a, "qty_a": 3, "price_b": price_b, "qty_b": 2}
    return Message.assistant(None, (ToolCall("c-total", "linear_total", args),))


def mock_synthesizer(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    numbers = re.findall(r"\d+", _last_user(messages).split("TASK RESULTS:")[1])
    return Message.assistant(f"The total price is {numbers[-1]}.")


async def main() -> None:
    args = parse_args(__doc__ or "")
    llms: dict[str, LLMClient]
    if args.mock:
        llms = {
            "planner": RuleLLM(mock_planner),
            "researcher": RuleLLM(mock_researcher),
            "calculator": RuleLLM(mock_calculator),
            "synthesizer": RuleLLM(mock_synthesizer),
        }
    else:
        shared = real_client()
        llms = dict.fromkeys(("planner", "researcher", "calculator", "synthesizer"), shared)

    researcher = Agent(
        llms["researcher"],
        tools=ToolRegistry([lookup_price]),
        config=AgentConfig(system_prompt="You are a researcher. Use lookup_price to find prices."),
    )
    calculator = Agent(
        llms["calculator"],
        tools=ToolRegistry([linear_total]),
        config=AgentConfig(system_prompt="You are a calculator. Use linear_total for arithmetic."),
    )
    specs = [
        WorkerSpec("research", "looks up facts such as unit prices"),
        WorkerSpec("compute", "does arithmetic on numbers found by others"),
    ]
    scheduler = Scheduler({"research": researcher, "compute": calculator})
    agent = PlannedAgent(Planner(llms["planner"], specs), scheduler, llms["synthesizer"])

    result = await agent.arun(GOAL)
    print("workers: researcher, calculator")
    print("id | kind | depends on | status | attempts")
    started = [e for e in result.trace if e.type == "task_start"]
    ended = {e.data["id"]: e.data for e in result.trace if e.type == "task_end"}
    for event in started:
        info = ended[event.data["id"]]
        node = PLAN["tasks"][int(event.data["id"][1:]) - 1]
        deps = ",".join(node.get("depends_on", [])) or "-"
        print(
            f"{event.data['id']} | {node['kind']} | {deps} | {info['status']} | {info['attempts']}"
        )
    print(f"Answer: {result.content}")
    print(f"tokens: {result.usage.prompt_tokens} in / {result.usage.completion_tokens} out")


if __name__ == "__main__":
    asyncio.run(main())
