"""Example 06: reflection -- a checker reviews the answer; the agent retries once with the reason.

The mock model first answers with the price of one widget although three were asked for. The
mock checker sees the task, the tool calls and the answer, rejects it with a reason, and the agent
answers again. With a real model, the same model plays both parts.

    python examples/06_reflection.py --mock
"""

import asyncio
import json

from _common import parse_args, pick

from trellis_agent.agent import Agent
from trellis_agent.llm import RuleLLM
from trellis_agent.reflect import LLMReflector, ReflectingAgent, ReflectionRecord
from trellis_agent.tools import ToolRegistry, tool
from trellis_agent.types import Message, ToolCall, ToolSchema


@tool
def get_price(item: str) -> int:
    """Look up the unit price of an item."""
    return {"widget": 14, "gadget": 9}[item]


def model_rule(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """Answers too fast the first time; after the reviewer's note, multiplies by the quantity."""
    if messages[-1].role == "user" and "reviewer" not in (messages[-1].content or ""):
        return Message.assistant(None, (ToolCall("c1", "get_price", {"item": "widget"}),))
    if "reviewer" in (messages[-1].content or ""):
        return Message.assistant("3 widgets at 14 each cost 42.\nANSWER: 42")
    return Message.assistant("A widget costs 14.\nANSWER: 14")


def checker_rule(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    answer = (messages[-1].content or "").rsplit("FINAL ANSWER:", 1)[-1]
    if "ANSWER: 14" in answer:
        verdict = {"ok": False, "reason": "the task asks for 3 widgets; 14 is the price of one"}
    else:
        verdict = {"ok": True}
    return Message.assistant(json.dumps(verdict))


async def main() -> None:
    args = parse_args(__doc__ or "")
    llm = pick(args.mock, RuleLLM(model_rule))
    checker = pick(args.mock, RuleLLM(checker_rule))
    records: list[ReflectionRecord] = []
    agent = ReflectingAgent(
        Agent(llm, tools=ToolRegistry([get_price])), LLMReflector(checker), on_record=records.append
    )
    result = await agent.arun("How much do 3 widgets cost? End with a line 'ANSWER: <number>'.")
    (record,) = records
    print(f"first answer : {record.first_answer.splitlines()[-1]}")
    print(f"checker      : ok={record.verdict_ok} ({record.reason})")
    print(f"final answer : {result.content.splitlines()[-1]}")
    calls = sum(1 for e in result.trace if e.type == "model_request")
    print(f"model calls  : {calls} (lookup, answer, check, retry); stop: {result.stop_reason}")


if __name__ == "__main__":
    asyncio.run(main())
