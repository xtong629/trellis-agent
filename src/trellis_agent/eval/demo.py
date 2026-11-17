"""An offline demo target and mock models, so ``evals/smoke`` runs with no API key.

The mock models are tiny rule-based fakes. They exist to exercise the agent loop and the evaluation
machinery deterministically; passing the smoke set says nothing about real-model quality.
"""

from __future__ import annotations

import json
import re

from ..agent import Agent, AgentConfig
from ..cost import CostMeter
from ..llm import LLMClient, RuleLLM
from ..tools import ToolRegistry, tool
from ..types import Message, ToolCall, ToolSchema, Usage

_CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Norway": "Oslo"}
_ADD = re.compile(r"(-?\d+)\s*\+\s*(-?\d+)")
_MUL = re.compile(r"(-?\d+)\s*\*\s*(-?\d+)")
_CAPITAL = re.compile(r"capital of ([A-Za-z ]+?)\s*\??\s*$", re.IGNORECASE)

SYSTEM_PROMPT = "You are a careful assistant. Use tools for arithmetic and lookups."


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def lookup_capital(country: str) -> str:
    """Return the capital city of a country."""
    return _CAPITALS.get(country, "unknown")


def demo_tools() -> ToolRegistry:
    return ToolRegistry([add, multiply, lookup_capital])


def make_target(llm: LLMClient) -> Agent:
    """Default evaluation target: a tool-using agent around ``llm``."""
    return Agent(llm, tools=demo_tools(), config=AgentConfig(system_prompt=SYSTEM_PROMPT))


def _last(messages: list[Message], role: str) -> Message | None:
    return next((m for m in reversed(messages) if m.role == role), None)


def _demo_rule(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    question = (_last(messages, "user") or Message.user("")).content or ""
    cid = f"call_{len(messages)}"
    results = [m for m in messages if m.role == "tool"]
    if results:
        outcome = results[-1].content or ""
        if outcome.startswith("Error"):
            return Message.assistant("I could not compute that.")
        if match := _CAPITAL.search(question):
            return Message.assistant(f"The capital of {match.group(1)} is {outcome}.")
        return Message.assistant(f"The answer is {outcome}.")
    if match := _ADD.search(question):
        args = {"a": int(match.group(1)), "b": int(match.group(2))}
        return Message.assistant(None, (ToolCall(cid, "add", args),))
    if match := _MUL.search(question):
        args = {"a": int(match.group(1)), "b": int(match.group(2))}
        return Message.assistant(None, (ToolCall(cid, "multiply", args),))
    if match := _CAPITAL.search(question):
        return Message.assistant(
            None, (ToolCall(cid, "lookup_capital", {"country": match.group(1)}),)
        )
    lowered = question.lower()
    if "ignore previous" in lowered or "secret" in lowered:
        return Message.assistant("I can't help with that.")
    if lowered.startswith(("hello", "hi ")):
        return Message.assistant("Hello! How can I help?")
    if "refund" in lowered:
        return Message.assistant("Refunds are reviewed within three business days.")
    return Message.assistant("I don't know.")


def make_mock_llm(meter: CostMeter | None = None) -> RuleLLM:
    return RuleLLM(_demo_rule, model="mock", usage_per_call=Usage(120, 30), meter=meter)


def _judge_rule(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    body = (_last(messages, "user") or Message.user("")).content or ""
    answer = body.split("ANSWER:\n", 1)[-1]
    good = "three business days" in answer
    verdict = {"score": 0.9 if good else 0.1, "reason": "mock judge: keyword check"}
    return Message.assistant(json.dumps(verdict))


def make_mock_judge(meter: CostMeter | None = None) -> RuleLLM:
    return RuleLLM(_judge_rule, model="mock-judge", usage_per_call=Usage(300, 25), meter=meter)
