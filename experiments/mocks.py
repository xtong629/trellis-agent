"""Deterministic stand-ins for models, used only to prove the pipeline works before real runs.

They understand just the experiment tasks (by pattern) and report token usage proportional to the
text they see, so cost comparisons between window and memory are meaningful in shape (not in
size). Mock results say nothing about real model quality."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from experiments.c1_tasks import CATALOG
from trellis_agent.cost import CostMeter
from trellis_agent.llm import RuleLLM
from trellis_agent.types import Message, ToolCall, ToolSchema, Usage

Fn = Callable[[list[Message], list[ToolSchema] | None], Message]


class SizedMock(RuleLLM):
    """A RuleLLM whose reported usage follows the text it was given (chars / 4, at least 1)."""

    async def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        sent = sum(len(m.content or "") + sum(len(str(c.arguments)) for c in m.tool_calls)
                   for m in messages)  # fmt: skip
        self.usage_per_call = Usage(prompt_tokens=max(1, sent // 4), completion_tokens=8)
        return await super().chat(messages, **kwargs)


def mock(fn: Fn, meter: CostMeter | None) -> SizedMock:
    return SizedMock(fn, model="mock", meter=meter)


# ------------------------------------------------------------------ experiment A: understanding
Step = tuple[str, dict[str, Any]]


def understand(task: str) -> tuple[list[Step], str, dict[str, Any]]:
    """(lookups, combine-op, params) for a task text. ops: first, mul, sum, mul_add."""
    if m := re.search(r"(?:A|Convert the price of one) (\w+?)(?: is listed| from).*?(\w+)"
                      r"(?: to|\. Look.*costs in) (\w+)", task):  # fmt: skip
        item, a, b = m.group(1), m.group(2), m.group(3)
        return [("get_price", {"item": item}), ("get_exchange_rate",
                {"from_currency": a, "to_currency": b})], "mul", {}  # fmt: skip
    if m := re.search(r"price of (\d+) (\w+?)s and add 5", task):
        return [("get_price", {"item": m.group(2)})], "mul_add", {"qty": int(m.group(1))}
    if m := re.search(r"How many (\w+?)s are in stock.*warehouses (.+?)\? ", task):
        steps = [
            ("get_stock", {"warehouse": w, "item": m.group(1)}) for w in m.group(2).split(", ")
        ]
        return steps, "sum", {}  # fmt: skip
    if m := re.search(r"population, in millions, of (.+?)\? ", task):
        return [("get_population", {"city": c}) for c in m.group(1).split(", ")], "sum", {}
    if m := re.search(r"unit price of a (\w+)\?", task):
        return [("get_price", {"item": m.group(1)})], "first", {}
    raise ValueError(f"the mock does not understand: {task[:60]}")


def combine(op: str, params: dict[str, Any], values: list[float]) -> str | None:
    """The calculator expression for an op, or None when the lookup value is the answer."""
    if op == "mul":
        return f"{values[0]} * {values[1]}"
    if op == "mul_add":
        return f"{values[0]} * {params['qty']} + 5"
    if op == "sum":
        return " + ".join(str(v) for v in values)
    return None


def _tool_values(messages: list[Message]) -> tuple[list[Message], list[float], bool]:
    tools = [m for m in messages if m.role == "tool"]
    busy = bool(tools) and "busy" in (tools[-1].content or "")
    values = [float(m.content or "0") for m in tools if "busy" not in (m.content or "")
              and not (m.content or "").startswith(("ValueError", "Error"))]  # fmt: skip
    return tools, values, busy


def _first_user(messages: list[Message]) -> str:
    return next(m.content or "" for m in messages if m.role == "user")


def _answer(value: float) -> Message:
    return Message.assistant(f"Worked it out.\nANSWER: {round(value, 4):g}")


def solver(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """One agent doing everything: lookups one by one, a retry on 'busy', then the arithmetic."""
    steps, op, params = understand(_first_user(messages))
    called, values, busy = _tool_values(messages)
    if busy:
        name, args = steps[len(values)]
        return Message.assistant(None, (ToolCall(f"r{len(called)}", name, args),))
    if len(values) < len(steps):
        name, args = steps[len(values)]
        return Message.assistant(None, (ToolCall(f"c{len(called)}", name, args),))
    expression = combine(op, params, values)
    if expression is None:
        return _answer(values[-1])
    if len(values) == len(steps):
        return Message.assistant(None, (ToolCall("calc", "calculate", {"expression": expression}),))
    return _answer(values[-1])


def planner(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    steps, op, params = understand(
        _first_user(messages).split("GOAL:\n", 1)[-1].split("\n\nWORKERS")[0]
    )
    tasks: list[dict[str, Any]] = [
        {"id": f"t{i}", "kind": "research", "goal": f"Call {n} with {json.dumps(a)}."}
        for i, (n, a) in enumerate(steps, start=1)
    ]
    if combine(op, params, [1.0] * len(steps)) is not None:
        ids = [t["id"] for t in tasks]
        goal = f"Compute {op} over {','.join(ids)} with {json.dumps(params)}."
        tasks.append(
            {"id": f"t{len(tasks) + 1}", "kind": "compute", "goal": goal, "depends_on": ids}
        )
    return Message.assistant(json.dumps({"tasks": tasks}))  # fmt: skip


def researcher(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    _, values, busy = _tool_values(messages)
    called = [m for m in messages if m.role == "tool"]
    goal = _first_user(messages).split("Your task", 1)[-1]
    name, raw = re.search(r"Call (\w+) with (\{.*?\})\.", goal).groups()  # type: ignore[union-attr]
    if busy or not values:
        return Message.assistant(None, (ToolCall(f"c{len(called)}", name, json.loads(raw)),))
    return Message.assistant(f"value: {values[-1]:g}")


def calculator(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    _, values, _ = _tool_values(messages)
    if values:
        return Message.assistant(f"value: {values[-1]:g}")
    text = _first_user(messages)
    goal = text.split("Your task", 1)[-1].split("Results from", 1)[0]
    op, ids, raw = re.search(r"Compute (\w+) over ([\w,]+) with (\{.*?\})\.", goal).groups()  # type: ignore[union-attr]
    found = [float(re.search(rf"\[{i}\] value: ([\d.]+)", text).group(1)) for i in ids.split(",")]  # type: ignore[union-attr]
    expression = combine(op, json.loads(raw), found)
    assert expression is not None
    return Message.assistant(None, (ToolCall("calc", "calculate", {"expression": expression}),))


def synthesizer(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    results = _first_user(messages).split("TASK RESULTS:", 1)[-1]
    return _answer(float(re.findall(r"value: ([\d.]+)", results)[-1]))


# ------------------------------------------------------------------ experiment B: conversations
_FACTS = [
    ("allergy", re.compile(r"allergic to ([\w ]+?)\.")),
    ("deadline", re.compile(r"deadline (?:is|moved to) ([A-Z][a-z]+ \d+)")),
    ("refund", re.compile(r"退款单号是 (RF-\d+)")),
]
_ASKS = {"allergy": "allergic to?", "deadline": "report deadline", "refund": "退款单号是什么"}


def _facts_in(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        for key, pattern in _FACTS:
            if m := pattern.search(line):
                found[key] = m.group(1)  # later lines win, so a correction overrides
    return found


def conversation_model(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """Chats; extracts facts when asked; answers a question from what is visible in its prompt."""
    system = " ".join(m.content or "" for m in messages if m.role == "system")
    if system.startswith("Extract durable facts"):
        convo = next(m.content or "" for m in messages if m.role == "user")
        facts = [{"text": line.removeprefix("user: "), "key": key, "importance": 0.8}
                 for line in convo.splitlines() for key, pattern in _FACTS
                 if pattern.search(line)]  # fmt: skip
        return Message.assistant(json.dumps({"facts": facts}))
    question = next(m.content or "" for m in reversed(messages) if m.role == "user")
    if m := re.search(r"What is (\d+) plus (\d+)\?", question):
        return Message.assistant(f"ANSWER: {int(m.group(1)) + int(m.group(2))}")
    visible = "\n".join(m.content or "" for m in messages if m.role in ("system", "user"))
    visible = visible.replace(question, "")
    known = _facts_in(visible)
    for key, marker in _ASKS.items():
        if marker in question:
            return Message.assistant(known.get(key, "I don't know"))
    return Message.assistant("Sure.")


# ------------------------------------------------------------------ v0.2 C1: wiring only
def c1_solver(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """Looks up the first catalog item named in the task and answers with its listed price.
    Right on the ``easy`` family only; it exists to exercise the C1 wiring, not to solve tasks."""
    done = [m for m in messages if m.role == "tool"]
    if not done:
        text = _first_user(messages)
        item = next((i for i in CATALOG if re.search(rf"\b{i}\b", text)), "bolt")
        return Message.assistant(None, (ToolCall("c0", "get_item", {"item": item}),))
    price = json.loads(done[-1].content or "{}").get("price", 0)
    return _answer(float(price))


def c1_checker(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """Wiring-only checker: rejects an answer of 0, accepts anything else."""
    answer = _first_user(messages).rsplit("FINAL ANSWER:", 1)[-1]
    ok = re.search(r"ANSWER:\s*0(?:\.0)?\s*$", answer) is None
    return Message.assistant(json.dumps({"ok": ok} if ok else {"ok": False, "reason": "zero?"}))


# ------------------------------------------------------------------ v0.2 C2: wiring only
_C2_TASK = re.compile(r"units of (\w+) are held in total at the depots (.+?)\? ")


def _c2_steps(text: str) -> list[Step]:
    m = _C2_TASK.search(text)
    if m is None:
        raise ValueError(f"the mock does not understand: {text[:60]}")
    return [("get_units", {"depot": d, "product": m.group(1)}) for d in m.group(2).split(", ")]


def c2_solver(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    """Looks each depot up in turn (calling again after 'busy'), then answers with the sum."""
    steps = _c2_steps(_first_user(messages))
    called, values, busy = _tool_values(messages)
    if busy or len(values) < len(steps):
        name, args = steps[len(values)]
        return Message.assistant(None, (ToolCall(f"c{len(called)}", name, args),))
    return _answer(sum(values))


def c2_planner(messages: list[Message], tools: list[ToolSchema] | None) -> Message:
    goal = _first_user(messages).split("GOAL:\n", 1)[-1].split("\n\nWORKERS")[0]
    steps = _c2_steps(goal)
    tasks: list[dict[str, Any]] = [
        {"id": f"t{i}", "kind": "research", "goal": f"Call {n} with {json.dumps(a)}."}
        for i, (n, a) in enumerate(steps, start=1)
    ]
    ids = [t["id"] for t in tasks]
    goal_sum = f"Compute sum over {','.join(ids)} with {{}}."
    tasks.append(
        {"id": f"t{len(tasks) + 1}", "kind": "compute", "goal": goal_sum, "depends_on": ids}
    )
    return Message.assistant(json.dumps({"tasks": tasks}))
