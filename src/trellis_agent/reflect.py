"""Reflection: check a finished answer, and on rejection let the agent try again with the reason.

``ReflectingAgent`` wraps an agent that keeps per-session history (a plain ``Agent``). After a run
that ended with a final answer, a ``Reflector`` reviews it. If the verdict is "not ok", the reason
goes back to the agent as a new user turn in the same session, and the agent answers again (it may
call tools again). With ``max_retries=1`` (the default) there is one check and at most one retry;
the retry's answer is final and is not checked again.

Accounting follows the rest of the library: the result carries the usage and cost of every attempt
and every check (checks are tagged ``reflect`` in the meter); a check that crosses the budget stops
the run with ``budget_exceeded`` and is still counted; a failing checker never fails the run (it is
traced and the current answer is returned). Not every agent can be wrapped: ``PlannedAgent`` keeps
no session history, so a retry would not see its first attempt.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .cost import BudgetExceeded
from .jsonutil import extract_json
from .llm import LLMClient
from .types import Message, RunResult, TraceEvent, Usage

CHECK_PROMPT = (
    "You review an assistant's answer to a task. You see the task, the tool calls the assistant "
    "made with their results, and its final answer. Check every step against the tool results, "
    "check that every condition stated in the task was applied, check the arithmetic, and check "
    "that the answer has the format the task asked for. Do not assume facts that are not in the "
    'tool results. Reply with ONLY a JSON object: {"ok": true} if the answer is right, or '
    '{"ok": false, "reason": "<what is wrong, in one or two sentences>"}.'
)
RETRY_PROMPT = (
    "A reviewer checked your answer and found a problem: {reason}\n"
    "Check your work again, using the tools if needed, and give a corrected final answer in the "
    "format the task asked for."
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""
    parsed: bool = True  # False: the checker's reply was unreadable and counted as ok
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None
    calls: int = 0  # model calls the check made


class Reflector(Protocol):
    async def check(self, query: str, result: RunResult) -> Verdict: ...


@dataclass(frozen=True)
class ReflectionRecord:
    """One run's reflection outcome, for counting fixes and harms afterwards."""

    session_id: str
    query: str
    first_answer: str
    first_stop_reason: str
    checked: bool
    verdict_ok: bool | None
    reason: str
    parsed: bool | None
    retried: bool
    final_answer: str
    final_stop_reason: str


def transcript(messages: list[Message], *, max_chars: int = 6000) -> str:
    """The tool calls and their results of a run, one line each (system messages left out)."""
    lines: list[str] = []
    for m in messages:
        if m.role == "assistant":
            lines += [f"CALL {c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
                      for c in m.tool_calls]  # fmt: skip
        elif m.role == "tool":
            lines.append(f"  -> {m.content}")
    text = "\n".join(lines) or "(no tool calls)"
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


class LLMReflector:
    """A model as the checker: temperature 0, a fixed generic prompt, strict JSON parsing.

    It sees only the task, the run's tool steps and the final answer; never expected answers or
    scoring rules. An unreadable reply counts as ``ok`` (no retry) with ``parsed=False``."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        prompt: str = CHECK_PROMPT,
        tag: str = "reflect",
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self.llm, self.prompt, self.tag = llm, prompt, tag
        self.max_tokens, self.extra_params = max_tokens, extra_params

    async def check(self, query: str, result: RunResult) -> Verdict:
        body = (
            f"TASK:\n{query}\n\nTOOL CALLS AND RESULTS:\n{transcript(result.messages)}"
            f"\n\nFINAL ANSWER:\n{result.content or '(empty)'}"
        )
        response = await self.llm.chat(
            [Message.system(self.prompt), Message.user(body)],
            temperature=0.0,
            max_tokens=self.max_tokens,
            tag=self.tag,
            extra_params=self.extra_params,
        )
        found = extract_json(response.message.content or "")
        ok = None if found is None else found.get("ok")
        if not isinstance(ok, bool):
            return Verdict(True, "", False, response.usage, response.cost_usd, 1)
        reason = "" if ok else str(found.get("reason", "")).strip()  # type: ignore[union-attr]
        return Verdict(ok, reason, True, response.usage, response.cost_usd, 1)


class _Runnable(Protocol):
    async def arun(self, query: str, *, session_id: str = "default") -> RunResult: ...


def _event(kind: str, **data: object) -> TraceEvent:
    return TraceEvent(kind, time.time(), dict(data), "root", None)


class ReflectingAgent:
    def __init__(
        self,
        inner: _Runnable,
        reflector: Reflector,
        *,
        max_retries: int = 1,
        retry_prompt: str = RETRY_PROMPT,
        on_record: Callable[[ReflectionRecord], None] | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.inner, self.reflector = inner, reflector
        self.max_retries, self.retry_prompt, self.on_record = max_retries, retry_prompt, on_record

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        first = current = await self.inner.arun(query, session_id=session_id)
        usage, costs, trace = current.usage, [current.cost_usd], list(current.trace)
        stop, error = current.stop_reason, current.error
        verdict: Verdict | None = None
        retried = False
        for attempt in range(1, self.max_retries + 1):
            if current.stop_reason != "end_turn":
                break
            try:
                verdict = await self.reflector.check(query, current)
            except BudgetExceeded as exc:
                paid = exc.response
                if paid is not None:
                    usage, costs = usage + paid.usage, [*costs, paid.cost_usd]
                    event = _event("model_request", phase="reflect", over_budget=True,
                                   usage=paid.usage.to_dict(), cost_usd=paid.cost_usd)  # fmt: skip
                    trace.append(event)
                trace.append(_event("error", category="budget_exceeded", message=str(exc)))
                stop, error = "budget_exceeded", str(exc)
                break
            except Exception as exc:  # a broken checker must never fail the run
                message = f"{type(exc).__name__}: {exc}"
                trace.append(_event("error", category="reflect_error", message=message))
                break
            usage, costs = usage + verdict.usage, [*costs, verdict.cost_usd]
            trace += [_event("model_request", phase="reflect", usage=verdict.usage.to_dict(),
                             cost_usd=verdict.cost_usd) for _ in range(verdict.calls)]  # fmt: skip
            trace.append(_event("reflection", attempt=attempt, ok=verdict.ok,
                                reason=verdict.reason, parsed=verdict.parsed))  # fmt: skip
            if verdict.ok:
                break
            retry = self.retry_prompt.format(reason=verdict.reason or "(no reason given)")
            current = await self.inner.arun(retry, session_id=session_id)
            retried = True
            usage, costs = usage + current.usage, [*costs, current.cost_usd]
            trace += current.trace
            stop, error = current.stop_reason, current.error
        if self.on_record is not None:
            self.on_record(ReflectionRecord(
                session_id, query, first.content, first.stop_reason, verdict is not None,
                None if verdict is None else verdict.ok, "" if verdict is None else verdict.reason,
                None if verdict is None else verdict.parsed, retried, current.content, stop,
            ))  # fmt: skip
        known = [c for c in costs if c is not None]
        cost = sum(known) if len(known) == len(costs) else None
        return RunResult(current.content, current.messages, usage, cost, trace, error, stop)
