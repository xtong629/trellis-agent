"""Deterministic stand-ins for a real model. They cost nothing and make tests/CI reproducible.

A mock's behaviour says nothing about how a real model would perform; it only exercises our code.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..cost import BudgetExceeded, CostMeter
from ..types import Message, ToolSchema, Usage
from .base import LLMResponse

DEFAULT_MOCK_USAGE = Usage(prompt_tokens=10, completion_tokens=5)


class ScriptExhausted(RuntimeError):
    """The scripted model was asked for more replies than the script contains."""


@dataclass(frozen=True)
class CallRecord:
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...] | None
    temperature: float
    tag: str


class _MockLLM:
    def __init__(self, *, model: str, usage_per_call: Usage, meter: CostMeter | None) -> None:
        self.model = model
        self.usage_per_call = usage_per_call
        self.meter = meter
        self.calls: list[CallRecord] = []

    def _respond(self, message: Message, tag: str) -> LLMResponse:
        finish = "tool_calls" if message.tool_calls else "stop"
        cost: float | None = None
        if self.meter is not None:
            try:
                cost = self.meter.record(self.model, self.usage_per_call, tag=tag)
            except BudgetExceeded as exc:
                exc.response = LLMResponse(
                    message, self.usage_per_call, 0.0, finish, self.model, exc.call_cost
                )
                raise
        return LLMResponse(message, self.usage_per_call, 0.0, finish, self.model, cost)

    def _record(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        temperature: float,
        tag: str,
    ) -> None:
        self.calls.append(
            CallRecord(tuple(messages), None if tools is None else tuple(tools), temperature, tag)
        )


class ScriptedLLM(_MockLLM):
    """Replays a fixed list of assistant messages (or callables producing them), in order."""

    def __init__(
        self,
        script: Sequence[Message | Callable[[list[Message]], Message]],
        *,
        model: str = "mock",
        usage_per_call: Usage = DEFAULT_MOCK_USAGE,
        meter: CostMeter | None = None,
    ) -> None:
        super().__init__(model=model, usage_per_call=usage_per_call, meter=meter)
        self._script = list(script)
        self._next = 0

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tag: str = "",
    ) -> LLMResponse:
        self._record(messages, tools, temperature, tag)
        if self._next >= len(self._script):
            raise ScriptExhausted(f"script has only {len(self._script)} replies; asked for more")
        item = self._script[self._next]
        self._next += 1
        message = item if isinstance(item, Message) else item(list(messages))
        return self._respond(message, tag)


class RuleLLM(_MockLLM):
    """Decides each reply with a function of (messages, tools): a tiny rule-based fake model."""

    def __init__(
        self,
        fn: Callable[[list[Message], list[ToolSchema] | None], Message],
        *,
        model: str = "mock",
        usage_per_call: Usage = DEFAULT_MOCK_USAGE,
        meter: CostMeter | None = None,
    ) -> None:
        super().__init__(model=model, usage_per_call=usage_per_call, meter=meter)
        self._fn = fn

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tag: str = "",
    ) -> LLMResponse:
        self._record(messages, tools, temperature, tag)
        message = self._fn(list(messages), None if tools is None else list(tools))
        return self._respond(message, tag)


class HashEmbedder:
    """Bag-of-hashed-words vectors: deterministic, offline, and similar for texts sharing words."""

    def __init__(self, dim: int = 64) -> None:
        if dim < 1:
            raise ValueError("dim must be positive")
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vec[value % self.dim] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return vec if norm == 0 else [x / norm for x in vec]
