"""LLM and embedding protocols. Everything above this layer talks only to these."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..types import Message, ToolSchema, Usage


class LLMError(RuntimeError):
    """The provider returned something unusable (no choices, transport failure, ...)."""


@dataclass(frozen=True)
class LLMResponse:
    message: Message
    usage: Usage
    latency_ms: float
    finish_reason: str
    model: str
    cost_usd: float | None = None


class LLMClient(Protocol):
    """A chat model. Clients record their own spend in the ``CostMeter`` they were built with."""

    model: str

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tag: str = "",
        extra_params: Mapping[str, Any] | None = None,
    ) -> LLMResponse: ...


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
