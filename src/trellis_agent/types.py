"""Plain data types shared by every module. No behaviour beyond validation and small helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Usage:
    """Token usage of one or more model calls.

    ``prompt_tokens`` is the *total* prompt size; ``cached_tokens`` is the part of it served from a
    provider cache (a subset, never additional). ``estimated`` is True when the provider did not
    report usage and we guessed from text length -- such numbers must never be presented as exact.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    estimated: bool = False

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0 or self.cached_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens cannot exceed prompt_tokens")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
            self.estimated or other.estimated,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Usage:
        return cls(
            int(raw.get("prompt_tokens", 0)),
            int(raw.get("completion_tokens", 0)),
            int(raw.get("cached_tokens", 0)),
            bool(raw.get("estimated", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "estimated": self.estimated,
        }


@dataclass(frozen=True)
class ToolCall:
    """A model's request to call a tool.

    If the model produced arguments that are not a valid JSON object, ``arguments`` is empty and
    ``arguments_error`` explains why, so the error can be reported back to the model instead of
    crashing the run.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str | None = None
    arguments_error: str | None = None


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("a tool message requires tool_call_id")

    @classmethod
    def system(cls, content: str) -> Message:
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str | None, tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls("assistant", content, tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls("tool", content, tool_call_id=tool_call_id)


@dataclass(frozen=True)
class ToolSchema:
    """What the model is told about a tool (JSON Schema for the parameters)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TraceEvent:
    type: str
    ts: float
    data: dict[str, Any]
    span_id: str
    parent_id: str | None


@dataclass
class RunResult:
    content: str
    messages: list[Message]
    usage: Usage
    cost_usd: float | None
    trace: list[TraceEvent] = field(default_factory=list)
    error: str | None = None
    stop_reason: str = "end_turn"
