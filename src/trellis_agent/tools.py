"""Tools: JSON Schema from type hints, argument validation, and error-to-model reporting.

A tool failure must never crash an agent run. Bad arguments, unknown tools and exceptions are all
turned into an error ``ToolResult`` whose text goes back to the model so it can correct itself.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import types
import typing
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin, overload

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from .types import ToolCall, ToolSchema

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCALARS: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema_for(tp: Any, where: str) -> dict[str, Any]:
    origin = get_origin(tp)
    if origin is typing.Annotated:
        base, *meta = get_args(tp)
        schema = _schema_for(base, where)
        notes = [m for m in meta if isinstance(m, str)]
        if notes:
            schema = {**schema, "description": notes[0]}
        return schema
    if tp is Any:
        return {}
    if tp is type(None):
        return {"type": "null"}
    if tp in _SCALARS:
        return {"type": _SCALARS[tp]}
    if origin is Literal:
        return {"enum": list(get_args(tp))}
    if origin in (typing.Union, types.UnionType):
        return {"anyOf": [_schema_for(arg, where) for arg in get_args(tp)]}
    if tp is list or origin is list:
        args = get_args(tp)
        return (
            {"type": "array", "items": _schema_for(args[0], where)} if args else {"type": "array"}
        )
    if tp is dict or origin is dict:
        args = get_args(tp)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _schema_for(args[1], where)}
        return {"type": "object"}
    raise TypeError(f"unsupported annotation {tp!r} for {where}")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> ToolSchema:
        return ToolSchema(self.name, self.description, self.parameters)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


def _build_tool(fn: Callable[..., Any], name: str | None, description: str | None) -> Tool:
    tool_name = name or fn.__name__
    if not _NAME_RE.match(tool_name):
        raise ValueError(f"invalid tool name {tool_name!r}: use 1-64 of letters, digits, _ or -")
    hints = typing.get_type_hints(fn, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {tool_name!r}: *args/**kwargs are not supported")
        if param.name not in hints:
            raise TypeError(f"tool {tool_name!r}: parameter {param.name!r} needs a type annotation")
        properties[param.name] = _schema_for(hints[param.name], f"{tool_name}.{param.name}")
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    doc = inspect.getdoc(fn) or ""
    text = description if description is not None else (doc.splitlines()[0] if doc else "")
    return Tool(tool_name, text, parameters, fn)


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(
    *, name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Tool]: ...
def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed function into a ``Tool``. Usable as ``@tool`` or ``@tool(name=...)``."""
    if fn is not None:
        return _build_tool(fn, name, description)
    return lambda f: _build_tool(f, name, description)


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False
    error_type: str | None = None
    latency_ms: float = 0.0


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), *, max_result_chars: int = 20_000) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self.max_result_chars = max_result_chars
        for item in tools:
            self.register(item)

    def register(self, tool_: Tool) -> None:
        if tool_.name in self._tools:
            raise ValueError(f"tool {tool_.name!r} is already registered")
        self._tools[tool_.name] = tool_
        self._validators[tool_.name] = Draft202012Validator(tool_.parameters)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[ToolSchema]:
        return [t.schema() for t in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    async def call(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        def error(kind: str, text: str) -> ToolResult:
            return ToolResult(f"Error: {text}", True, kind, elapsed())

        target = self._tools.get(call.name)
        if target is None:
            available = ", ".join(self._tools) or "(none)"
            return error(
                "unknown_tool", f"unknown tool {call.name!r}. Available tools: {available}"
            )
        if call.arguments_error is not None:
            return error(
                "invalid_arguments",
                f"arguments for tool {call.name!r} are not valid JSON "
                f"({call.arguments_error}); received: {call.raw_arguments!r}",
            )
        problem = best_match(self._validators[call.name].iter_errors(call.arguments))
        if problem is not None:
            where = "".join(f"/{p}" for p in problem.absolute_path)
            suffix = f" (at {where})" if where else ""
            return error(
                "invalid_arguments",
                f"invalid arguments for tool {call.name!r}: {problem.message}{suffix}",
            )
        try:
            if inspect.iscoroutinefunction(target.fn):
                outcome = await target.fn(**call.arguments)
            else:
                outcome = await asyncio.to_thread(target.fn, **call.arguments)
                if inspect.isawaitable(outcome):
                    outcome = await _await(outcome)
        except Exception as exc:
            return error(
                type(exc).__name__, f"tool {call.name!r} raised {type(exc).__name__}: {exc}"
            )
        text = (
            outcome
            if isinstance(outcome, str)
            else json.dumps(outcome, ensure_ascii=False, default=str)
        )
        if len(text) > self.max_result_chars:
            dropped = len(text) - self.max_result_chars
            text = text[: self.max_result_chars] + f"\n...[truncated {dropped} chars]"
        return ToolResult(text, False, None, elapsed())


async def _await(value: Awaitable[Any]) -> Any:
    return await value
