"""Client for any OpenAI-compatible chat-completions endpoint (OpenAI, DeepSeek, Qwen, vLLM, ...).

Real providers are sloppy in small ways, and each is handled here rather than crashing the agent:
missing ``usage`` (estimated and flagged), tool-call arguments that are not valid JSON objects
(surfaced via ``ToolCall.arguments_error`` so the model can retry), missing tool-call ids
(synthesised), and transport/HTTP failures (wrapped as ``LLMError``).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from ..cost import BudgetExceeded, CostMeter
from ..types import Message, ToolCall, ToolSchema, Usage
from .base import LLMError, LLMResponse

_RESERVED = {"model", "messages", "tools", "temperature", "max_tokens", "stream", "extra_body"}


def check_extra_params(extra: Mapping[str, Any] | None) -> dict[str, Any]:
    clash = sorted(set(extra or {}) & _RESERVED)
    if clash:
        raise ValueError(f"extra_params cannot override fields the client controls: {clash}")
    return dict(extra or {})


def _estimate_tokens(text: str) -> int:
    """Rough chars/4 heuristic; only used when the provider reports no usage."""
    return max(1, len(text) // 4)


def _to_wire(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    # Unparseable arguments are replayed as "{}": strict providers reject invalid
                    # JSON in history, and the error text in the tool message already carries the
                    # original string for the model.
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return wire


def _parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    if raw is None or not raw.strip():
        return {}, None  # some providers send "" for a tool that takes no arguments
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    if not isinstance(value, dict):
        return {}, f"arguments must be a JSON object, got {type(value).__name__}"
    return value, None


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        meter: CostMeter | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.extra_params = check_extra_params(extra_params)  # e.g. {"reasoning_effort": "low"}
        self.meter = meter
        self.base_url = base_url
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
        )

    @classmethod
    def from_env(
        cls, meter: CostMeter | None = None, *, model: str | None = None, **kwargs: Any
    ) -> OpenAICompatClient:
        """Build from ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` (see ``.env.example``).

        ``model`` overrides ``LLM_MODEL`` (e.g. to use a different model as a judge).
        """
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
        if missing:
            raise LLMError(f"missing environment variables: {', '.join(missing)}")
        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=model or os.environ["LLM_MODEL"],
            meter=meter,
            **kwargs,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        tag: str = "",
        extra_params: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        wire = [_to_wire(m) for m in messages]
        params: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        extra = {**self.extra_params, **check_extra_params(extra_params)}
        if extra:
            params["extra_body"] = extra  # the SDK merges these into the top level of the body
        if tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        started = time.perf_counter()
        try:
            completion = await self._client.chat.completions.create(**params)
        except openai.APITimeoutError as exc:
            raise LLMError(f"request timed out: {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"could not connect to {self.base_url}: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"HTTP {exc.status_code}: {exc.message}") from exc
        except (openai.OpenAIError, ValueError) as exc:
            raise LLMError(f"unusable response from provider: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        if not completion.choices:
            raise LLMError("provider response contained no choices")
        choice = completion.choices[0]
        raw_message = choice.message

        calls: list[ToolCall] = []
        for index, item in enumerate(raw_message.tool_calls or []):
            function = getattr(item, "function", None)
            if function is None:
                continue  # non-function tool call types are not supported
            arguments, problem = _parse_arguments(function.arguments)
            calls.append(
                ToolCall(
                    id=item.id or f"call_{index}",
                    name=function.name,
                    arguments=arguments,
                    raw_arguments=function.arguments if problem else None,
                    arguments_error=problem,
                )
            )
        message = Message.assistant(raw_message.content, tuple(calls))

        usage = self._usage(completion.usage, wire, message)
        finish = choice.finish_reason or "stop"
        cost: float | None = None
        if self.meter is not None:
            try:
                cost = self.meter.record(self.model, usage, tag=tag)
            except BudgetExceeded as exc:
                # The call is already paid for: hand the response to the caller for accounting.
                exc.response = LLMResponse(
                    message, usage, latency_ms, finish, self.model, exc.call_cost
                )
                raise
        return LLMResponse(message, usage, latency_ms, finish, self.model, cost)

    @staticmethod
    def _usage(raw: Any, wire: list[dict[str, Any]], reply: Message) -> Usage:
        if raw is not None and raw.prompt_tokens is not None and raw.completion_tokens is not None:
            details = getattr(raw, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0
            reasoning = getattr(
                getattr(raw, "completion_tokens_details", None), "reasoning_tokens", None
            )
            return Usage(
                raw.prompt_tokens,
                raw.completion_tokens,
                min(cached, raw.prompt_tokens),
                reasoning_tokens=None
                if reasoning is None
                else min(reasoning, raw.completion_tokens),
            )
        completion_text = (reply.content or "") + "".join(
            c.name + json.dumps(c.arguments) for c in reply.tool_calls
        )
        return Usage(
            _estimate_tokens(json.dumps(wire, ensure_ascii=False)),
            _estimate_tokens(completion_text),
            estimated=True,
        )
