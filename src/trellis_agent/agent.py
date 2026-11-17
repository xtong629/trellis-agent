"""The agent loop. ``stream`` is the only implementation; ``arun``/``run`` just collect it.

Loop: send messages (+ tool schemas) to the model; if it asks for tools, run them and feed the
results back; repeat until it answers without tool calls, or a stop condition fires
(``max_steps``, budget exhausted, provider failure). Failures are reported in ``RunResult``;
they are not raised, so one bad run never takes down an evaluation or a scheduler.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .cost import BudgetExceeded
from .llm import LLMClient
from .tools import ToolRegistry
from .trace import Tracer
from .types import Message, RunResult, TraceEvent, Usage


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str = ""
    max_steps: int = 8
    temperature: float = 0.0
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


@dataclass(frozen=True)
class AgentEvent:
    """One step of progress. ``done`` is always last and carries the final ``RunResult``."""

    type: Literal["tool_call", "tool_result", "text", "done"]
    data: dict[str, Any] = field(default_factory=dict)
    result: RunResult | None = None


class Agent:
    """A tool-using agent with per-session in-memory history.

    Not safe to run two turns of the *same* session concurrently; different sessions are fine.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        trace_sink: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.config = config or AgentConfig()
        self._trace_sink = trace_sink
        self._sessions: dict[str, list[Message]] = {}

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        result: RunResult | None = None
        async for event in self.stream(query, session_id=session_id):
            if event.type == "done":
                result = event.result
        assert result is not None  # stream() always ends with "done"
        return result

    def run(self, query: str, *, session_id: str = "default") -> RunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(query, session_id=session_id))
        raise RuntimeError("run() cannot be called from a running event loop; use arun()")

    async def stream(self, query: str, *, session_id: str = "default") -> AsyncIterator[AgentEvent]:
        cfg = self.config
        tracer = Tracer(sink=self._trace_sink)  # one tracer per run keeps concurrent runs apart
        started = time.perf_counter()
        messages = [*self._sessions.get(session_id, []), Message.user(query)]
        system = [Message.system(cfg.system_prompt)] if cfg.system_prompt else []
        schemas = self.tools.schemas() or None
        usage = Usage()
        costs: list[float] = []
        stop_reason = "max_steps"
        error: str | None = f"max_steps ({cfg.max_steps}) reached without a final answer"
        steps = 0

        tracer.emit("run_start", session_id=session_id, query=query)
        for steps in range(1, cfg.max_steps + 1):
            try:
                response = await self.llm.chat(
                    [*system, *messages],
                    tools=schemas,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
            except BudgetExceeded as exc:
                stop_reason, error = "budget_exceeded", str(exc)
                paid = exc.response
                if paid is not None:  # paid for, so it counts, even though the run stops here
                    usage = usage + paid.usage
                    if paid.cost_usd is not None:
                        costs.append(paid.cost_usd)
                    tracer.emit(
                        "model_request",
                        step=steps,
                        model=paid.model,
                        latency_ms=paid.latency_ms,
                        usage=paid.usage.to_dict(),
                        cost_usd=paid.cost_usd,
                        finish_reason=paid.finish_reason,
                        over_budget=True,
                    )
                tracer.emit("error", category="budget_exceeded", message=error, step=steps)
                break
            except Exception as exc:
                stop_reason, error = "llm_error", f"{type(exc).__name__}: {exc}"
                tracer.emit("error", category="llm_error", message=error, step=steps)
                break

            usage = usage + response.usage
            if response.cost_usd is not None:
                costs.append(response.cost_usd)
            tracer.emit(
                "model_request",
                step=steps,
                model=response.model,
                latency_ms=response.latency_ms,
                usage=response.usage.to_dict(),
                cost_usd=response.cost_usd,
                finish_reason=response.finish_reason,
            )
            messages.append(response.message)

            if not response.message.tool_calls:
                stop_reason, error = "end_turn", None
                yield AgentEvent("text", {"content": response.message.content or ""})
                break

            for call in response.message.tool_calls:
                tracer.emit("tool_call", id=call.id, name=call.name, arguments=call.arguments)
                yield AgentEvent("tool_call", {"id": call.id, "name": call.name})
                outcome = await self.tools.call(call)
                tracer.emit(
                    "tool_result",
                    id=call.id,
                    name=call.name,
                    latency_ms=outcome.latency_ms,
                    error=outcome.is_error,
                    error_type=outcome.error_type,
                )
                messages.append(Message.tool(call.id, outcome.content))
                yield AgentEvent(
                    "tool_result",
                    {
                        "id": call.id,
                        "name": call.name,
                        "content": outcome.content,
                        "error": outcome.is_error,
                    },
                )

        duration_ms = (time.perf_counter() - started) * 1000
        tracer.emit("run_end", stop_reason=stop_reason, steps=steps, duration_ms=duration_ms)
        self._sessions[session_id] = messages  # committed only if the run was not cancelled
        final_text = next(
            (m.content for m in reversed(messages) if m.role == "assistant" and m.content), ""
        )
        result = RunResult(
            content=final_text or "",
            messages=[*system, *messages],
            usage=usage,
            cost_usd=sum(costs) if costs else None,
            trace=tracer.events,
            error=error,
            stop_reason=stop_reason,
        )
        yield AgentEvent("done", {"stop_reason": stop_reason}, result)
