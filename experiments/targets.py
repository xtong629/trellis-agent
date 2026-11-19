"""Targets for the experiments, for `--target module:callable` (each takes the LLM client).

A: ``a_single`` (one agent with every tool) vs ``a_planned`` (planner + research/compute workers).
B: ``b_window`` (sliding window only) vs ``b_memory`` (same window plus tiered memory), and
``b_full`` (reference: the complete, untruncated history, no memory). The window size is the same
for the first two, so that comparison is memory vs no memory at an equal working budget. With the
offline mock model (``model == "mock"``) these use experiment-specific mocks instead."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from experiments import mocks
from experiments.c1_tasks import make_c1_tools
from experiments.c2_tasks import Cell, make_c2_tools
from experiments.scenarios import SEPARATOR
from experiments.tools import make_tools
from experiments.window import B_SYSTEM_PROMPT
from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.llm import LLMClient
from trellis_agent.memory.extract import Extraction, FactExtractor, LLMExtractor
from trellis_agent.memory.manager import MemoryConfig, MemoryManager
from trellis_agent.memory.store import MemoryStore
from trellis_agent.planner import PlannedAgent, Planner, Scheduler, WorkerSpec
from trellis_agent.reflect import LLMReflector, ReflectingAgent, ReflectionRecord
from trellis_agent.tools import Tool, ToolRegistry
from trellis_agent.types import Message, RunResult

WINDOW_TOKENS = 150  # derived by experiments/window.py (rule in docs/EXPERIMENTS.md); not tunable
_SPECS = [
    WorkerSpec("research", "looks up prices, stock, populations and exchange rates"),
    WorkerSpec("compute", "does arithmetic on numbers found by other tasks, using the calculator"),
]


def _is_mock(llm: LLMClient) -> bool:
    return llm.model == "mock"


def _meter(llm: LLMClient) -> Any:
    return getattr(llm, "meter", None)


def a_single(llm: LLMClient) -> Agent:
    model = mocks.mock(mocks.solver, _meter(llm)) if _is_mock(llm) else llm
    system = "Use the tools to answer. If a tool says a service is busy, call it again."
    return Agent(
        model, tools=ToolRegistry(make_tools().all), config=AgentConfig(system_prompt=system)
    )


def _worker(llm: LLMClient, tools: list[Tool], prompt: str) -> Agent:
    return Agent(llm, tools=ToolRegistry(tools), config=AgentConfig(system_prompt=prompt))


def a_planned(llm: LLMClient) -> PlannedAgent:
    meter = _meter(llm)
    if _is_mock(llm):
        plan, res, calc, synth = (
            mocks.mock(f, meter)
            for f in (mocks.planner, mocks.researcher, mocks.calculator, mocks.synthesizer)
        )
    else:
        plan = res = calc = synth = llm
    tools = make_tools()
    research = _worker(
        res, tools.lookup, "You are a researcher. Use the lookup tools. If busy, call again."
    )
    compute = _worker(calc, tools.calc, "You are a calculator. Use calculate for arithmetic.")
    scheduler = Scheduler({"research": research, "compute": compute})
    return PlannedAgent(Planner(plan, _SPECS), scheduler, synth)


class _NoFacts:
    """Extractor for the window baseline: stores nothing, costs nothing."""

    async def extract(self, turn: Sequence[Message]) -> Extraction:
        return Extraction([])


class Conversation:
    """Plays a scripted conversation (turns split by SEPARATOR) through one agent in one session
    and returns the last turn's result with the whole conversation's usage and cost."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        results = [await self.agent.arun(t, session_id=session_id) for t in query.split(SEPARATOR)]
        last = results[-1]
        usage = results[0].usage
        for r in results[1:]:
            usage = usage + r.usage
        costs = [r.cost_usd for r in results]
        cost = None if any(c is None for c in costs) else sum(c for c in costs if c is not None)
        trace = [e for r in results for e in r.trace]
        return RunResult(
            last.content, last.messages, usage, cost, trace, last.error, last.stop_reason
        )


def _chat_agent(llm: LLMClient, *, remember: bool) -> Conversation:
    model = mocks.mock(mocks.conversation_model, _meter(llm)) if _is_mock(llm) else llm
    extractor: FactExtractor = LLMExtractor(model) if remember else _NoFacts()
    memory = MemoryManager(
        MemoryStore(), extractor, MemoryConfig(working_token_budget=WINDOW_TOKENS)
    )
    config = AgentConfig(system_prompt=B_SYSTEM_PROMPT)
    return Conversation(Agent(model, memory=memory, config=config))


def b_full(llm: LLMClient) -> Conversation:
    """Upper-bound reference: the whole conversation is sent every turn (no window, no memory)."""
    model = mocks.mock(mocks.conversation_model, _meter(llm)) if _is_mock(llm) else llm
    return Conversation(Agent(model, config=AgentConfig(system_prompt=B_SYSTEM_PROMPT)))


def b_window(llm: LLMClient) -> Conversation:
    return _chat_agent(llm, remember=False)


def b_memory(llm: LLMClient) -> Conversation:
    return _chat_agent(llm, remember=True)


# ------------------------------------------------------------ v0.2, experiment C1 (reflection)
C1_SYSTEM = "Use the tools to answer. Tool results are the only source of data."
C1_MAX_STEPS = 12  # the same for every C1 system; deep tasks need more than the default 8 steps


def c1_agent(llm: LLMClient) -> Agent:
    model = mocks.mock(mocks.c1_solver, _meter(llm)) if _is_mock(llm) else llm
    config = AgentConfig(system_prompt=C1_SYSTEM, max_steps=C1_MAX_STEPS)
    return Agent(model, tools=ToolRegistry(make_c1_tools()), config=config)


def c1_baseline(llm: LLMClient) -> Agent:
    """C1 without reflection: one agent with all C1 tools."""
    return c1_agent(llm)


def c1_reflect(
    llm: LLMClient, on_record: Callable[[ReflectionRecord], None] | None = None
) -> ReflectingAgent:
    """C1 with reflection: the same agent, checked by the same model, at most one retry."""
    checker = mocks.mock(mocks.c1_checker, _meter(llm)) if _is_mock(llm) else llm
    return ReflectingAgent(c1_agent(llm), LLMReflector(checker), on_record=on_record)


# ------------------------------------------------------------ v0.2, experiment C2 (latency, faults)
C2_SYSTEM = "Use the tools to answer. If a tool says a service is busy, call it again."
C2_MAX_STEPS = 12
_C2_SPECS = [
    WorkerSpec("research", "looks up how many units of a product a depot holds"),
    WorkerSpec("compute", "adds up numbers found by other tasks, using the calculator"),
]


def c2_single(llm: LLMClient, cell: Cell) -> Agent:
    """One agent with every C2 tool; it runs its tool calls one after another."""
    model = mocks.mock(mocks.c2_solver, _meter(llm)) if _is_mock(llm) else llm
    config = AgentConfig(system_prompt=C2_SYSTEM, max_steps=C2_MAX_STEPS)
    return Agent(model, tools=ToolRegistry(make_c2_tools(cell).all), config=config)


def c2_planned(llm: LLMClient, cell: Cell) -> PlannedAgent:
    """Planner + research/compute workers (scheduler default: up to 4 workers in parallel)."""
    meter = _meter(llm)
    if _is_mock(llm):
        plan, res, calc, synth = (
            mocks.mock(f, meter)
            for f in (mocks.c2_planner, mocks.researcher, mocks.calculator, mocks.synthesizer)
        )
    else:
        plan = res = calc = synth = llm
    tools = make_c2_tools(cell)
    research = _worker(
        res, tools.lookup, "You are a researcher. Use the lookup tools. If busy, call again."
    )
    compute = _worker(calc, tools.calc, "You are a calculator. Use calculate for arithmetic.")
    scheduler = Scheduler({"research": research, "compute": compute})
    return PlannedAgent(Planner(plan, _C2_SPECS), scheduler, synth)
