"""Targets for the experiments, for `--target module:callable` (each takes the LLM client).

A: ``a_single`` (one agent with every tool) vs ``a_planned`` (planner + research/compute workers).
B: ``b_window`` (sliding window only) vs ``b_memory`` (same window plus tiered memory). The window
size is the same, so the comparison is memory vs no memory at an equal working budget. With the
offline mock model (``model == "mock"``) these use experiment-specific mocks instead."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiments import mocks
from experiments.scenarios import SEPARATOR
from experiments.tools import make_tools
from experiments.window import B_SYSTEM_PROMPT
from trellis_agent.agent import Agent, AgentConfig
from trellis_agent.llm import LLMClient
from trellis_agent.memory.extract import Extraction, FactExtractor, LLMExtractor
from trellis_agent.memory.manager import MemoryConfig, MemoryManager
from trellis_agent.memory.store import MemoryStore
from trellis_agent.planner import PlannedAgent, Planner, Scheduler, WorkerSpec
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


def b_window(llm: LLMClient) -> Conversation:
    return _chat_agent(llm, remember=False)


def b_memory(llm: LLMClient) -> Conversation:
    return _chat_agent(llm, remember=True)
