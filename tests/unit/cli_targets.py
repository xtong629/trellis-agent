"""Importable evaluation targets used by the CLI tests (``--target tests.unit.cli_targets:...``)."""

from __future__ import annotations

from trellis_agent.llm import LLMClient
from trellis_agent.types import RunResult, Usage


class _Broken:
    """A deliberately bad system: it answers every question with the same wrong sentence."""

    async def arun(self, query: str, *, session_id: str = "default") -> RunResult:
        return RunResult("The answer is 6", [], Usage(10, 5), None, [], None, "end_turn")


def broken(llm: LLMClient) -> _Broken:
    return _Broken()


def not_callable_factory() -> int:  # wrong signature on purpose
    return 1
