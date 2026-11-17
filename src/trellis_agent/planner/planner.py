"""LLM planner: turns a goal into a validated task graph.

A model's plan is untrusted input. Every reply is parsed and validated (shape, dependencies,
cycles, size, known worker kinds); a rejected plan goes back to the model together with the exact
problems, up to a repair limit. Every attempt's tokens and cost are accounted, including failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..cost import BudgetExceeded
from ..jsonutil import extract_json
from ..llm import LLMClient, LLMError
from ..types import Message, Usage
from .graph import GraphError, Plan, PlanError, TaskGraph, TaskNode, WorkerSpec, parse_tasks

_OUTPUT_FORMAT = (
    'Reply with ONLY a JSON object: {"tasks": [{"id": "t1", "goal": "...", '
    '"kind": "<worker kind>", "depends_on": ["<id of a task whose result is needed first>"]}]}'
)
_OUTPUT_CAP = 500


def _truncate(text: str, limit: int = _OUTPUT_CAP) -> str:
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


class Planner:
    def __init__(
        self,
        llm: LLMClient,
        workers: Sequence[WorkerSpec],
        *,
        max_repair_attempts: int = 2,
        max_nodes: int = 20,
    ) -> None:
        if not workers:
            raise ValueError("a planner needs at least one worker kind to assign tasks to")
        if max_repair_attempts < 0 or max_nodes < 1:
            raise ValueError("max_repair_attempts must be >= 0 and max_nodes >= 1")
        self.llm = llm
        self.workers = list(workers)
        self.max_repair_attempts = max_repair_attempts
        self.max_nodes = max_nodes
        self._kinds = {w.kind for w in workers}

    # ------------------------------------------------------------ public API
    async def plan(self, goal: str) -> Plan:
        system = self._system_prompt(
            "Break the goal into a small graph of tasks. Make tasks independent where possible "
            "so they can run in parallel; add depends_on only when a task needs another's result."
        )
        user = f"GOAL:\n{goal}\n\nWORKERS:\n{self._worker_list()}"
        return await self._converse([Message.system(system), Message.user(user)], "plan")

    async def replan(
        self,
        goal: str,
        failed: TaskNode,
        error: str,
        done: Mapping[str, str],
        remaining: Sequence[TaskNode],
    ) -> Plan:
        system = self._system_prompt(
            "A task of an earlier plan failed. Return replacement tasks for ALL unfinished work "
            "(the failed task and everything that came after it). You may depend on finished task "
            "ids; do not reuse a finished id."
        )
        finished = "\n".join(f"- {t}: {_truncate(out)}" for t, out in done.items()) or "(none)"
        left = "\n".join(
            f"- {n.id} [{n.kind}] {n.goal} (depends on: {', '.join(n.depends_on) or 'nothing'})"
            for n in remaining
        )
        user = (
            f"GOAL:\n{goal}\n\nWORKERS:\n{self._worker_list()}\n\n"
            f"FAILED TASK: {failed.id} [{failed.kind}] {failed.goal}\nERROR: {error}\n\n"
            f"FINISHED TASKS (id: result):\n{finished}\n\nUNFINISHED TASKS (old plan):\n{left}"
        )
        pseudo = [TaskNode(t, "(finished)", self.workers[0].kind) for t in done]
        return await self._converse(
            [Message.system(system), Message.user(user)], "replan", finished=pseudo
        )

    # ------------------------------------------------------------ internals
    def _system_prompt(self, task: str) -> str:
        return (
            f"You are a planner. {task}\n{_OUTPUT_FORMAT}\n"
            f"Rules: task ids are short unique strings; kind must be one of the listed workers; "
            f"no dependency cycles; at most {self.max_nodes} tasks."
        )

    def _worker_list(self) -> str:
        return "\n".join(f"- {w.kind}: {w.description or '(no description)'}" for w in self.workers)

    def _parse(self, text: str, finished: Sequence[TaskNode]) -> TaskGraph:
        value = extract_json(text, kinds=(dict, list))
        if value is None:
            raise GraphError(["the reply was not valid JSON; reply with ONLY a JSON object"])
        if isinstance(value, dict):
            if "tasks" not in value:
                raise GraphError(["the JSON object must have a 'tasks' list"])
            value = value["tasks"]
        graph = parse_tasks(value, max_nodes=self.max_nodes)
        # Validate together with finished tasks so ids cannot collide and dependencies can resolve.
        TaskGraph([*finished, *graph.nodes], max_nodes=self.max_nodes + len(finished)).validate(
            self._kinds
        )
        return graph

    async def _converse(
        self, messages: list[Message], tag: str, finished: Sequence[TaskNode] = ()
    ) -> Plan:
        usage = Usage()
        costs: list[float | None] = []
        problems: list[str] = []

        def total() -> float | None:
            known = [c for c in costs if c is not None]
            return sum(known) if len(known) == len(costs) else None

        for attempt in range(1, self.max_repair_attempts + 2):
            try:
                response = await self.llm.chat(messages, temperature=0.0, tag=tag)
            except BudgetExceeded as exc:
                if exc.response is not None:
                    usage = usage + exc.response.usage
                    costs.append(exc.response.cost_usd)
                raise PlanError(
                    [f"planning stopped: {exc}"],
                    usage=usage,
                    cost_usd=total(),
                    attempts=attempt,
                    budget_exceeded=True,
                ) from exc
            except LLMError as exc:
                raise PlanError(
                    [f"planner model failed: {exc}"],
                    usage=usage,
                    cost_usd=total(),
                    attempts=attempt,
                ) from exc
            usage = usage + response.usage
            costs.append(response.cost_usd)
            text = response.message.content or ""
            try:
                graph = self._parse(text, finished)
            except GraphError as exc:
                problems = exc.problems
            else:
                return Plan(graph, usage, total(), attempt)
            messages = [
                *messages,
                Message.assistant(text),
                Message.user(
                    f"That plan was rejected: {'; '.join(problems)}. "
                    "Reply with the corrected plan as JSON only."
                ),
            ]
        raise PlanError(
            problems, usage=usage, cost_usd=total(), attempts=self.max_repair_attempts + 1
        )
