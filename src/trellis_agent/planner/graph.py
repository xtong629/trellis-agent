"""Task graphs: nodes, dependencies and validation.

A plan produced by an LLM cannot be trusted. ``TaskGraph.validate`` reports *every* problem it finds
(unknown or self dependencies, cycles, size, unregistered kinds) in one ``GraphError`` whose text is
meant to be sent back to the planner so it can repair the plan in a single round trip.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..types import Usage


class GraphError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


class PlanError(RuntimeError):
    """The planner could not produce a usable plan. It still reports what the attempts cost."""

    def __init__(
        self,
        problems: list[str],
        *,
        usage: Usage | None = None,
        cost_usd: float | None = 0.0,
        attempts: int = 1,
        budget_exceeded: bool = False,
    ) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
        self.usage = usage or Usage()
        self.cost_usd = cost_usd
        self.attempts = attempts
        self.budget_exceeded = budget_exceeded


@dataclass(frozen=True)
class TaskNode:
    id: str
    goal: str
    kind: str = "default"  # which worker handles it
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A planner's answer: the graph, plus what producing it cost (planning is not free)."""

    graph: TaskGraph
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = 0.0
    attempts: int = 1


@dataclass(frozen=True)
class WorkerSpec:
    """What a worker is good at; shown to the planner so it can assign a ``kind`` to each task."""

    kind: str
    description: str = ""


class TaskGraph:
    def __init__(self, nodes: Collection[TaskNode], *, max_nodes: int = 20) -> None:
        self.nodes: tuple[TaskNode, ...] = tuple(nodes)
        self.max_nodes = max_nodes
        self._by_id: dict[str, TaskNode] = {}
        for item in self.nodes:
            self._by_id.setdefault(item.id, item)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._by_id

    def __iter__(self) -> Iterator[TaskNode]:
        return iter(self.nodes)

    def ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def get(self, task_id: str) -> TaskNode:
        return self._by_id[task_id]

    def ready(self, done: set[str]) -> list[TaskNode]:
        """Nodes not yet done whose dependencies are all done, in declaration order."""
        return [n for n in self.nodes if n.id not in done and all(d in done for d in n.depends_on)]

    def dependents(self, task_id: str) -> set[str]:
        """Every task that (transitively) depends on ``task_id``."""
        children: dict[str, list[str]] = {}
        for n in self.nodes:
            for dep in n.depends_on:
                children.setdefault(dep, []).append(n.id)
        found: set[str] = set()
        queue = deque([task_id])
        while queue:
            for child in children.get(queue.popleft(), []):
                if child not in found:
                    found.add(child)
                    queue.append(child)
        return found

    def validate(self, known_kinds: Collection[str] | None = None) -> None:
        problems: list[str] = []
        if not self.nodes:
            raise GraphError(["a plan needs at least one task"])
        if len(self.nodes) > self.max_nodes:
            problems.append(f"too many tasks: {len(self.nodes)}; use at most {self.max_nodes}")
        seen: set[str] = set()
        for n in self.nodes:
            if not n.id.strip():
                problems.append("a task has an empty id")
            elif n.id in seen:
                problems.append(f"duplicate task id {n.id!r}")
            seen.add(n.id)
        for n in self.nodes:
            for dep in n.depends_on:
                if dep == n.id:
                    problems.append(f"task {n.id!r} depends on itself")
                elif dep not in seen:
                    problems.append(f"task {n.id!r} depends on unknown task {dep!r}")
            if known_kinds is not None and n.kind not in known_kinds:
                options = ", ".join(sorted(known_kinds)) or "(none)"
                problems.append(f"task {n.id!r} has unknown kind {n.kind!r}; available: {options}")
        blocked = self._cycle_members(seen)
        if blocked:
            problems.append(
                "dependency cycle: tasks that can never start: " + ", ".join(sorted(blocked))
            )
        if problems:
            raise GraphError(problems)

    def _cycle_members(self, known: set[str]) -> set[str]:
        """Kahn's algorithm: repeatedly remove tasks with no unmet dependencies; what is left can
        never run, because it sits on (or behind) a cycle."""
        remaining = {
            n.id: {d for d in n.depends_on if d in known and d != n.id} for n in self.nodes
        }
        progress = True
        while progress:
            progress = False
            for task_id in [t for t, deps in remaining.items() if not deps]:
                del remaining[task_id]
                for deps in remaining.values():
                    deps.discard(task_id)
                progress = True
        return set(remaining)


def parse_tasks(raw: Any, *, max_nodes: int = 20) -> TaskGraph:
    """Build a graph from the list of task objects an LLM returned (extra keys are ignored)."""
    if not isinstance(raw, list):
        raise GraphError(["'tasks' must be a list of task objects"])
    if not raw:
        raise GraphError(["a plan needs at least one task"])
    problems: list[str] = []
    nodes: list[TaskNode] = []
    for index, item in enumerate(raw, start=1):
        label = f"task #{index}"
        if not isinstance(item, dict):
            problems.append(f"{label} must be an object")
            continue
        task_id, goal = item.get("id"), item.get("goal")
        kind, deps = item.get("kind"), item.get("depends_on")
        if not isinstance(task_id, str) or not task_id.strip():
            problems.append(f"{label}: 'id' must be a non-empty string")
            continue
        label = f"task {task_id!r}"
        if not isinstance(goal, str) or not goal.strip():
            problems.append(f"{label}: 'goal' must be a non-empty string")
        if kind is not None and not isinstance(kind, str):
            problems.append(f"{label}: 'kind' must be a string")
        if deps is not None and (
            not isinstance(deps, list) or not all(isinstance(d, str) for d in deps)
        ):
            problems.append(f"{label}: 'depends_on' must be a list of task ids")
        if problems:
            continue
        nodes.append(TaskNode(task_id, str(goal), kind or "default", tuple(deps or ())))
    if problems:
        raise GraphError(problems)
    return TaskGraph(nodes, max_nodes=max_nodes)
