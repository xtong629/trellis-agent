"""Task set C2 (`exp-c2-latency`, v0.2): independent lookups with slow, sometimes failing tools.

Supplements experiment A (it does not replace it). Every task needs 3 or 4 independent lookups and
one sum, like A's fanout family, but on new fictional data. The lookup tools sleep for a fixed
time before answering (a real ``asyncio.sleep``, so the planner's parallel workers can overlap it)
and can fail with a transient "busy" error. Both are properties of the tools, set per experiment
cell, not of the task set, so one frozen task set serves every cell.

Fault schedule: whether a call fails depends only on (seed, tool, canonical arguments, how many
times this exact call was made before in the same job), never on timing or call order, so the single
agent and the planner meet the same faults. A call never fails a third time in a row, so every task
stays solvable.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

from experiments.c1_tasks import _evaluate
from trellis_agent.planner.retry import TransientError
from trellis_agent.tools import Tool, tool

NAME = "exp-c2-latency"
ASK = "End your reply with a line of the form 'ANSWER: <number>'."
MAX_FAILS_IN_A_ROW = 2

DEPOTS = ["Arden", "Bexley", "Corrin", "Dalby"]
PRODUCTS = {  # units held per depot, in the order of DEPOTS
    "lantern": (14, 9, 0, 21), "kettle": (6, 18, 11, 4), "satchel": (0, 7, 25, 13),
    "compass": (31, 2, 8, 0), "blanket": (12, 12, 5, 19), "canteen": (3, 27, 14, 6),
    "trowel": (17, 0, 22, 9), "hammock": (8, 15, 1, 30),
}  # fmt: skip
STOCK = {d: {p: units[i] for p, units in PRODUCTS.items()} for i, d in enumerate(DEPOTS)}
TASKS = [  # (product, depots asked about)
    ("lantern", DEPOTS[:3]), ("kettle", DEPOTS), ("satchel", DEPOTS[1:]), ("compass", DEPOTS),
    ("blanket", DEPOTS[:3]), ("canteen", DEPOTS), ("trowel", DEPOTS[1:]), ("hammock", DEPOTS),
]  # fmt: skip


@dataclass(frozen=True)
class Cell:
    """One experimental condition: tool latency in seconds and the share of calls that fail."""

    latency_s: float
    fault_rate: float
    seed: int = 20261119

    @property
    def label(self) -> str:
        return f"l{self.latency_s:g}" + (f"f{self.fault_rate:g}" if self.fault_rate else "")


CELLS = [Cell(1.0, 0.0), Cell(4.0, 0.0), Cell(10.0, 0.0), Cell(4.0, 0.3)]


def fails(cell: Cell, tool_name: str, args: dict[str, Any], attempt: int) -> bool:
    """Deterministic fault decision for the attempt-th identical call (1-based)."""
    if cell.fault_rate <= 0 or attempt > MAX_FAILS_IN_A_ROW:
        return False
    canon = ",".join(f"{k}={str(v).strip().casefold()}" for k, v in sorted(args.items()))
    digest = hashlib.sha256(f"{cell.seed}|{tool_name}|{canon}|{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < cell.fault_rate


@dataclass
class C2Tools:
    lookup: list[Tool]
    calc: list[Tool]

    @property
    def all(self) -> list[Tool]:
        return [*self.lookup, *self.calc]


def make_c2_tools(cell: Cell) -> C2Tools:
    """A fresh set of tools for one job (the per-call attempt counter lives here)."""
    seen: Counter[str] = Counter()

    async def slow(name: str, args: dict[str, Any]) -> None:
        key = f"{name}|{sorted((k, str(v).strip().casefold()) for k, v in args.items())}"
        seen[key] += 1
        await asyncio.sleep(cell.latency_s)
        if fails(cell, name, args, seen[key]):
            raise TransientError("depot service busy, try again")

    @tool
    async def get_units(depot: str, product: str) -> int:
        """Units of a product held at a depot (Arden, Bexley, Corrin or Dalby). Slow."""
        await slow("get_units", {"depot": depot, "product": product})
        if depot not in STOCK or product not in PRODUCTS:
            raise ValueError(f"unknown depot {depot!r} or product {product!r}")
        return STOCK[depot][product]

    @tool
    def calculate(expression: str) -> float:
        """Evaluate an arithmetic expression using numbers and + - * / with parentheses."""
        import ast

        try:
            return _evaluate(ast.parse(expression, mode="eval"))
        except SyntaxError as exc:
            raise ValueError(f"not a valid expression: {exc.msg}") from exc

    return C2Tools([get_units], [calculate])


def taskset_c2() -> dict[str, Any]:
    tasks = []
    why = "independent slow lookups: parallel workers can overlap the waiting"
    for n, (product, depots) in enumerate(TASKS, start=1):
        total = sum(STOCK[d][product] for d in depots)
        text = (f"How many units of {product} are held in total at the depots "
                f"{', '.join(depots)}? Check each depot. {ASK}")  # fmt: skip
        tasks.append({
            "id": f"latency-{n:02d}", "input": text, "tags": ["latency", "benefit"],
            "scorers": [{"type": "number_match", "value": total, "tolerance": 0.01,
                         "which": "marked"},
                        {"type": "tool_called", "name": "get_units", "min_times": len(depots)}],
            "metadata": {"family": "latency", "expected": "benefit", "why": why, "answer": total,
                         "lookups": len(depots)},
        })  # fmt: skip
    return {"name": NAME, "version": "1", "tasks": tasks}
