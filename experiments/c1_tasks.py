"""Task set C1 (`exp-c1-reflect`, v0.2): harder tool tasks for the reflection experiment.

All data is fictional and every expected answer is computed here from that data. The set is new
(own name and hash); the v2 sets of experiments A and B are not touched. Families:

* ``deep``: one order through several dependent steps (item, supplier discount, a currency
  conversion that needs two or three hops because only neighbouring currencies have a direct rate).
* ``constraint``: pick the cheapest item in a category that satisfies a stock condition; some
  items are priced per pack, so the price per single unit has to be worked out.
* ``trap``: the quantity is in single units while the item is priced per pack.
* ``easy``: one lookup (control; reflection is not expected to help).

``LEVEL`` is the single pre-registered difficulty knob (docs/EXPERIMENTS_V02.md). It changes only
the ``deep`` and ``constraint`` families: level 0 is easier, level 2 harder, level 1 is the design.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from trellis_agent.tools import Tool, tool

NAME = "exp-c1-reflect"
FROZEN_LEVEL = 2  # set by the pre-registered band rule after the trials (EXPERIMENTS_V02.md)
ASK = "End your reply with a line of the form 'ANSWER: <number>'."
_TOL = 0.01

SUPPLIERS = {  # supplier: (currency, discount in percent)
    "Orvane": ("quill", 10), "Kestrel": ("marr", 0), "Pelling": ("sett", 25), "Tamsin": ("ovo", 20),
}  # fmt: skip
CATALOG = {  # item: (category, supplier, price, units the price buys)
    "bolt": ("fasteners", "Orvane", 24, 12), "rivet": ("fasteners", "Orvane", 3, 1),
    "screw": ("fasteners", "Orvane", 15, 10), "anchor": ("fasteners", "Orvane", 5, 1),
    "gasket": ("seals", "Pelling", 40, 8), "oring": ("seals", "Pelling", 36, 24),
    "grommet": ("seals", "Pelling", 7, 1), "shim": ("seals", "Pelling", 2, 1),
    "lead": ("cables", "Kestrel", 12, 1), "braid": ("cables", "Kestrel", 90, 10),
    "coil": ("cables", "Kestrel", 55, 1), "clip": ("cables", "Kestrel", 20, 5),
    "bulb": ("lamps", "Tamsin", 16, 4), "tube": ("lamps", "Tamsin", 9, 1),
    "globe": ("lamps", "Tamsin", 30, 1), "strip": ("lamps", "Tamsin", 48, 6),
}  # fmt: skip
STOCK = {  # units on hand
    "North": {"bolt": 40, "rivet": 5, "screw": 0, "anchor": 12, "gasket": 16, "oring": 0,
              "grommet": 9, "shim": 2, "lead": 7, "braid": 0, "coil": 4, "clip": 25, "bulb": 0,
              "tube": 6, "globe": 3, "strip": 18},
    "South": {"bolt": 0, "rivet": 30, "screw": 50, "anchor": 12, "gasket": 0, "oring": 48,
              "grommet": 3, "shim": 20, "lead": 0, "braid": 30, "coil": 4, "clip": 10, "bulb": 12,
              "tube": 0, "globe": 9, "strip": 0},
}  # fmt: skip
CHAIN = ["quill", "marr", "sett", "ovo"]  # direct rates exist only between neighbours
DIRECT = {
    ("quill", "marr"): 2.0, ("marr", "quill"): 0.5, ("marr", "sett"): 4.0,
    ("sett", "marr"): 0.25, ("sett", "ovo"): 0.5, ("ovo", "sett"): 2.0,
}  # fmt: skip


def rate(a: str, b: str) -> float:
    """The rate from a to b along the chain (product of the direct rates on the way)."""
    i, j = CHAIN.index(a), CHAIN.index(b)
    step = 1 if j > i else -1
    return math.prod(DIRECT[(CHAIN[k], CHAIN[k + step])] for k in range(i, j, step))


def unit_price(item: str) -> float:
    _, _, price, units = CATALOG[item]
    return price / units


def order_cost(item: str, qty: int) -> float:
    """Cost of qty single units bought in whole packs, before discount, in the supplier currency."""
    _, _, price, units = CATALOG[item]
    return math.ceil(qty / units) * price


def currency_of(item: str) -> str:
    return SUPPLIERS[CATALOG[item][1]][0]


def discount_of(item: str) -> int:
    return SUPPLIERS[CATALOG[item][1]][1]


def items_in(category: str) -> list[str]:
    return [i for i, row in CATALOG.items() if row[0] == category]


# ------------------------------------------------------------------ tools
_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
}  # fmt: skip


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return float(_BINARY[type(node.op)](_evaluate(node.left), _evaluate(node.right)))
    raise ValueError("only numbers and + - * / with parentheses are allowed")


def make_c1_tools() -> list[Tool]:
    """A fresh set of the C1 tools. Deterministic; no state between calls."""

    @tool
    def list_items(category: str) -> list[str]:
        """List the items in a catalog category."""
        found = items_in(category)
        if not found:
            known = sorted({row[0] for row in CATALOG.values()})
            raise ValueError(f"unknown category {category!r}; categories: {', '.join(known)}")
        return found

    @tool
    def get_item(item: str) -> dict[str, Any]:
        """Look up a catalog item: its supplier and listed price."""
        if item not in CATALOG:
            raise ValueError(f"unknown item {item!r}")
        category, supplier, price, units = CATALOG[item]
        return {
            "item": item, "category": category, "supplier": supplier, "price": price,
            "currency": currency_of(item),
            "price_is_for": "1 unit" if units == 1 else f"a pack of {units} units",
        }  # fmt: skip

    @tool
    def get_supplier(supplier: str) -> dict[str, Any]:
        """Look up a supplier: its currency and the discount it gives on every order."""
        if supplier not in SUPPLIERS:
            raise ValueError(f"unknown supplier {supplier!r}")
        currency, discount = SUPPLIERS[supplier]
        return {"supplier": supplier, "currency": currency, "discount_percent": discount}

    @tool
    def get_stock(warehouse: str, item: str) -> int:
        """Units of an item on hand in a warehouse (North or South)."""
        if warehouse not in STOCK or item not in CATALOG:
            raise ValueError(f"unknown warehouse {warehouse!r} or item {item!r}")
        return STOCK[warehouse][item]

    @tool
    def get_rate(from_currency: str, to_currency: str) -> float:
        """How many units of to_currency one unit of from_currency buys."""
        if (from_currency, to_currency) not in DIRECT:
            raise ValueError(
                f"no direct rate from {from_currency!r} to {to_currency!r}; direct rates exist "
                f"only between neighbours in the chain {' - '.join(CHAIN)}"
            )
        return DIRECT[(from_currency, to_currency)]

    @tool
    def calculate(expression: str) -> float:
        """Evaluate an arithmetic expression using numbers and + - * / with parentheses."""
        try:
            return _evaluate(ast.parse(expression, mode="eval"))
        except SyntaxError as exc:
            raise ValueError(f"not a valid expression: {exc.msg}") from exc

    return [list_items, get_item, get_supplier, get_stock, get_rate, calculate]


# ------------------------------------------------------------------ task set
def _task(task_id: str, text: str, family: str, expected: str, why: str, value: float,
          tools: list[str]) -> dict[str, Any]:  # fmt: skip
    scorers: list[dict[str, Any]] = [
        {"type": "number_match", "value": round(value, 4), "tolerance": _TOL, "which": "marked"}
    ]
    scorers += [{"type": "tool_called", "name": name} for name in tools]
    return {
        "id": task_id, "input": f"{text} {ASK}", "scorers": scorers, "tags": [expected, family],
        "metadata": {"family": family, "expected": expected, "why": why, "answer": round(value, 4)},
    }  # fmt: skip


DEEP = {  # level: (item, quantity in single units, target currency)
    0: [("rivet", 10, "marr"), ("anchor", 8, "marr"), ("grommet", 6, "marr"),
        ("lead", 5, "sett"), ("tube", 10, "sett"), ("globe", 4, "sett")],
    1: [("rivet", 10, "sett"), ("anchor", 8, "ovo"), ("grommet", 6, "quill"),
        ("lead", 5, "ovo"), ("tube", 10, "marr"), ("globe", 4, "quill")],
    2: [("bolt", 24, "sett"), ("screw", 30, "ovo"), ("gasket", 16, "quill"),
        ("braid", 20, "ovo"), ("bulb", 8, "marr"), ("strip", 12, "quill")],
}  # fmt: skip
CONSTRAINT = [  # (category, warehouse, k)
    ("fasteners", "North", 10), ("seals", "North", 5), ("lamps", "North", 5),
    ("seals", "South", 10), ("lamps", "South", 5), ("cables", "North", 5),
]  # fmt: skip
TRAP = [("bolt", 36), ("screw", 50), ("gasket", 24), ("braid", 30), ("clip", 15), ("strip", 12)]
EASY = ["rivet", "anchor", "grommet", "shim", "lead", "tube"]


def deep_answer(item: str, qty: int, target: str) -> float:
    cost = order_cost(item, qty) * (1 - discount_of(item) / 100)
    return cost * rate(currency_of(item), target)


def constraint_candidates(category: str, warehouse: str, k: int, level: int) -> dict[str, float]:
    """Eligible items and the number each one is ranked by, for a level."""
    if level == 0:
        return {i: unit_price(i) for i in items_in(category)}
    eligible = [i for i in items_in(category) if STOCK[warehouse][i] >= k]
    if level == 1:
        return {i: unit_price(i) for i in eligible}
    return {i: float(order_cost(i, k)) for i in eligible}


def constraint_answer(category: str, warehouse: str, k: int, level: int) -> float:
    ranked = constraint_candidates(category, warehouse, k, level)
    return min(ranked.values())


def _constraint_text(category: str, warehouse: str, k: int, level: int) -> str:
    if level == 0:
        return (f"Among the items in the {category} category, which has the lowest price per "
                "single unit? Give that price per single unit in the supplier's currency, "
                "before any discount.")  # fmt: skip
    if level == 1:
        return (f"Among the items in the {category} category, consider only those with at least "
                f"{k} units on hand in the {warehouse} warehouse. Which of them has the lowest "
                "price per single unit? Give that price per single unit in the supplier's "
                "currency, before any discount.")  # fmt: skip
    return (f"I need exactly {k} single units of one item from the {category} category, taken "
            f"from the {warehouse} warehouse, so only items with at least {k} units on hand there "
            "qualify. Items sold in packs must be bought in whole packs. Which qualifying item "
            "is cheapest for this purchase? Give what the purchase costs in the supplier's "
            "currency, before any discount.")  # fmt: skip


def taskset_c1(level: int = 1) -> dict[str, Any]:
    if level not in DEEP:
        raise ValueError("level must be 0, 1 or 2")
    tasks: list[dict[str, Any]] = []
    why = "several dependent steps (item, discount, multi-hop conversion): a check can catch a skip"
    for n, (item, qty, target) in enumerate(DEEP[level], start=1):
        text = (f"We are ordering {qty} single units of {item}. Look up the item and its "
                "supplier, apply the supplier's discount, and give the total cost in "
                f"{target}.")  # fmt: skip
        tasks.append(_task(f"deep-{n:02d}", text, "deep", "benefit", why,
                           deep_answer(item, qty, target),
                           ["get_item", "get_supplier", "get_rate"]))  # fmt: skip
    why = "a condition and per-pack prices must both be applied: a check can catch a missed one"
    for n, (category, warehouse, k) in enumerate(CONSTRAINT, start=1):
        tools = ["list_items", "get_item"] + ([] if level == 0 else ["get_stock"])
        tasks.append(_task(f"constraint-{n:02d}", _constraint_text(category, warehouse, k, level),
                           "constraint", "benefit", why,
                           constraint_answer(category, warehouse, k, level), tools))  # fmt: skip
    why = "the price is per pack while the quantity is in single units"
    for n, (item, qty) in enumerate(TRAP, start=1):
        text = (f"How much do {qty} single units of {item} cost, in the supplier's own currency "
                "and before any discount?")  # fmt: skip
        tasks.append(_task(f"trap-{n:02d}", text, "trap", "benefit", why,
                           float(order_cost(item, qty)), ["get_item"]))  # fmt: skip
    why = "one lookup; nothing for a check to fix (control for harm)"
    for n, item in enumerate(EASY, start=1):
        text = f"What is the listed price of one {item}, in its supplier's currency?"
        tasks.append(_task(f"easy-{n:02d}", text, "easy", "no-benefit", why,
                           float(CATALOG[item][2]), ["get_item"]))  # fmt: skip
    return {"name": NAME, "version": f"1-level{level}", "tasks": tasks}
