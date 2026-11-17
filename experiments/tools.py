"""Deterministic fake tools for experiment A. All data is fictional, so nothing can be answered
from world knowledge: the model has to use the tools."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from trellis_agent.planner.retry import TransientError
from trellis_agent.tools import Tool, tool

ITEMS = {  # unit prices
    "widget": 14, "gadget": 9, "gizmo": 21, "sprocket": 5,
    "flange": 33, "bracket": 12, "valve": 27, "coupler": 18,
}  # fmt: skip
WAREHOUSES = ["Harbor", "Summit", "Delta"]
_UNITS = {  # units in stock, in the order of ITEMS; 0 means out of stock
    "Harbor": (12, 0, 4, 0, 7, 3, 0, 9),
    "Summit": (0, 8, 0, 6, 0, 11, 5, 0),
    "Delta": (5, 2, 9, 0, 0, 0, 14, 3),
}
STOCK = {warehouse: dict(zip(ITEMS, units, strict=True)) for warehouse, units in _UNITS.items()}
CITIES = {  # population in millions
    "Aldoria": 3.2, "Brevik": 1.8, "Cantera": 5.5, "Dunmore": 0.9, "Eastvale": 2.4,
    "Fenwick": 4.1, "Garnet": 1.1, "Halden": 6.3, "Ivory": 2.9, "Juneau": 0.7,
}  # fmt: skip
CURRENCIES = ["zorin", "krell", "dax"]
RATES = {
    ("zorin", "krell"): 1.5, ("krell", "zorin"): 0.6, ("krell", "dax"): 0.8,
    ("dax", "krell"): 1.25, ("dax", "zorin"): 2.0, ("zorin", "dax"): 0.5,
}  # fmt: skip

_BINARY: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}  # fmt: skip


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    ):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_evaluate(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return float(_BINARY[type(node.op)](left, right))
    raise ValueError("only numbers and + - * / ** with parentheses are allowed")


@dataclass
class ToolBundle:
    lookup: list[Tool]  # the data tools
    calc: list[Tool]  # arithmetic
    all: list[Tool]


def make_tools(*, flaky: bool = True) -> ToolBundle:
    """A fresh set of tools (one per job). With ``flaky`` the exchange-rate service is busy on its
    first call, a transient error that a retry fixes."""
    calls = {"rates": 0}

    @tool
    def get_price(item: str) -> int:
        """Look up the unit price of an item."""
        if item not in ITEMS:
            raise ValueError(f"unknown item {item!r}")
        return ITEMS[item]

    @tool
    def get_stock(warehouse: str, item: str) -> int:
        """Look up how many units of an item are in stock in a warehouse (0 means out of stock)."""
        if warehouse not in STOCK or item not in ITEMS:
            raise ValueError(f"unknown warehouse {warehouse!r} or item {item!r}")
        return STOCK[warehouse][item]

    @tool
    def get_population(city: str) -> float:
        """Look up the population of a city, in millions."""
        if city not in CITIES:
            raise ValueError(f"unknown city {city!r}")
        return CITIES[city]

    @tool
    def get_exchange_rate(from_currency: str, to_currency: str) -> float:
        """Look up how many units of to_currency one unit of from_currency buys."""
        calls["rates"] += 1
        if flaky and calls["rates"] == 1:
            raise TransientError("rate service busy, try again")
        if (from_currency, to_currency) not in RATES:
            raise ValueError(f"no rate for {from_currency!r} -> {to_currency!r}")
        return RATES[(from_currency, to_currency)]

    @tool
    def calculate(expression: str) -> float:
        """Evaluate an arithmetic expression using numbers and + - * / ** with parentheses."""
        try:
            return _evaluate(ast.parse(expression, mode="eval"))
        except SyntaxError as exc:
            raise ValueError(f"not a valid expression: {exc.msg}") from exc

    lookup = [get_price, get_stock, get_population, get_exchange_rate]
    return ToolBundle(lookup, [calculate], [*lookup, calculate])
