"""Prices, spend accounting and hard budgets.

Prices are supplied by the user (never built in): an unknown model yields ``None`` cost, not zero,
so missing prices cannot silently look like "free".
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .types import Usage

_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def __post_init__(self) -> None:
        values = (self.input_per_million, self.output_per_million, self.cached_input_per_million)
        if any(v is not None and v < 0 for v in values):
            raise ValueError("prices must be non-negative")


class PriceTable:
    def __init__(self, prices: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices = dict(prices or {})

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, Any]]) -> PriceTable:
        return cls({model: ModelPrice(**dict(fields)) for model, fields in data.items()})

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriceTable:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("price file must be a mapping of model name to prices")
        return cls.from_dict(loaded)

    def get(self, model: str) -> ModelPrice | None:
        return self._prices.get(model)

    def cost(self, model: str, usage: Usage) -> float | None:
        price = self.get(model)
        if price is None:
            return None
        cached_rate = (
            price.cached_input_per_million
            if price.cached_input_per_million is not None
            else price.input_per_million
        )
        fresh = usage.prompt_tokens - usage.cached_tokens
        return (
            fresh * price.input_per_million
            + usage.cached_tokens * cached_rate
            + usage.completion_tokens * price.output_per_million
        ) / _MILLION


class BudgetExceeded(RuntimeError):
    def __init__(self, spent: float, budget: float) -> None:
        super().__init__(f"budget exceeded: spent ${spent:.6f} of ${budget:.6f}")
        self.spent = spent
        self.budget = budget


class UnpricedModelError(ValueError):
    """A budget is set but the model has no price, so the budget cannot be enforced."""


class CostMeter:
    """Accumulates spend across calls; enforces an optional hard budget.

    ``record`` always books the spend first and only then raises, because the call has already
    happened and its cost must show up in reports.
    """

    def __init__(self, prices: PriceTable, budget_usd: float | None = None) -> None:
        if budget_usd is not None and budget_usd < 0:
            raise ValueError("budget must be non-negative")
        self.prices = prices
        self.budget_usd = budget_usd
        self.total_usd = 0.0
        self.by_tag: dict[str, float] = {}
        self.usage = Usage()
        self.calls = 0
        self.unpriced_calls = 0
        self.any_estimated = False
        self._lock = threading.Lock()

    @property
    def remaining_usd(self) -> float | None:
        return None if self.budget_usd is None else self.budget_usd - self.total_usd

    @property
    def exhausted(self) -> bool:
        return self.budget_usd is not None and self.total_usd >= self.budget_usd

    def check(self) -> None:
        """Raise if nothing is left to spend; call before starting new work."""
        if self.exhausted:
            assert self.budget_usd is not None
            raise BudgetExceeded(self.total_usd, self.budget_usd)

    def record(self, model: str, usage: Usage, *, tag: str = "") -> float | None:
        cost = self.prices.cost(model, usage)
        with self._lock:
            self.calls += 1
            self.usage = self.usage + usage
            self.any_estimated = self.any_estimated or usage.estimated
            if cost is None:
                self.unpriced_calls += 1
            else:
                self.total_usd += cost
                self.by_tag[tag] = self.by_tag.get(tag, 0.0) + cost
            total = self.total_usd
        if cost is None and self.budget_usd is not None:
            raise UnpricedModelError(f"no price for model {model!r}; cannot enforce the budget")
        if self.budget_usd is not None and total > self.budget_usd:
            raise BudgetExceeded(total, self.budget_usd)
        return cost
