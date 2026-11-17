"""Prices, spend accounting and hard budgets.

Prices are supplied by the user (never built in): an unknown model yields ``None`` cost, not zero,
so missing prices cannot silently look like "free".
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .types import Usage

if TYPE_CHECKING:  # annotation only; llm imports cost at runtime
    from .llm.base import LLMResponse

_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """Price per million tokens in ``currency``, with where the number came from."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    currency: str = "USD"
    per: str = "1M tokens"  # the only unit supported; kept so a file says what its numbers mean
    source: str | None = None  # link to the page the numbers were read from
    checked_on: str | None = None  # date it was checked against the source
    note: str | None = None
    effective_from: str | None = None

    def __post_init__(self) -> None:
        values = (self.input_per_million, self.output_per_million, self.cached_input_per_million)
        if any(v is not None and v < 0 for v in values):
            raise ValueError("prices must be non-negative")
        if not self.currency.strip():
            raise ValueError("a price needs a currency")
        if self.per != "1M tokens":
            raise ValueError(f"unsupported price unit {self.per!r}; only '1M tokens' is supported")

    def to_dict(self) -> dict[str, Any]:
        """The fields that went into a cost number, for reports (unset metadata is left out)."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class FxRate:
    """A declared exchange rate: 1 ``from_currency`` = ``rate`` ``to_currency``. Never built in."""

    from_currency: str
    to_currency: str
    rate: float
    source: str
    as_of: str

    def __post_init__(self) -> None:
        if self.from_currency == self.to_currency or not self.from_currency or not self.to_currency:
            raise ValueError("an fx rate needs two different currencies")
        if not isinstance(self.rate, int | float) or isinstance(self.rate, bool) or self.rate <= 0:
            raise ValueError("an fx rate must be a positive number")
        if not self.source.strip() or not self.as_of.strip():
            raise ValueError("every fx rate needs a source and an as_of date")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENTRY_KEYS = {
    "currency",
    "per",
    "input",
    "output",
    "cached_input",
    "source",
    "checked_on",
    "note",
    "effective_from",
}


def _text(value: Any) -> str | None:
    return None if value is None else str(value)  # YAML turns 2026-11-17 into a date


def _parse_price(name: str, entry: Mapping[str, Any]) -> ModelPrice:
    unknown = sorted(set(entry) - _ENTRY_KEYS)
    if unknown:
        raise ValueError(f"price for {name!r} has unknown field(s) {unknown}")
    for required in ("currency", "per", "input", "output"):
        if entry.get(required) in (None, ""):
            raise ValueError(f"price for {name!r} is missing {required!r}")
    return ModelPrice(
        entry["input"], entry["output"], entry.get("cached_input"), str(entry["currency"]),
        str(entry["per"]), _text(entry.get("source")), _text(entry.get("checked_on")),
        _text(entry.get("note")), _text(entry.get("effective_from")),
    )  # fmt: skip


class PriceTable:
    def __init__(
        self, prices: Mapping[str, ModelPrice] | None = None, fx: Sequence[FxRate] = ()
    ) -> None:
        self._prices = dict(prices or {})
        self.fx = tuple(fx)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriceTable:
        models = data.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("price file needs a 'models' mapping (model name -> price entry)")
        unknown = sorted(set(data) - {"models", "fx"})
        if unknown:
            raise ValueError(f"price file has unknown top-level key(s) {unknown}")
        fx = [
            FxRate(r["from"], r["to"], r["rate"], str(r.get("source", "")), str(r.get("as_of", "")))
            for r in data.get("fx") or []
        ]
        return cls({name: _parse_price(name, entry) for name, entry in models.items()}, fx)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PriceTable:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("price file must be a mapping with a 'models' section")
        return cls.from_dict(loaded)

    def fx_rate(self, from_currency: str, to_currency: str) -> FxRate | None:
        """Only an explicitly declared pair; no inverse or cross rate is ever inferred."""
        return next(
            (
                r
                for r in self.fx
                if (r.from_currency, r.to_currency) == (from_currency, to_currency)
            ),
            None,
        )

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
    """Spending passed the budget.

    The call that crossed the line has already been paid for. ``call_cost`` is its price, and the
    LLM client attaches the full ``response`` (``None`` when raised by ``CostMeter.check`` or when
    no call was involved) so callers can still account for its tokens and cost.
    """

    def __init__(
        self, spent: float, budget: float, *, call_cost: float | None = None, currency: str = "USD"
    ) -> None:
        super().__init__(f"budget exceeded: spent {spent:.6f} of {budget:.6f} {currency}")
        self.currency = currency
        self.spent = spent
        self.budget = budget
        self.call_cost = call_cost
        self.response: LLMResponse | None = None


class UnpricedModelError(ValueError):
    """A budget is set but the model has no price, so the budget cannot be enforced."""


class CurrencyMismatchError(ValueError):
    """A cost is in a currency other than the meter's and no exchange rate was declared."""


class CostMeter:
    """Accumulates spend across calls; enforces an optional hard budget.

    All amounts are in one currency: the budget's, or else the first priced call's. A call priced
    in another currency is converted only with an FX rate declared in the price file (and the
    conversion is listed in ``fx_applied``); otherwise it is an error. Native per-currency totals
    are always kept in ``by_currency``. ``record`` books the spend first and only then raises,
    because the call has already happened and its cost must show up in reports.
    """

    def __init__(
        self,
        prices: PriceTable,
        budget_usd: float | None = None,
        *,
        budget: float | None = None,
        currency: str | None = None,
    ) -> None:
        if budget_usd is not None and budget is not None:
            raise ValueError("pass budget or budget_usd, not both")
        if budget is not None and not currency:
            raise ValueError("a budget needs a currency, e.g. budget=10, currency='CNY'")
        amount = budget if budget is not None else budget_usd
        if amount is not None and amount < 0:
            raise ValueError("budget must be non-negative")
        self.prices = prices
        self.budget = amount
        self.currency: str | None = (currency or "USD") if amount is not None else None
        # Per-call costs are kept and summed with math.fsum (exactly rounded, order-independent):
        # a running float total would depend on the order concurrent calls finish in.
        self._costs: list[float] = []
        self._native: dict[str, list[float]] = {}
        self._tag_costs: dict[str, list[float]] = {}
        self._fx: dict[tuple[str, str], FxRate] = {}
        self.usage = Usage()
        self.calls = 0
        self.unpriced_calls = 0
        self.any_estimated = False
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        with self._lock:
            return math.fsum(self._costs)

    total_usd = (
        total  # legacy alias: the amount is in ``currency``, which is only USD for USD prices
    )
    budget_usd = property(lambda self: self.budget)  # legacy alias

    @property
    def by_tag(self) -> dict[str, float]:
        with self._lock:
            return {tag: math.fsum(costs) for tag, costs in self._tag_costs.items()}

    @property
    def by_currency(self) -> dict[str, float]:
        """Spend per price currency, exactly as priced (never converted or added together)."""
        with self._lock:
            return {cur: math.fsum(costs) for cur, costs in self._native.items()}

    @property
    def fx_applied(self) -> list[dict[str, Any]]:
        return [rate.to_dict() for rate in self._fx.values()]

    @property
    def remaining(self) -> float | None:
        return None if self.budget is None else self.budget - self.total

    remaining_usd = remaining  # legacy alias

    @property
    def exhausted(self) -> bool:
        return self.budget is not None and self.total >= self.budget

    def check(self) -> None:
        """Raise if nothing is left to spend; call before starting new work."""
        if self.exhausted:
            assert self.budget is not None
            raise BudgetExceeded(self.total, self.budget, currency=self.currency or "USD")

    def record(self, model: str, usage: Usage, *, tag: str = "") -> float | None:
        """Book one call; returns its cost in the meter's currency (None if unpriced)."""
        price, cost, amount = self.prices.get(model), self.prices.cost(model, usage), None
        with self._lock:
            self.calls += 1
            self.usage = self.usage + usage
            self.any_estimated = self.any_estimated or usage.estimated
            if cost is None or price is None:
                self.unpriced_calls += 1
            else:
                self._native.setdefault(price.currency, []).append(cost)
                self.currency = self.currency or price.currency
                if price.currency == self.currency:
                    amount = cost
                elif (rate := self.prices.fx_rate(price.currency, self.currency)) is not None:
                    amount = cost * rate.rate
                    self._fx[(rate.from_currency, rate.to_currency)] = rate
                if amount is not None:
                    self._costs.append(amount)
                    self._tag_costs.setdefault(tag, []).append(amount)
            total = math.fsum(self._costs)
        if cost is None and self.budget is not None:
            raise UnpricedModelError(f"no price for model {model!r}; cannot enforce the budget")
        if price is not None and cost is not None and amount is None:
            raise CurrencyMismatchError(
                f"model {model!r} is priced in {price.currency} but this run's currency is "
                f"{self.currency}; declare an fx rate {price.currency}->{self.currency} (with a "
                "source and date) in the price file, or price every model in one currency"
            )
        if self.budget is not None and total > self.budget:
            raise BudgetExceeded(
                total, self.budget, call_cost=amount, currency=self.currency or "USD"
            )
        return amount
