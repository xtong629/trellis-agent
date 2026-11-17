from pathlib import Path

import pytest

from trellis_agent.cost import (
    BudgetExceeded,
    CostMeter,
    CurrencyMismatchError,
    ModelPrice,
    PriceTable,
    UnpricedModelError,
)
from trellis_agent.types import Usage

# Test-only prices: round numbers so the arithmetic is checkable by hand. NOT real vendor prices.
PRICES = PriceTable({"m": ModelPrice(input_per_million=2.0, output_per_million=10.0)})


def test_cost_is_tokens_times_rate() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    assert PRICES.cost("m", usage) == pytest.approx(2.0 + 5.0)


def test_unknown_model_cost_is_none_not_zero() -> None:
    assert PRICES.cost("other", Usage(prompt_tokens=10)) is None


def test_cached_tokens_use_cached_rate_when_configured() -> None:
    table = PriceTable(
        {
            "m": ModelPrice(
                input_per_million=2.0, output_per_million=10.0, cached_input_per_million=0.5
            )
        }
    )
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=400_000)
    # 600k fresh * 2.0 + 400k cached * 0.5 (per million)
    assert table.cost("m", usage) == pytest.approx(1.2 + 0.2)


def test_cached_tokens_fall_back_to_input_rate() -> None:
    usage = Usage(prompt_tokens=1_000_000, cached_tokens=400_000)
    assert PRICES.cost("m", usage) == pytest.approx(2.0)


def test_negative_price_rejected() -> None:
    with pytest.raises(ValueError):
        ModelPrice(input_per_million=-1, output_per_million=1)


def test_price_table_from_dict_and_yaml(tmp_path: Path) -> None:
    data = {"models": {"m": {"currency": "USD", "per": "1M tokens", "input": 1.0, "output": 3.0}}}
    assert PriceTable.from_dict(data).cost("m", Usage(1_000_000, 1_000_000)) == pytest.approx(4.0)
    path = tmp_path / "prices.yaml"
    path.write_text(
        "models:\n  m:\n    currency: USD\n    per: 1M tokens\n"
        "    input: 1.0\n    output: 3.0\n    cached_input: 0.1\n"
    )
    table = PriceTable.from_yaml(path)
    assert table.get("m") == ModelPrice(1.0, 3.0, 0.1, currency="USD")


def test_meter_accumulates_by_tag() -> None:
    meter = CostMeter(PRICES)
    meter.record("m", Usage(prompt_tokens=1_000_000), tag="plan")
    meter.record("m", Usage(completion_tokens=100_000), tag="work")
    meter.record("m", Usage(prompt_tokens=500_000), tag="plan")
    assert meter.total_usd == pytest.approx(2.0 + 1.0 + 1.0)
    assert meter.by_tag == {"plan": pytest.approx(3.0), "work": pytest.approx(1.0)}
    assert meter.calls == 3
    assert meter.usage.prompt_tokens == 1_500_000


def test_meter_raises_budget_exceeded_but_keeps_the_spend() -> None:
    meter = CostMeter(PRICES, budget_usd=3.0)
    meter.record("m", Usage(prompt_tokens=1_000_000))  # 2.0, within budget
    with pytest.raises(BudgetExceeded) as info:
        meter.record("m", Usage(prompt_tokens=1_000_000))  # 4.0 total > 3.0
    assert info.value.spent == pytest.approx(4.0) and info.value.budget == 3.0
    assert meter.total_usd == pytest.approx(4.0)  # the call already happened; never hide its cost


def test_spend_exactly_at_budget_is_allowed_but_exhausted() -> None:
    meter = CostMeter(PRICES, budget_usd=2.0)
    meter.record("m", Usage(prompt_tokens=1_000_000))
    assert meter.exhausted is True
    with pytest.raises(BudgetExceeded):
        meter.check()  # nothing left: do not start new work


def test_unpriced_model_without_budget_is_counted_not_charged() -> None:
    meter = CostMeter(PRICES)
    assert meter.record("other", Usage(prompt_tokens=10)) is None
    assert meter.total_usd == 0.0 and meter.unpriced_calls == 1
    assert meter.usage.prompt_tokens == 10


def test_unpriced_model_with_budget_fails_closed() -> None:
    meter = CostMeter(PRICES, budget_usd=1.0)
    with pytest.raises(UnpricedModelError):
        meter.record("other", Usage(prompt_tokens=10))


def test_meter_tracks_estimated_usage() -> None:
    meter = CostMeter(PRICES)
    meter.record("m", Usage(prompt_tokens=10, estimated=True))
    assert meter.any_estimated is True


def test_meter_remaining_budget() -> None:
    meter = CostMeter(PRICES, budget_usd=5.0)
    meter.record("m", Usage(prompt_tokens=1_000_000))
    assert meter.remaining_usd == pytest.approx(3.0)
    assert CostMeter(PRICES).remaining_usd is None


def test_budget_exceeded_reports_the_cost_of_the_call_that_crossed_it() -> None:
    meter = CostMeter(PRICES, budget_usd=3.0)
    meter.record("m", Usage(prompt_tokens=1_000_000))
    with pytest.raises(BudgetExceeded) as info:
        meter.record("m", Usage(prompt_tokens=1_000_000))
    assert info.value.call_cost == pytest.approx(2.0)
    assert info.value.response is None  # only the client knows the response; it attaches it


def test_total_does_not_depend_on_the_order_calls_finish() -> None:
    # Float addition is not associative, and with concurrent calls the finish order varies. The
    # meter must sum order-independently or two identical evaluations produce different reports.
    import itertools

    prices = PriceTable({"m": ModelPrice(input_per_million=100_000.0, output_per_million=0.0)})
    tokens = [1, 2, 3, 7, 100_001, 5, 123_457]  # irregular per-call costs (0.1 per token)
    totals: set[float] = set()
    tagged: set[float] = set()
    for order in itertools.permutations(tokens):
        meter = CostMeter(prices)
        for t in order:
            meter.record("m", Usage(prompt_tokens=t), tag="x")
        totals.add(meter.total_usd)
        tagged.add(meter.by_tag["x"])
    assert len(totals) == 1 and len(tagged) == 1
    assert totals == tagged  # the per-tag figure and the grand total agree exactly


# ----------------------------------------------------------------- price metadata and file format
FULL = {
    "currency": "CNY",
    "per": "1M tokens",
    "input": 0.8,
    "output": 2.8,
    "cached_input": 0.23,
    "source": "https://example.test/pricing",
    "checked_on": "2026-11-17",
    "note": "list price",
    "effective_from": "2026-01-01",
}


def test_a_price_carries_its_currency_and_provenance() -> None:
    price = PriceTable.from_dict({"models": {"m": FULL}}).get("m")
    assert price is not None and price.currency == "CNY"
    assert price.to_dict() == {
        "currency": "CNY",
        "per": "1M tokens",
        "input_per_million": 0.8,
        "output_per_million": 2.8,
        "cached_input_per_million": 0.23,
        "source": "https://example.test/pricing",
        "checked_on": "2026-11-17",
        "note": "list price",
        "effective_from": "2026-01-01",
    }


def test_yaml_dates_become_strings(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(
        "models:\n  m:\n    currency: CNY\n    per: 1M tokens\n    input: 1\n    output: 2\n"
        "    checked_on: 2026-11-17\n"
    )
    price = PriceTable.from_yaml(path).get("m")
    assert price is not None and price.checked_on == "2026-11-17"


@pytest.mark.parametrize(
    ("entry", "fragment"),
    [
        ({"per": "1M tokens", "input": 1, "output": 1}, "currency"),  # never guess a currency
        ({"currency": "USD", "input": 1, "output": 1}, "per"),
        ({"currency": "USD", "per": "1K tokens", "input": 1, "output": 1}, "1M tokens"),
        ({"currency": "", "per": "1M tokens", "input": 1, "output": 1}, "currency"),
        ({"currency": "USD", "per": "1M tokens", "input": 1}, "output"),
        ({"currency": "USD", "per": "1M tokens", "input": 1, "output": 1, "inptu": 2}, "inptu"),
        ({"currency": "USD", "per": "1M tokens", "input": -1, "output": 1}, "non-negative"),
    ],
)
def test_invalid_price_entries_are_rejected_with_the_reason(
    entry: dict[str, object], fragment: str
) -> None:
    with pytest.raises(ValueError, match=fragment):
        PriceTable.from_dict({"models": {"m": entry}})


def test_the_old_flat_format_is_rejected_with_a_migration_hint() -> None:
    with pytest.raises(ValueError, match="models"):
        PriceTable.from_dict({"m": {"input_per_million": 1.0, "output_per_million": 3.0}})


FX = {
    "from": "USD",
    "to": "CNY",
    "rate": 7.0,
    "source": "https://example.test/fx",
    "as_of": "2026-11-17",
}


def test_declared_fx_rates_are_available_only_for_the_declared_pair() -> None:
    table = PriceTable.from_dict({"models": {"m": FULL}, "fx": [FX]})
    rate = table.fx_rate("USD", "CNY")
    assert rate is not None and (rate.rate, rate.source, rate.as_of) == (
        7.0,
        FX["source"],
        "2026-11-17",
    )
    assert table.fx_rate("CNY", "USD") is None  # no inverse is guessed
    assert (
        PriceTable.from_dict({"models": {"m": FULL}}).fx_rate("USD", "CNY") is None
    )  # none built in


@pytest.mark.parametrize(
    "change",
    [{"source": ""}, {"as_of": ""}, {"rate": 0}, {"rate": -1}, {"to": "USD"}, {"rate": "x"}],
)
def test_every_fx_rate_needs_a_source_and_a_date_and_must_be_sane(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PriceTable.from_dict({"models": {"m": FULL}, "fx": [{**FX, **change}]})


# ----------------------------------------------------------------- currencies in the meter
def two_currency_table(with_fx: bool) -> PriceTable:
    def entry(cur: str) -> dict[str, object]:
        return {"currency": cur, "per": "1M tokens", "input": 1.0, "output": 0.0}

    data: dict[str, object] = {"models": {"cny-model": entry("CNY"), "usd-model": entry("USD")}}
    if with_fx:
        data["fx"] = [{**FX, "rate": 7.0}]  # 1 USD = 7 CNY
    return PriceTable.from_dict(data)


def test_the_meter_works_in_the_currency_of_the_first_priced_call() -> None:
    meter = CostMeter(two_currency_table(False))
    assert meter.currency is None
    assert meter.record("cny-model", Usage(prompt_tokens=1_000_000)) == pytest.approx(1.0)
    assert meter.currency == "CNY" and meter.total == pytest.approx(1.0)
    assert meter.by_currency == {"CNY": pytest.approx(1.0)} and meter.fx_applied == []


def test_a_second_currency_without_a_declared_rate_is_an_error_and_is_never_added() -> None:
    meter = CostMeter(two_currency_table(False))
    meter.record("cny-model", Usage(prompt_tokens=1_000_000))
    with pytest.raises(CurrencyMismatchError, match=r"USD.*CNY.*fx"):
        meter.record("usd-model", Usage(prompt_tokens=1_000_000))
    assert meter.total == pytest.approx(1.0)  # the dollars did NOT leak into the yuan total
    assert meter.by_currency == {
        "CNY": pytest.approx(1.0),
        "USD": pytest.approx(1.0),
    }  # but are kept


def test_with_a_declared_rate_the_amount_is_converted_and_the_conversion_is_annotated() -> None:
    meter = CostMeter(two_currency_table(True))
    meter.record("cny-model", Usage(prompt_tokens=1_000_000))
    converted = meter.record("usd-model", Usage(prompt_tokens=1_000_000))
    assert converted == pytest.approx(7.0)  # returned in the meter's currency
    assert meter.total == pytest.approx(8.0)
    assert meter.by_currency == {
        "CNY": pytest.approx(1.0),
        "USD": pytest.approx(1.0),
    }  # native, apart
    assert [
        (r["from_currency"], r["to_currency"], r["rate"], r["source"]) for r in meter.fx_applied
    ] == [("USD", "CNY", 7.0, FX["source"])]
    assert meter.by_tag == {"": pytest.approx(8.0)}


def test_a_budget_is_an_amount_in_a_currency() -> None:
    meter = CostMeter(two_currency_table(False), budget=1.5, currency="CNY")
    meter.record("cny-model", Usage(prompt_tokens=1_000_000))
    assert meter.remaining == pytest.approx(0.5) and meter.currency == "CNY"
    with pytest.raises(BudgetExceeded, match="CNY"):
        meter.record("cny-model", Usage(prompt_tokens=1_000_000))


def test_a_cost_outside_the_budget_currency_is_an_error_unless_a_rate_is_declared() -> None:
    meter = CostMeter(two_currency_table(False), budget=10, currency="CNY")
    with pytest.raises(CurrencyMismatchError):
        meter.record("usd-model", Usage(prompt_tokens=1_000_000))
    converted = CostMeter(two_currency_table(True), budget=10, currency="CNY")
    assert converted.record("usd-model", Usage(prompt_tokens=1_000_000)) == pytest.approx(7.0)
    with pytest.raises(BudgetExceeded):  # budget is enforced on the converted amount
        converted.record("usd-model", Usage(prompt_tokens=1_000_000))


def test_the_legacy_usd_budget_is_a_usd_budget_and_does_not_accept_yuan() -> None:
    meter = CostMeter(two_currency_table(False), budget_usd=5.0)
    assert meter.currency == "USD" and meter.budget_usd == 5.0
    with pytest.raises(CurrencyMismatchError):
        meter.record("cny-model", Usage(prompt_tokens=1_000_000))


def test_a_budget_needs_a_currency_and_only_one_budget_style() -> None:
    with pytest.raises(ValueError, match="currency"):
        CostMeter(PriceTable(), budget=5.0)
    with pytest.raises(ValueError, match="not both"):
        CostMeter(PriceTable(), budget_usd=1.0, budget=1.0, currency="USD")


def test_unknown_models_still_fail_closed_under_a_currency_budget() -> None:
    meter = CostMeter(two_currency_table(False), budget=10, currency="CNY")
    with pytest.raises(UnpricedModelError):
        meter.record("mystery", Usage(prompt_tokens=10))
