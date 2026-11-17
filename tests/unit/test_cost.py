from pathlib import Path

import pytest

from trellis_agent.cost import (
    BudgetExceeded,
    CostMeter,
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
    data = {"m": {"input_per_million": 1.0, "output_per_million": 3.0}}
    assert PriceTable.from_dict(data).cost("m", Usage(1_000_000, 1_000_000)) == pytest.approx(4.0)
    path = tmp_path / "prices.yaml"
    path.write_text(
        "m:\n  input_per_million: 1.0\n  output_per_million: 3.0\n  cached_input_per_million: 0.1\n"
    )
    table = PriceTable.from_yaml(path)
    assert table.get("m") == ModelPrice(1.0, 3.0, 0.1)


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
