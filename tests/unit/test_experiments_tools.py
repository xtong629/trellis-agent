import pytest
from experiments.tools import CITIES, CURRENCIES, ITEMS, WAREHOUSES, make_tools

from trellis_agent.planner.retry import TransientError
from trellis_agent.tools import ToolRegistry
from trellis_agent.types import ToolCall


def run(registry: ToolRegistry, name: str, **args: object):  # type: ignore[no-untyped-def]
    import asyncio

    return asyncio.run(registry.call(ToolCall("c", name, dict(args))))


def test_data_is_fictional_and_complete() -> None:
    assert len(ITEMS) >= 8 and len(CITIES) >= 8 and len(CURRENCIES) == 3 and len(WAREHOUSES) == 3
    # nothing here can be answered from world knowledge
    assert not {"Tokyo", "Paris", "London", "Oslo", "Berlin"} & set(CITIES)


def test_lookup_tools_are_deterministic() -> None:
    reg = ToolRegistry(make_tools().lookup)
    assert (
        run(reg, "get_price", item="widget").content == run(reg, "get_price", item="widget").content
    )
    assert run(reg, "get_price", item="widget").content == str(ITEMS["widget"])
    stock = run(reg, "get_stock", warehouse=WAREHOUSES[0], item="widget")
    assert stock.content.isdigit() and not stock.is_error
    assert float(run(reg, "get_population", city=next(iter(CITIES))).content) > 0


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("get_price", {"item": "nope"}),
        ("get_stock", {"warehouse": "Nowhere", "item": "widget"}),
        ("get_population", {"city": "Nowhere"}),
        ("get_exchange_rate", {"from_currency": "a", "to_currency": "b"}),
    ],
)
def test_unknown_names_are_errors_not_guesses(tool: str, args: dict[str, object]) -> None:
    result = run(ToolRegistry(make_tools(flaky=False).lookup), tool, **args)
    assert result.is_error and result.error_type == "ValueError"


def test_the_exchange_rate_service_is_busy_on_the_first_call_of_each_bundle_only() -> None:
    a, b = CURRENCIES[0], CURRENCIES[1]
    flaky = ToolRegistry(make_tools(flaky=True).lookup)
    first = run(flaky, "get_exchange_rate", from_currency=a, to_currency=b)
    assert first.is_error and first.error_type == "TransientError" and "busy" in first.content
    second = run(flaky, "get_exchange_rate", from_currency=a, to_currency=b)
    assert not second.is_error and float(second.content) > 0
    fresh = ToolRegistry(
        make_tools(flaky=True).lookup
    )  # a new bundle (a new job) starts flaky again
    assert run(fresh, "get_exchange_rate", from_currency=a, to_currency=b).is_error
    steady = ToolRegistry(make_tools(flaky=False).lookup)
    assert not run(steady, "get_exchange_rate", from_currency=a, to_currency=b).is_error


def test_the_transient_error_is_the_retryable_kind() -> None:
    assert issubclass(TransientError, Exception)
    tools = make_tools(flaky=True)
    fn = next(t for t in tools.lookup if t.name == "get_exchange_rate")
    with pytest.raises(TransientError):
        fn.fn(CURRENCIES[0], CURRENCIES[1])


def test_calculate_does_arithmetic_and_nothing_else() -> None:
    calc = ToolRegistry(make_tools().calc)
    assert float(run(calc, "calculate", expression="(14 + 6) * 3 / 4").content) == 15.0
    assert float(run(calc, "calculate", expression="2 ** 3").content) == 8.0
    for bad in ("__import__('os').system('x')", "abs(3)", "1 +", "a + 1", "10 ** 10 ** 10"):
        assert run(calc, "calculate", expression=bad).is_error, bad


def test_the_bundles_cover_the_same_tools_split_by_role() -> None:
    tools = make_tools()
    names = lambda ts: {t.name for t in ts}  # noqa: E731
    assert names(tools.lookup) == {"get_price", "get_stock", "get_population", "get_exchange_rate"}
    assert names(tools.calc) == {"calculate"}
    assert names(tools.all) == names(tools.lookup) | names(tools.calc)
