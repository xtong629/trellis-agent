"""Task set C1 (v0.2 reflection experiment): data, answers, tools and the scoring of plausible
reply shapes. Written before any real call, so scorer problems show up here and not in a trial."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
from experiments.c1_tasks import (
    CATALOG,
    CONSTRAINT,
    STOCK,
    TRAP,
    constraint_answer,
    constraint_candidates,
    make_c1_tools,
    rate,
    taskset_c1,
    unit_price,
)

from trellis_agent.eval import ScorerSpec, build_scorer, parse_taskset, scorer_types
from trellis_agent.tools import ToolRegistry
from trellis_agent.types import Message, RunResult, ToolCall, Usage

LEVELS = (0, 1, 2)


def families(doc: dict[str, Any]) -> Counter[str]:
    return Counter(t["metadata"]["family"] for t in doc["tasks"])


@pytest.mark.parametrize("level", LEVELS)
def test_shape_names_and_validation(level: int) -> None:
    doc = taskset_c1(level)
    assert doc["name"] == "exp-c1-reflect" and doc["version"] == f"1-level{level}"
    assert families(doc) == {"deep": 6, "constraint": 6, "trap": 6, "easy": 6}
    ids = [t["id"] for t in doc["tasks"]]
    assert len(ids) == len(set(ids)) == 24
    for t in doc["tasks"]:
        types = {s["type"] for s in t["scorers"]}
        assert "llm_judge" not in types and types <= set(scorer_types())
        number = next(s for s in t["scorers"] if s["type"] == "number_match")
        assert number["which"] == "marked" and number["value"] == t["metadata"]["answer"]
        assert t["input"].endswith("ANSWER: <number>'.")
    parse_taskset(doc, known_scorers=scorer_types())


def test_levels_change_only_deep_and_constraint() -> None:
    by_level = {lv: {t["id"]: t for t in taskset_c1(lv)["tasks"]} for lv in LEVELS}
    for task_id, task in by_level[1].items():
        if task["metadata"]["family"] in ("trap", "easy"):
            assert by_level[0][task_id] == task == by_level[2][task_id]
        else:
            assert by_level[0][task_id]["input"] != task["input"] != by_level[2][task_id]["input"]


def test_hand_checked_answers_at_the_design_level() -> None:
    answers = {t["id"]: t["metadata"]["answer"] for t in taskset_c1(1)["tasks"]}
    # 10 rivets x 3 quill x 0.9 = 27 quill; quill -> marr -> sett = 2 x 4 = 8
    assert answers["deep-01"] == 216.0
    # 6 grommets x 7 sett x 0.75 = 31.5 sett; sett -> marr -> quill = 0.25 x 0.5
    assert answers["deep-03"] == 3.9375
    assert answers["constraint-01"] == 2.0  # bolt 24 per 12; anchor 5; rivet/screw short of stock
    assert answers["trap-01"] == 72.0  # 36 bolts = 3 packs of 12 at 24
    assert answers["easy-01"] == 3.0


def test_rates_chain_through_neighbours_only() -> None:
    assert rate("quill", "ovo") == 4.0 and rate("ovo", "quill") == 0.25
    assert rate("marr", "marr") == 1.0


@pytest.mark.parametrize("level", LEVELS)
def test_constraint_answers_are_unique(level: int) -> None:
    for category, warehouse, k in CONSTRAINT:
        ranked = constraint_candidates(category, warehouse, k, level)
        best = min(ranked.values())
        assert list(ranked.values()).count(best) == 1, (category, warehouse, k)


def test_the_traps_are_real_traps() -> None:
    """A reply that ignores the pack size gets a different number from the expected one."""
    for item, qty in TRAP:
        price, units = CATALOG[item][2], CATALOG[item][3]
        assert units > 1 and qty % units == 0
        assert qty * price != qty // units * price
    ignoring_packs = ignoring_stock = 0
    for category, warehouse, k in CONSTRAINT:
        expected = constraint_answer(category, warehouse, k, 1)
        eligible = [i for i in CATALOG if CATALOG[i][0] == category and STOCK[warehouse][i] >= k]
        ignoring_packs += min(CATALOG[i][2] for i in eligible) != expected
        ignoring_stock += constraint_answer(category, warehouse, k, 0) != expected
        assert expected == min(unit_price(i) for i in eligible)
    assert ignoring_packs == 6 and ignoring_stock == 3


async def test_tools_answer_and_refuse_as_documented() -> None:
    registry = ToolRegistry(make_c1_tools())

    async def call(name: str, **args: Any) -> tuple[str, bool]:
        out = await registry.call(ToolCall("x", name, args))
        return out.content, out.is_error

    item, err = await call("get_item", item="bolt")
    assert not err and json.loads(item)["price_is_for"] == "a pack of 12 units"
    assert json.loads((await call("get_item", item="rivet"))[0])["price_is_for"] == "1 unit"
    assert json.loads((await call("get_supplier", supplier="Pelling"))[0])["discount_percent"] == 25
    assert (await call("get_stock", warehouse="North", item="bolt"))[0] == "40"
    assert (await call("get_rate", from_currency="quill", to_currency="marr"))[0] == "2.0"
    text, err = await call("get_rate", from_currency="quill", to_currency="sett")
    assert err and "quill - marr - sett - ovo" in text
    assert json.loads((await call("list_items", category="lamps"))[0]) == [
        "bulb", "tube", "globe", "strip"
    ]  # fmt: skip
    assert (await call("list_items", category="tools"))[1]
    assert (await call("calculate", expression="27 * 2 * 4"))[0] == "216.0"


def _shapes(value: float) -> list[str]:
    """Plausible reply shapes for a correct value (answer line placed and decorated differently)."""
    v = f"{value:g}"
    two = f"{value:.2f}"
    return [
        f"ANSWER: {v}",
        f"The total is {v}.\n\nANSWER: {v}",
        f"Price 24 per pack of 12, so 3 packs.\nANSWER: {two}",
        f"Working: 27 * 8 = 216 and other numbers 3.5, 12\n**ANSWER: {v}**",
        f"answer: {v} quill",
        f"ANSWER: {v}\n(after a 10% discount, converted twice)",
        f"- ANSWER: {float(value):,.4f}",
    ]


async def _number_scorer(task: dict[str, Any]) -> Any:
    spec = next(s for s in task["scorers"] if s["type"] == "number_match")
    return build_scorer(ScorerSpec("number_match", {k: v for k, v in spec.items() if k != "type"}))


def _result(content: str) -> RunResult:
    return RunResult(content, [Message.assistant(content)], Usage(), 0.0)


@pytest.mark.parametrize("level", LEVELS)
async def test_golden_reply_shapes_pass_and_wrong_values_fail(level: int) -> None:
    for task in taskset_c1(level)["tasks"]:
        scorer = await _number_scorer(task)
        value = task["metadata"]["answer"]
        for shape in _shapes(value):
            outcome = await scorer.score(None, _result(shape))
            assert outcome.passed, (task["id"], shape, outcome.reason)
        for wrong in (value * 2 + 1, value + 0.5, 0.0 if value else 1.0):
            for shape in _shapes(wrong)[:3]:
                assert not (await scorer.score(None, _result(shape))).passed, (task["id"], shape)
        missing = await scorer.score(None, _result(f"The answer is {value:g}."))
        assert not missing.passed  # no marked line: a format failure, the same for every system


async def test_a_full_width_colon_is_not_read_as_the_marker() -> None:
    """Documented scorer limit, kept as is (the scorer is shared with the frozen v2 sets): tasks
    are in English and ask for 'ANSWER:', so a full-width colon counts as a format failure."""
    scorer = await _number_scorer(taskset_c1(1)["tasks"][0])
    assert not (await scorer.score(None, _result("ANSWER\uff1a216"))).passed


def test_c1_is_frozen_at_the_level_chosen_by_the_band_rule() -> None:
    """The committed file equals the generator at FROZEN_LEVEL, and its hash is the frozen one."""
    from pathlib import Path

    from experiments.c1_tasks import FROZEN_LEVEL

    from trellis_agent.eval import load_taskset

    root = Path(__file__).resolve().parents[2]
    entry = json.loads((root / "experiments" / "FROZEN.json").read_text())["v0.2"]["exp_c1_reflect"]
    committed = load_taskset(root / "evals" / "exp_c1_reflect")
    assert FROZEN_LEVEL == entry["level"] == 2
    assert committed.content_hash == entry["content_hash"]
    assert committed.content_hash == parse_taskset(taskset_c1(FROZEN_LEVEL)).content_hash
    assert (committed.version, len(committed.tasks)) == ("1-level2", entry["tasks"])
