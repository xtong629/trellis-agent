"""Generators for the two experiment task sets. Everything is derived from the fictional data in
``experiments.tools`` or written out here, so each expected answer is computable, never guessed.

Experiment A (planner on/off): tool tasks. Hypothesis per family is stored in ``metadata.why``.
Experiment B (tiered memory vs sliding window): multi-turn scripts; turns are joined by
``SEPARATOR`` inside ``input`` and only the last turn's reply is scored."""

from __future__ import annotations

from typing import Any

from experiments.tools import CITIES, ITEMS, RATES, STOCK, WAREHOUSES

SEPARATOR = "\n---\n"
_TOL = 0.01


def _task(
    task_id: str, text: str, family: str, expected: str, why: str, scorers: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": task_id,
        "input": text,
        "scorers": scorers,
        "tags": [expected, family],
        "metadata": {"family": family, "expected": expected, "why": why},
    }


def _answer(value: float) -> dict[str, Any]:
    return {"type": "number_match", "value": round(value, 4), "tolerance": _TOL, "which": "last"}


def _called(name: str) -> dict[str, Any]:
    return {"type": "tool_called", "name": name}


def taskset_a() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    conversions = [
        ("widget", "zorin", "krell"), ("gizmo", "krell", "dax"), ("flange", "dax", "zorin"),
        ("valve", "zorin", "dax"), ("coupler", "krell", "zorin"), ("bracket", "dax", "krell"),
        ("gadget", "zorin", "krell"), ("sprocket", "dax", "zorin"),
    ]  # fmt: skip
    why = "needs price -> rate -> multiply in order; a plan makes the dependencies explicit"
    for i, (item, a, b) in enumerate(conversions[:6]):
        rate = RATES[(a, b)]
        tasks.append(
            _task(
                f"serial-{i + 1:02d}",
                f"A {item} is listed at its price in {a}. Look the price up, then tell me what it "
                f"costs in {b} (use the exchange rate service). Answer with the number.",
                "serial", "benefit", why,
                [_answer(ITEMS[item] * rate), _called("get_price"), _called("get_exchange_rate")],
            )
        )  # fmt: skip
    for i, item in enumerate(["widget", "valve"], start=7):
        qty = 3 + i
        tasks.append(
            _task(
                f"serial-{i:02d}",
                f"Look up the price of {qty} {item}s and add 5 for handling. "
                "Use the calculator for the arithmetic. Answer with the number.",
                "serial", "benefit", why,
                [_answer(ITEMS[item] * qty + 5), _called("get_price"), _called("calculate")],
            )
        )  # fmt: skip

    why = "independent lookups can run in parallel before one final sum"
    for i, item in enumerate(["widget", "gizmo", "bracket", "valve"], start=1):
        total = sum(STOCK[w][item] for w in WAREHOUSES)
        names = ", ".join(WAREHOUSES)
        tasks.append(
            _task(
                f"fanout-{i:02d}",
                f"How many {item}s are in stock in total across the warehouses {names}? "
                "Check each warehouse. Answer with the number.",
                "fanout", "benefit", why,
                [_answer(total), _called("get_stock")],
            )
        )  # fmt: skip
    cities = list(CITIES)
    for i in range(4):
        trio = [cities[(2 * i + k) % len(cities)] for k in range(3)]
        total = sum(CITIES[c] for c in trio)
        tasks.append(
            _task(
                f"fanout-{i + 5:02d}",
                f"What is the combined population, in millions, of {', '.join(trio)}? "
                "Answer with the number.",
                "fanout", "benefit", why,
                [_answer(total), _called("get_population")],
            )
        )  # fmt: skip

    why = "the exchange-rate service is busy on its first call; tests failure handling and retry"
    for i, (item, a, b) in enumerate(conversions[2:8], start=1):
        tasks.append(
            _task(
                f"flaky-{i:02d}",
                f"Convert the price of one {item} from {a} to {b}: look up the price, then the "
                "exchange rate. If a service is busy, try again. Answer with the number.",
                "flaky", "benefit", why,
                [_answer(ITEMS[item] * RATES[(a, b)]), _called("get_exchange_rate")],
            )
        )  # fmt: skip

    why = "one lookup; planning adds only overhead"
    for i, item in enumerate(ITEMS, start=1):
        tasks.append(
            _task(
                f"single-{i:02d}",
                f"What is the unit price of a {item}? Answer with the number.",
                "single", "no-benefit", why,
                [_answer(ITEMS[item]), _called("get_price")],
            )
        )  # fmt: skip
    return {"name": "exp-a-planner", "version": "1", "tasks": tasks}


def _script(*turns: str) -> str:
    return SEPARATOR.join(turns)


_FILLER = [
    "Tell me a short fun fact about tides.",
    "Suggest a name for a small paper boat.",
    "What colour goes well with grey?",
    "Give me one tip for a long train ride.",
    "Name an animal that lives in cold places.",
    "Say something encouraging about Mondays.",
    "Describe a rainy afternoon in five words.",
    "Recommend a quiet hobby.",
]


def _fillers(n: int, offset: int) -> list[str]:
    return [_FILLER[(offset + k) % len(_FILLER)] for k in range(n)]


def taskset_b() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    why = "an early fact must survive many unrelated turns; a window that is too short drops it"
    allergens = ["kelpberry", "zandal nut", "moonroot", "fenwick plum", "ashgrain", "tarn seed"]
    for i, allergen in enumerate(allergens):
        text = _script(
            f"For the record: I am allergic to {allergen}.",
            *_fillers(6 + i, i),
            "Which ingredient must you never put in a snack for me? Name it and nothing else.",
        )
        tasks.append(
            _task(
                f"allergy-{i + 1:02d}", text, "allergy", "benefit", why,
                [{"type": "contains", "value": allergen}],
            )
        )  # fmt: skip

    why = "a fact is stated early; some scripts later correct it (the newer value must win)"
    dates = [("March 4", "June 19"), ("May 8", "July 2"), ("April 22", "May 30")]
    for i, (old, new) in enumerate(dates):
        text = _script(
            f"The report deadline is {old}.",
            *_fillers(6, i),
            f"Correction: the report deadline moved to {new}.",
            *_fillers(3, i + 3),
            "What is the report deadline now? Answer with the date.",
        )
        tasks.append(
            _task(
                f"deadline-{i + 1:02d}", text, "deadline", "benefit", why,
                [{"type": "contains", "value": new}, {"type": "not_contains", "value": old}],
            )
        )  # fmt: skip
    for i, date in enumerate(["August 11", "September 3"], start=4):
        text = _script(
            f"The report deadline is {date}.",
            *_fillers(7, i),
            "What is the report deadline? Answer with the date.",
        )
        tasks.append(
            _task(
                f"deadline-{i + 1:02d}", text, "deadline", "benefit", why,
                [{"type": "contains", "value": date}],
            )
        )  # fmt: skip

    why = "Chinese retrieval: the stored fact must be found by a Chinese question (jieba path)"
    for i, code in enumerate(["RF-4821", "RF-7305", "RF-1196"]):
        text = _script(
            f"我要申请退款,我的退款单号是 {code}。",
            *_fillers(6, i),
            "我之前说的退款单号是什么?只回答单号。",
        )
        tasks.append(
            _task(
                f"refund-zh-{i + 1:02d}", text, "refund-zh", "benefit", why,
                [{"type": "contains", "value": code}],
            )
        )  # fmt: skip

    why = "the fact is still inside a sliding window; memory has nothing to add"
    for i, allergen in enumerate(["quillnut", "brack leaf", "dun berry"]):
        text = _script(
            f"For the record: I am allergic to {allergen}.",
            _FILLER[i],
            "Which ingredient must you never put in a snack for me? Name it and nothing else.",
        )
        tasks.append(
            _task(
                f"fits-{i + 1:02d}", text, "fits", "no-benefit", why,
                [{"type": "contains", "value": allergen}],
            )
        )  # fmt: skip

    why = "the last question does not depend on anything said earlier; memory only costs tokens"
    for i, (a, b) in enumerate([(17, 25), (48, 19), (36, 27)]):
        text = _script(
            f"For the record: I am allergic to {allergens[i]}.",
            *_fillers(5, i),
            f"What is {a} plus {b}? Answer with the number.",
        )
        tasks.append(
            _task(
                f"irrelevant-{i + 1:02d}", text, "irrelevant", "no-benefit", why,
                [{"type": "number_match", "value": a + b, "which": "last"}],
            )
        )  # fmt: skip
    return {"name": "exp-b-memory", "version": "1", "tasks": tasks}
