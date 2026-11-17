from typing import Any

import pytest

from trellis_agent.eval.scorers import (
    CustomScorer,
    ScorerConfigError,
    ScoreResult,
    build_scorer,
    check_success,
    scorer_types,
)
from trellis_agent.eval.taskset import EvalTask, ScorerSpec
from trellis_agent.types import RunResult, TraceEvent, Usage

TASK = EvalTask(id="t", input="q")
DEFAULT_USAGE = Usage(10, 5)


def make_result(
    content: str = "",
    *,
    error: str | None = None,
    tools: tuple[str, ...] = (),
    usage: Usage = DEFAULT_USAGE,
    cost: float | None = None,
    duration_ms: float | None = None,
    stop_reason: str = "end_turn",
) -> RunResult:
    trace = [TraceEvent("tool_call", 0.0, {"name": n}, "root", None) for n in tools]
    if duration_ms is not None:
        trace.append(TraceEvent("run_end", 0.0, {"duration_ms": duration_ms}, "root", None))
    return RunResult(content, [], usage, cost, trace, error, stop_reason)


async def judge(kind: str, result: RunResult, **params: Any) -> ScoreResult:
    return await build_scorer(ScorerSpec(kind, params)).score(TASK, result)


# ----------------------------------------------------------------- text scorers
@pytest.mark.parametrize(
    ("params", "content", "passed"),
    [
        ({"value": "Paris"}, "Paris", True),
        ({"value": "Paris"}, "  Paris \n", True),  # strips whitespace by default
        ({"value": "Paris", "strip": False}, " Paris", False),
        ({"value": "Paris"}, "paris", False),
        ({"value": "Paris", "casefold": True}, "pARIS", True),
        ({"value": "Paris"}, "Paris, France", False),
    ],
)
async def test_exact_match(params: dict[str, Any], content: str, passed: bool) -> None:
    assert (await judge("exact_match", make_result(content), **params)).passed is passed


async def test_contains_and_not_contains() -> None:
    assert (await judge("contains", make_result("the answer is 5"), value="answer is 5")).passed
    miss = await judge("contains", make_result("nope"), value="answer is 5")
    assert not miss.passed and "answer is 5" in miss.reason
    assert (await judge("contains", make_result("ABC"), value="abc", casefold=True)).passed
    assert (await judge("not_contains", make_result("safe"), value="peanut")).passed
    leaked = await judge("not_contains", make_result("has Peanut"), value="peanut", casefold=True)
    assert not leaked.passed and "peanut" in leaked.reason


async def test_regex() -> None:
    assert (await judge("regex", make_result("order RF-20931 ok"), pattern=r"RF-\d{5}")).passed
    assert not (await judge("regex", make_result("none"), pattern=r"RF-\d{5}")).passed
    assert (await judge("regex", make_result("rf-1"), pattern=r"RF-\d", ignore_case=True)).passed


@pytest.mark.parametrize(
    ("content", "params", "passed"),
    [
        ("The total is 25.", {"value": 25}, True),
        ("It costs 1,234.50 dollars", {"value": 1234.5}, True),
        ("18C and 7 more", {"value": 7}, True),  # any number in the text may match
        ("18C and 7 more", {"value": 7, "which": "last"}, True),
        ("18C and 7 more", {"value": 18, "which": "last"}, False),
        ("about 24.9", {"value": 25, "tolerance": 0.2}, True),
        ("about 24.5", {"value": 25, "tolerance": 0.2}, False),
        ("no numbers here", {"value": 1}, False),
        ("negative -3 degrees", {"value": -3}, True),
    ],
)
async def test_number_match(content: str, params: dict[str, Any], passed: bool) -> None:
    assert (await judge("number_match", make_result(content), **params)).passed is passed


# ----------------------------------------------------------------- trace / cost scorers
async def test_tool_scorers_use_the_trace() -> None:
    r = make_result("x", tools=("add", "add", "search"))
    assert (await judge("tool_called", r, name="add")).passed
    assert (await judge("tool_called", r, name="add", min_times=2)).passed
    assert not (await judge("tool_called", r, name="add", min_times=3)).passed
    missing = await judge("tool_called", r, name="delete")
    assert not missing.passed and "delete" in missing.reason and "add" in missing.reason
    assert (await judge("tool_forbidden", r, name="delete")).passed
    assert not (await judge("tool_forbidden", r, name="search")).passed


async def test_budget_scorers() -> None:
    r = make_result("x", usage=Usage(100, 50), cost=0.02, duration_ms=120.0)
    assert (await judge("max_tokens", r, value=150)).passed
    assert not (await judge("max_tokens", r, value=149)).passed
    assert (await judge("max_cost_usd", r, value=0.02)).passed
    assert not (await judge("max_cost_usd", r, value=0.01)).passed
    assert (await judge("max_latency_ms", r, value=120.0)).passed
    assert not (await judge("max_latency_ms", r, value=100.0)).passed


async def test_unknowable_values_fail_closed_with_an_explanation() -> None:
    no_price = await judge("max_cost_usd", make_result("x", cost=None), value=1.0)
    assert not no_price.passed and "unknown" in no_price.reason
    no_timing = await judge("max_latency_ms", make_result("x"), value=1.0)
    assert not no_timing.passed and "latency" in no_timing.reason


async def test_stop_reason_scorer() -> None:
    assert (await judge("stop_reason", make_result("x"))).passed  # default end_turn
    stuck = await judge("stop_reason", make_result("x", stop_reason="max_steps"))
    assert not stuck.passed and "max_steps" in stuck.reason
    assert (
        await judge("stop_reason", make_result("x", stop_reason="max_steps"), value="max_steps")
    ).passed


# ----------------------------------------------------------------- results and names
async def test_score_result_shape() -> None:
    ok = await judge("contains", make_result("abc"), value="b")
    assert (ok.passed, ok.score, ok.judge, ok.error) == (True, 1.0, False, False)
    assert ok.name == "contains('b')"
    bad = await judge("contains", make_result("abc"), value="z")
    assert (bad.passed, bad.score) == (False, 0.0)
    assert ok.to_dict()["name"] == "contains('b')"


# ----------------------------------------------------------------- configuration errors
@pytest.mark.parametrize(
    ("kind", "params", "fragment"),
    [
        ("contains", {}, "value"),
        ("contains", {"value": 5}, "value"),
        ("contains", {"value": "x", "valeu": "typo"}, "valeu"),  # unknown parameter
        ("regex", {"pattern": "("}, "regex"),
        ("number_match", {"value": "5"}, "value"),
        ("number_match", {"value": 5, "which": "first"}, "which"),
        ("tool_called", {"name": "a", "min_times": 0}, "min_times"),
        ("max_tokens", {"value": -1}, "value"),
        ("nope", {}, "unknown scorer type"),
        ("llm_judge", {"rubric": "r"}, "judge"),  # needs a judge model
    ],
)
def test_bad_scorer_config_is_rejected_when_built(
    kind: str, params: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(ScorerConfigError, match=fragment):
        build_scorer(ScorerSpec(kind, params))


def test_registry_lists_known_types() -> None:
    assert {"exact_match", "contains", "not_contains", "regex", "number_match", "tool_called",
            "tool_forbidden", "max_latency_ms", "max_cost_usd", "max_tokens", "stop_reason",
            "llm_judge"} <= set(scorer_types())  # fmt: skip


# ----------------------------------------------------------------- implicit success check, custom
def test_check_success_default_expects_no_error() -> None:
    assert check_success(TASK, make_result("ok")).passed
    failed = check_success(TASK, make_result("", error="boom", stop_reason="llm_error"))
    assert not failed.passed and "boom" in failed.reason and "llm_error" in failed.reason


def test_check_success_can_expect_an_error() -> None:
    task = EvalTask(id="t", input="q", expect_success=False)
    assert check_success(task, make_result("", error="refused")).passed
    assert not check_success(task, make_result("fine")).passed


async def test_custom_scorer_accepts_several_return_shapes() -> None:
    assert (await CustomScorer("c", lambda t, r: True).score(TASK, make_result("x"))).passed
    assert (await CustomScorer("c", lambda t, r: None).score(TASK, make_result("x"))).passed
    bad = await CustomScorer("c", lambda t, r: False).score(TASK, make_result("x"))
    assert not bad.passed
    msg = await CustomScorer("c", lambda t, r: "output leaked a secret").score(
        TASK, make_result("x")
    )
    assert not msg.passed and msg.reason == "output leaked a secret"
    pair = await CustomScorer("c", lambda t, r: (False, "why")).score(TASK, make_result("x"))
    assert not pair.passed and pair.reason == "why"
    full = ScoreResult("c", True, 0.5, "half")
    assert (await CustomScorer("c", lambda t, r: full).score(TASK, make_result("x"))) is full


async def test_custom_scorer_exception_is_a_scorer_error_not_a_crash() -> None:
    def boom(task: EvalTask, result: RunResult) -> bool:
        raise RuntimeError("kaput")

    res = await CustomScorer("c", boom).score(TASK, make_result("x"))
    assert not res.passed and res.error and "kaput" in res.reason


# ------------------------------------------- marked answers (regression for taskset v1 -> v2)
# Replies of the kind a model gives: the answer plus its working. "which: last" took the last number
# in the text (1.5, 0, ...) and scored correct answers as wrong. The scorer now reads the number on
# an explicitly marked answer line, so working shown elsewhere cannot change the verdict.
@pytest.mark.parametrize(
    ("content", "value"),
    [
        ("The widget costs **21** krell (14 zorin x 1.5 exchange rate).\nANSWER: 21", 21),
        ("Harbor: 12, Summit: 0, Delta: 5. Total 17 across all three.\nANSWER: 17", 17),
        ("14 brackets in total (Harbor: 3, Summit: 11, Delta: 0).\n\nAnswer: 14", 14),
        ("ANSWER: 16.8 dax\n(21 krell x 0.8)", 16.8),
        ("working: 3 * 7 + 5 = 26\n**ANSWER: 26**", 26),
        ("ANSWER: 1\nOn reflection:\nANSWER: 2", 2),  # the last marked line is the final answer
    ],
)
async def test_marked_answer_uses_the_answer_line_not_the_working(
    content: str, value: float
) -> None:
    result = await judge(
        "number_match", make_result(content), value=value, tolerance=0.01, which="marked"
    )
    assert result.passed, result.reason


@pytest.mark.parametrize(
    "content",
    [
        "The widget costs **21** krell (14 zorin x 1.5 exchange rate).",  # correct but unmarked
        "working: 14 * 1.5 = 21\nANSWER: 1.5",  # wrong value on the marked line
        "ANSWER: none",
    ],
)
async def test_marked_answer_needs_a_matching_marked_number(content: str) -> None:
    result = await judge("number_match", make_result(content), value=21, which="marked")
    assert not result.passed and "ANSWER" in result.reason


async def test_marked_answer_marker_can_be_changed() -> None:
    ok = await judge(
        "number_match", make_result("FINAL = 7"), value=7, which="marked", marker="FINAL ="
    )
    assert ok.passed


# ------------------------------------------- contains is case-insensitive unless told otherwise
async def test_contains_ignores_case_by_default() -> None:
    assert (await judge("contains", make_result("Zandal nut"), value="zandal nut")).passed
    assert (await judge("contains", make_result("QUILLNUT"), value="quillnut")).passed
    leaked = await judge("not_contains", make_result("moved from JUNE 19"), value="June 19")
    assert not leaked.passed  # not_contains follows the same default


async def test_case_sensitive_matching_must_be_asked_for() -> None:
    strict = await judge("contains", make_result("Paris"), value="paris", casefold=False)
    assert not strict.passed
    assert (await judge("not_contains", make_result("Paris"), value="paris", casefold=False)).passed
