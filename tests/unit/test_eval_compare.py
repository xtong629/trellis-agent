"""The regression gate. People read its text when CI goes red, so the *text* is tested."""

from trellis_agent.eval.baseline import Comparison, compare
from trellis_agent.eval.report import CaseResult, EvalReport
from trellis_agent.eval.scorers import ScoreResult
from trellis_agent.types import Usage


def case(
    task: str,
    repeat: int = 0,
    *,
    ok: bool = True,
    cost: float | None = 0.001,
    status: str | None = None,
    reason: str = "output does not contain 'answer is 5'",
    check: str = "contains('answer is 5')",
    content: str = "The answer is 6",
) -> CaseResult:
    scores = [ScoreResult("run_succeeded", True, 1.0, "ok")]
    if not ok:
        scores.append(ScoreResult(check, False, 0.0, reason))
    return CaseResult(
        task,
        repeat,
        status or ("passed" if ok else "failed"),
        content=content,
        scores=scores,
        usage=Usage(10, 5),
        agent_cost_usd=cost,
    )


def report(
    *cases: CaseResult, hash_: str = "a" * 64, name: str = "demo", version: str = "1", **kw: object
) -> EvalReport:
    return EvalReport(name, version, hash_, {}, list(cases), **kw)  # type: ignore[arg-type]


BASE = report(case("add"), case("greet"), case("search"))


def test_identical_runs_pass() -> None:
    result = compare(BASE, BASE)
    assert isinstance(result, Comparison)
    assert result.ok and result.exit_code == 0
    text = result.render()
    assert text.startswith("COMPARE OK") and "exit code 0" in text
    assert "100.0% -> 100.0%" in text


def test_a_regressed_task_fails_and_the_report_names_the_case_and_the_reason() -> None:
    current = report(case("add", ok=False), case("greet"), case("search"))
    result = compare(BASE, current)
    assert not result.ok and result.exit_code == 1
    text = result.render()
    assert text.startswith("COMPARE FAILED")
    assert "1 task regressed" in text
    assert "add" in text and "1/1 -> 0/1" in text
    assert "contains('answer is 5')" in text and "output does not contain 'answer is 5'" in text
    assert "The answer is 6" in text  # what the system actually said
    assert "66.7%" in text and "-33.3 pp" in text
    assert "exit code 1" in text
    assert [f.task_id for f in result.findings if f.kind == "regression"] == ["add"]


def test_regression_is_detected_per_task_across_repeats() -> None:
    base = report(case("t", 0), case("t", 1), case("t", 2))
    cur = report(case("t", 0), case("t", 1, ok=False), case("t", 2, ok=False))
    text = compare(base, cur).render()
    assert "3/3 -> 1/3" in text and "(repeat 1)" in text


def test_a_task_may_regress_when_explicitly_allowed_but_the_pass_rate_gate_still_applies() -> None:
    current = report(case("add", ok=False), case("greet"), case("search"))
    assert compare(BASE, current, max_regressed_tasks=1, max_pass_drop=0.5).ok
    blocked = compare(BASE, current, max_regressed_tasks=1, max_pass_drop=0.1)
    assert blocked.exit_code == 1 and "pass rate dropped 33.3 pp" in blocked.render()
    assert "limit 10.0 pp" in blocked.render()


def test_improvements_are_reported_but_never_fail() -> None:
    base = report(case("a", ok=False), case("b"))
    cur = report(case("a"), case("b"))
    result = compare(base, cur)
    assert result.ok
    text = result.render()
    assert "IMPROVEMENTS" in text and "a" in text and "0/1 -> 1/1" in text


def test_different_task_sets_cannot_be_compared() -> None:
    other = report(case("add"), case("greet"), case("search"), hash_="b" * 64, version="2")
    result = compare(BASE, other)
    assert result.exit_code == 2 and not result.ok
    text = result.render()
    assert "task set changed" in text and "aaaaaaaaaaaa" in text and "bbbbbbbbbbbb" in text
    assert "v1" in text and "v2" in text and "Re-run the baseline" in text
    assert "REGRESSIONS" not in text  # no misleading per-task verdicts on different questions


def test_different_task_set_names_cannot_be_compared() -> None:
    other = report(case("add"), name="other")
    result = compare(BASE, other)
    assert result.exit_code == 2 and "different task sets" in result.render()


def test_an_incomplete_current_run_is_not_a_verdict() -> None:
    current = report(
        case("add"),
        case("greet", status="aborted"),
        case("search", status="not_run"),
        aborted_by_budget=True,
    )
    result = compare(BASE, current)
    assert result.exit_code == 2
    text = result.render()
    assert "current run is incomplete" in text and "budget" in text and "not run" in text


def test_an_incomplete_baseline_is_rejected() -> None:
    base = report(case("add"), case("greet", status="not_run"), aborted_by_budget=True)
    result = compare(base, BASE)
    assert result.exit_code == 2 and "baseline is incomplete" in result.render()


def test_cost_gate() -> None:
    cheap = report(case("a", cost=0.001), case("b", cost=0.001))
    dear = report(case("a", cost=0.00125), case("b", cost=0.00125))
    assert compare(cheap, dear).ok  # no cost gate configured
    assert compare(cheap, dear, max_cost_increase_pct=30.0).ok
    blocked = compare(cheap, dear, max_cost_increase_pct=10.0)
    assert blocked.exit_code == 1
    text = blocked.render()
    assert "cost rose 25.0%" in text and "limit 10.0%" in text
    assert "$0.002000 -> $0.002500" in text


def test_cost_gate_fails_closed_when_cost_is_unknown() -> None:
    unpriced = report(case("a", cost=None))
    priced = report(case("a", cost=0.01))
    result = compare(priced, unpriced, max_cost_increase_pct=10.0)
    assert result.exit_code == 2 and "cost is unknown" in result.render()
    assert compare(priced, unpriced).ok  # without a cost gate it is not a problem


def test_header_shows_both_runs() -> None:
    text = compare(BASE, BASE).render()
    assert "baseline: demo v1 [aaaaaaaaaaaa]" in text and "current:  demo v1 [aaaaaaaaaaaa]" in text


def test_findings_are_in_a_stable_order() -> None:
    base = report(case("z"), case("a"), case("m"))
    cur = report(case("z", ok=False), case("a", ok=False), case("m", ok=False))
    ids = [f.task_id for f in compare(base, cur).findings if f.kind == "regression"]
    assert ids == ["z", "a", "m"]  # task-set order, not alphabetical or random
