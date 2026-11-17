import json
from pathlib import Path

import pytest

from trellis_agent.eval.report import CaseResult, EvalReport, ReportError
from trellis_agent.eval.scorers import ScoreResult
from trellis_agent.types import Usage


def make_report(**overrides: object) -> EvalReport:
    cases = [
        CaseResult(
            "refund",
            0,
            "passed",
            content="退款流程:三个工作日内审核",
            stop_reason="end_turn",
            scores=[
                ScoreResult("run_succeeded", True, 1.0, "ok"),
                ScoreResult(
                    "llm_judge('mentions review')",
                    True,
                    0.9,
                    "good",
                    judge=True,
                    usage=Usage(200, 20),
                    cost_usd=0.0004,
                ),
            ],
            usage=Usage(1000, 100),
            judge_usage=Usage(200, 20),
            agent_cost_usd=0.002,
            judge_cost_usd=0.0004,
            judge_calls=1,
            model_requests=2,
            tool_calls=["search"],
        ),
        CaseResult(
            "math",
            0,
            "failed",
            content="6",
            scores=[ScoreResult("number_match(5)", False, 0.0, "no number equal to 5")],
            usage=Usage(100, 10),
            agent_cost_usd=0.0005,
            model_requests=1,
        ),
    ]
    fields: dict[str, object] = {
        "taskset_name": "demo",
        "taskset_version": "1",
        "taskset_hash": "a" * 64,
        "config": {"repeats": 1, "concurrency": 4, "budget_usd": None},
        "cases": cases,
        "meter_total_usd": 0.0029,
    }
    fields.update(overrides)
    return EvalReport(**fields)  # type: ignore[arg-type]


# ----------------------------------------------------------------- persistence
def test_save_then_load_round_trips(tmp_path: Path) -> None:
    report = make_report()
    path = tmp_path / "nested" / "dir" / "r.json"  # parent directories are created
    report.save(path)
    loaded = EvalReport.load(path)
    assert loaded.to_dict() == report.to_dict()
    assert loaded.cases[0].scores[1].usage == Usage(200, 20) and loaded.cases[0].tool_calls == [
        "search"
    ]


def test_saved_file_is_deterministic_readable_and_keeps_unicode(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    make_report().save(a)
    make_report().save(b)
    assert a.read_bytes() == b.read_bytes()
    text = a.read_text(encoding="utf-8")
    assert "退款流程" in text and "\\u" not in text  # not escaped
    assert text.endswith("\n") and '"schema_version": 1' in text


def test_timing_is_only_saved_when_present(tmp_path: Path) -> None:
    assert "timing_ms" not in make_report().to_dict()
    with_timing = make_report(timing={"refund#0": 12.5})
    assert with_timing.to_dict()["timing_ms"] == {"refund#0": 12.5}
    path = tmp_path / "t.json"
    with_timing.save(path)
    assert EvalReport.load(path).timing == {"refund#0": 12.5}


def test_summary_is_part_of_the_file_for_quick_reading(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    make_report().save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["passed"] == 1 and data["taskset"]["hash"] == "a" * 64


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d.update(schema_version=99), "schema_version"),
        (lambda d: d.pop("cases"), "cases"),
        (lambda d: d.pop("taskset"), "taskset"),
        (lambda d: d["cases"][0].pop("status"), "status"),
        (lambda d: d["cases"][0].update(status="weird"), "status"),
    ],
)
def test_invalid_report_files_are_rejected_clearly(
    tmp_path: Path, mutate: object, fragment: str
) -> None:
    data = make_report().to_dict()
    mutate(data)  # type: ignore[operator]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ReportError, match=fragment):
        EvalReport.load(path)


def test_unreadable_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="not found"):
        EvalReport.load(tmp_path / "missing.json")
    path = tmp_path / "junk.json"
    path.write_text("{nope")
    with pytest.raises(ReportError, match="not valid JSON"):
        EvalReport.load(path)


# ----------------------------------------------------------------- markdown
def test_markdown_separates_agent_and_judge_cost_and_lists_failures() -> None:
    md = make_report().render_markdown()
    assert md.startswith("# Eval report: demo v1")
    assert "aaaaaaaaaaaa" in md  # short hash
    assert "Pass rate | 50.0% (1/2 completed)" in md
    assert "Agent cost | $0.002500" in md
    assert "Judge cost (separate) | $0.000400" in md
    assert "Total cost | $0.002900" in md
    assert "Judge tokens (separate) | 200 in / 20 out" in md
    assert "Cost per success | $0.002900" in md
    assert "| math | 0 | failed |" in md and "number_match(5): no number equal to 5" in md
    assert "| refund | 0 | passed |" in md


def test_markdown_warns_about_incomplete_unpriced_and_estimated_data() -> None:
    cases = [
        CaseResult("a", 0, "passed", usage=Usage(10, 5, estimated=True), agent_cost_usd=None),
        CaseResult("b", 0, "aborted", agent_cost_usd=0.5),
        CaseResult("c", 0, "not_run"),
    ]
    md = make_report(cases=cases, aborted_by_budget=True, meter_total_usd=None).render_markdown()
    assert "ABORTED BY BUDGET" in md and "1 aborted, 1 not run" in md
    assert "unknown" in md and "no price" in md
    assert "estimated" in md


def test_markdown_for_an_empty_run_does_not_divide_by_zero() -> None:
    md = make_report(
        cases=[CaseResult("a", 0, "not_run")], aborted_by_budget=True
    ).render_markdown()
    assert "Pass rate | n/a" in md


# ----------------------------------------------------------------- reasoning tokens
def _with_reasoning() -> EvalReport:
    cases = [
        CaseResult(
            "a", 0, "passed", usage=Usage(100, 40, reasoning_tokens=30), agent_cost_usd=0.001
        ),
        CaseResult(
            "b", 0, "passed", usage=Usage(100, 60, reasoning_tokens=10), agent_cost_usd=0.001
        ),
    ]
    return make_report(cases=cases, meter_total_usd=None)


def test_the_summary_reports_reasoning_tokens_and_their_share_of_the_output() -> None:
    s = _with_reasoning().summary()
    assert s["usage"]["reasoning_tokens"] == 40 and s["usage"]["completion_tokens"] == 100
    assert s["reasoning_share"] == 0.4


def test_without_reasoning_data_the_summary_has_no_reasoning_keys() -> None:
    s = make_report().summary()
    assert "reasoning_share" not in s and "reasoning_tokens" not in s["usage"]


def test_markdown_shows_the_reasoning_row_only_when_known() -> None:
    assert "Reasoning tokens | 40 (40.0% of agent output)" in _with_reasoning().render_markdown()
    assert "Reasoning tokens" not in make_report().render_markdown()


def test_reasoning_tokens_survive_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    _with_reasoning().save(path)
    loaded = EvalReport.load(path)
    assert (
        loaded.cases[0].usage.reasoning_tokens == 30 and loaded.summary()["reasoning_share"] == 0.4
    )


def test_a_report_written_before_this_field_existed_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    make_report().save(path)
    assert "reasoning" not in path.read_text(encoding="utf-8")
    assert EvalReport.load(path).cases[0].usage.reasoning_tokens is None
