"""C4 (judge calibration) offline: answer set built from committed reports, references, the
consistency subset, categories and the analysis (no model calls)."""

from __future__ import annotations

from typing import Any

import pytest
from experiments import c4_judge

from trellis_agent.eval import load_taskset

V01 = {k: v for k, v in c4_judge.SOURCES.items() if not k.startswith("exp_c1")}


@pytest.fixture(scope="module")
def answers() -> dict[str, Any]:
    import asyncio

    saved = c4_judge.SOURCES
    c4_judge.SOURCES = V01
    try:
        return asyncio.run(c4_judge.build_answers())
    finally:
        c4_judge.SOURCES = saved


def test_answers_come_from_the_committed_v010_reports(answers: dict[str, Any]) -> None:
    rows = answers["rows"]
    excluded = sum(answers["excluded_error_runs"].values())
    assert len(rows) + excluded == 150 + 150 + 100 + 100 + 100
    assert len({r["id"] for r in rows}) == len(rows)
    # a passed case always passes its content rules; the converse can fail (tool checks)
    assert all(r["rule_content_pass"] for r in rows if r["case_status"] == "passed")
    assert answers["content_hash"] == c4_judge._hash(rows)


def test_reference_text_follows_the_scorers() -> None:
    a = {t.id: t for t in load_taskset(c4_judge.ROOT / "evals" / "exp_a_planner").tasks}
    b = {t.id: t for t in load_taskset(c4_judge.ROOT / "evals" / "exp_b_memory").tasks}
    assert c4_judge.expected_text(a["serial-01"]) == "the number 21 (within 0.01)"
    text = c4_judge.expected_text(b["deadline-01"])
    assert "'June 19'" in text and "does not give 'March 4'" in text


def test_consistency_subset_is_fixed_and_half_failures(answers: dict[str, Any]) -> None:
    rows = answers["rows"]
    subset = c4_judge.consistency_subset(rows)
    assert subset == c4_judge.consistency_subset(list(reversed(rows)))
    assert len(subset) == c4_judge.CONSISTENCY_N
    failing = {r["id"] for r in rows if not r["rule_content_pass"]}
    assert sum(i in failing for i in subset) == min(len(failing), c4_judge.CONSISTENCY_N // 2)


def test_categories() -> None:
    row = {"rule_content_pass": False, "rule_failures": []}
    fmt = {**row, "rule_failures": ["number_match: no number after the 'ANSWER:' answer line"]}
    old = {**row, "rule_failures": ["not_contains: output contains 'March 4'"]}
    assert c4_judge.category(fmt, True) == "format: right or wrong value, but no ANSWER line"
    assert c4_judge.category(old, True).startswith("mentions a value")
    assert c4_judge.category(row, True).startswith("value:")
    assert c4_judge.category({**row, "rule_content_pass": True}, False).startswith("judge rejected")
    assert c4_judge.category(row, False) is None


def test_analysis_counts_agreement_and_skips_judge_errors() -> None:
    rows = [
        {"id": f"r:{i}", "report": "exp_a_single", "rule_content_pass": i % 2 == 0,
         "rule_failures": [] if i % 2 == 0 else ["number_match: no number equal"],
         "case_status": "passed" if i % 2 == 0 else "failed", "content": "x"}
        for i in range(10)
    ]  # fmt: skip
    verdicts = {r["id"]: {"passed": r["rule_content_pass"], "reason": ""} for r in rows}
    verdicts["r:1"] = {"passed": True, "reason": "fine"}  # one disagreement
    verdicts["r:2"] = {"passed": None, "reason": "unreadable"}  # judge error
    out = c4_judge.analyse(rows, verdicts)
    assert (out["answers"], out["judged"], out["judge_errors"]) == (10, 9, 1)
    primary = out["primary_judge_vs_content_rule"]
    assert primary["n"] == 9 and primary["second_only"] == 0 and primary["first_only"] == 1
    assert out["disagreement_categories"] == {"value: the judge accepted what the rule rejects": 1}


def test_the_answer_set_is_frozen() -> None:
    import json

    data = json.loads(c4_judge.ANSWERS.read_text())
    frozen = json.loads((c4_judge.ROOT / "experiments" / "FROZEN.json").read_text())
    entry = frozen["v0.2"]["exp_c4_judge"]
    assert c4_judge._hash(data["rows"]) == data["content_hash"] == entry["content_hash"]
    assert len(data["rows"]) == entry["answers"] == 840
