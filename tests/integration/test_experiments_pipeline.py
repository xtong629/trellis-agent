"""The experiment pipeline end to end with mock models: it must run, score sensibly, and give
byte-identical reports on a repeat. Mock results say nothing about real models; they prove the
machinery (targets, tools, scorers, accounting) works before any money is spent."""

from __future__ import annotations

from pathlib import Path

import pytest

from trellis_agent.eval import EvalReport
from trellis_agent.eval.cli import main

ROOT = Path(__file__).resolve().parents[2]
PRICES = ROOT / "evals" / "prices.demo.yaml"
SETS = {"a": ROOT / "evals" / "exp_a_planner", "b": ROOT / "evals" / "exp_b_memory"}
TARGETS = {
    "a-single": ("a", "experiments.targets:a_single"),
    "a-planned": ("a", "experiments.targets:a_planned"),
    "b-window": ("b", "experiments.targets:b_window"),
    "b-memory": ("b", "experiments.targets:b_memory"),
    "b-full": ("b", "experiments.targets:b_full"),
}


def run(name: str, out: Path, *extra: str) -> EvalReport:
    taskset, target = TARGETS[name]
    argv = ["run", str(SETS[taskset]), "--llm", "mock", "--prices", str(PRICES)]
    argv += ["--target", target, "--out", str(out), "--meta", "run_date=2026-11-18", *extra]
    assert main(argv) == 0
    return EvalReport.load(out)


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, EvalReport]:
    tmp = tmp_path_factory.mktemp("exp")
    return {name: run(name, tmp / f"{name}.json") for name in TARGETS}


def by_family(report: EvalReport) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = {}
    for case in report.cases:
        family = case.task_id.rsplit("-", 1)[0]
        row = out.setdefault(family, [0, 0])
        row[0] += case.status == "passed"
        row[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def test_every_report_is_complete_and_priced(reports: dict[str, EvalReport]) -> None:
    for name, report in reports.items():
        assert not report.aborted_by_budget and report.summary()["meter_discrepancy"] == 0, name
        assert report.currency == "DEMO" and report.summary()["cost"] > 0, name
        assert report.config["meta"] == {"run_date": "2026-11-18"}


def test_mock_models_solve_experiment_a_with_and_without_the_planner(
    reports: dict[str, EvalReport],
) -> None:
    for name in ("a-single", "a-planned"):
        passed = sum(c.status == "passed" for c in reports[name].cases)
        assert passed == 30, f"{name}: {by_family(reports[name])}"


def test_the_planner_costs_extra_calls_on_single_lookups(reports: dict[str, EvalReport]) -> None:
    def calls(name: str, family: str) -> int:
        cases = [c for c in reports[name].cases if c.task_id.rsplit("-", 1)[0] == family]
        return sum(c.usage.total_tokens for c in cases)

    assert calls("a-planned", "single") > calls("a-single", "single")


def test_the_flaky_service_is_retried_not_fatal(reports: dict[str, EvalReport]) -> None:
    assert by_family(reports["a-single"])["flaky"] == (6, 6)
    assert by_family(reports["a-planned"])["flaky"] == (6, 6)


def test_the_window_baseline_forgets_early_facts_and_memory_keeps_them(
    reports: dict[str, EvalReport],
) -> None:
    window, memory = by_family(reports["b-window"]), by_family(reports["b-memory"])
    # The mock's replies are tiny, so its window holds more than a real model's would; it still
    # drops the early facts of the longest scripts.
    assert window["allergy"][0] < 6 and window["deadline"][0] < 5
    assert memory["allergy"] == (6, 6) and memory["deadline"] == (5, 5)
    assert memory["refund-zh"] == (3, 3)


def test_the_full_history_reference_keeps_everything_and_costs_the_most_input(
    reports: dict[str, EvalReport],
) -> None:
    full = reports["b-full"]
    assert all(c.status == "passed" for c in full.cases)  # nothing is ever truncated
    tokens = {n: sum(c.usage.prompt_tokens for c in r.cases) for n, r in reports.items()}
    assert tokens["b-full"] > tokens["b-window"]  # untruncated history is longer than the window


def test_no_benefit_families_pass_in_both_and_memory_costs_more(
    reports: dict[str, EvalReport],
) -> None:
    for family in ("fits", "irrelevant"):
        assert by_family(reports["b-window"])[family] == (3, 3)
        assert by_family(reports["b-memory"])[family] == (3, 3)
    assert reports["b-memory"].summary()["cost"] > reports["b-window"].summary()["cost"]


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_a_repeat_run_is_byte_identical(
    name: str, reports: dict[str, EvalReport], tmp_path: Path
) -> None:
    again = tmp_path / "again.json"
    run(name, again)
    first = tmp_path / "first.json"
    reports[name].save(first)
    assert first.read_bytes() == again.read_bytes()
