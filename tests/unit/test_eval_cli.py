"""End-to-end tests of the evaluation CLI on the committed smoke task set and demo mocks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FakeOpenAI
from trellis_agent.eval.baseline import compare
from trellis_agent.eval.cli import main
from trellis_agent.eval.report import EvalReport

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "evals" / "smoke"
PRICES = ROOT / "evals" / "prices.demo.yaml"
BASELINE = ROOT / "evals" / "baselines" / "smoke.json"


def run_smoke(out: Path, *extra: str) -> int:
    return main(
        ["run", str(SMOKE), "--llm", "mock", "--prices", str(PRICES), "--out", str(out), *extra]
    )


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# ----------------------------------------------------------------- the smoke run itself
def test_smoke_run_passes_and_reports_agent_and_judge_cost_separately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.json"
    assert run_smoke(out, "--md", str(tmp_path / "r.md")) == 0
    summary = load(out)["summary"]
    assert (
        summary["pass_rate"] == 1.0 and summary["failed"] == 0 and not summary["aborted_by_budget"]
    )
    assert summary["agent_cost_usd"] > 0 and summary["judge_cost_usd"] > 0
    assert summary["cost_usd"] == pytest.approx(
        summary["agent_cost_usd"] + summary["judge_cost_usd"]
    )
    assert summary["judge_usage"]["prompt_tokens"] > 0
    assert summary["meter_discrepancy_usd"] == pytest.approx(0.0)  # report cost == meter cost
    stdout = capsys.readouterr().out
    assert "# Eval report: smoke" in stdout and "Judge cost (separate)" in stdout
    assert (tmp_path / "r.md").read_text(encoding="utf-8") == stdout


def test_two_runs_of_the_same_task_set_are_byte_identical(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    args = ("--repeats", "2", "--concurrency", "4")
    assert run_smoke(a, "--md", str(tmp_path / "a.md"), *args) == 0
    assert run_smoke(b, "--md", str(tmp_path / "b.md"), *args) == 0
    assert a.read_bytes() == b.read_bytes()
    assert (tmp_path / "a.md").read_bytes() == (tmp_path / "b.md").read_bytes()


def test_concurrency_changes_nothing_but_the_recorded_setting(tmp_path: Path) -> None:
    fast, serial = tmp_path / "fast.json", tmp_path / "serial.json"
    assert run_smoke(fast, "--repeats", "2", "--concurrency", "8") == 0
    assert run_smoke(serial, "--repeats", "2", "--concurrency", "1") == 0
    a, b = load(fast), load(serial)
    assert (a["config"].pop("concurrency"), b["config"].pop("concurrency")) == (8, 1)
    assert a == b  # same cases, same order, same costs


def test_byte_identical_through_the_real_command_line_entry_point(tmp_path: Path) -> None:
    outs = [tmp_path / "1.json", tmp_path / "2.json"]
    for out in outs:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "trellis_agent.eval",
                "run",
                str(SMOKE),
                "--llm",
                "mock",
                "--prices",
                str(PRICES),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=ROOT,
            check=False,
        )
        assert done.returncode == 0, done.stderr
    assert outs[0].read_bytes() == outs[1].read_bytes()


def test_the_committed_baseline_is_still_valid(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert run_smoke(out) == 0
    result = compare(EvalReport.load(BASELINE), EvalReport.load(out))
    assert result.ok, result.render()


# ----------------------------------------------------------------- the regression workflow
def test_regression_workflow_names_the_failing_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good, bad, saved = tmp_path / "good.json", tmp_path / "bad.json", tmp_path / "base.json"
    assert run_smoke(good) == 0
    assert main(["save-baseline", str(good), "--out", str(saved)]) == 0
    assert "saved baseline" in capsys.readouterr().out
    assert saved.read_bytes() == good.read_bytes()

    assert run_smoke(bad, "--target", "tests.unit.cli_targets:broken") == 0  # run itself succeeds
    capsys.readouterr()
    code = main(["compare", str(saved), str(bad)])
    text = capsys.readouterr().out
    assert code == 1
    assert text.startswith("COMPARE FAILED")
    assert "add-numbers: passed 1/1 -> 0/1" in text
    assert "number_match(5)" in text and "no number equal to 5" in text
    assert "The answer is 6" in text  # what the system actually said
    assert "exit code 1" in text
    assert main(["compare", str(saved), str(good)]) == 0
    assert capsys.readouterr().out.startswith("COMPARE OK")


def test_compare_honours_the_gates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    run_smoke(good)
    run_smoke(bad, "--target", "tests.unit.cli_targets:broken")
    capsys.readouterr()
    # Even if every task may regress, the overall pass-rate gate still trips.
    code = main(["compare", str(good), str(bad), "--max-regressed-tasks", "100"])
    assert code == 1 and "pass rate dropped" in capsys.readouterr().out
    code = main(
        ["compare", str(good), str(bad), "--max-regressed-tasks", "100", "--max-pass-drop", "1.0"]
    )
    assert code == 0


# ----------------------------------------------------------------- budgets
def test_budget_abort_exits_3_and_is_flagged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.json"
    code = run_smoke(out, "--budget-usd", "0.0003", "--concurrency", "1")
    assert code == 3
    summary = load(out)["summary"]
    assert summary["aborted_by_budget"] is True
    assert summary["not_run_jobs"] + summary["aborted_jobs"] > 0
    assert "ABORTED BY BUDGET" in capsys.readouterr().out
    # a partial run must never become a baseline
    assert main(["save-baseline", str(out), "--out", str(tmp_path / "b.json")]) == 2
    assert "incomplete" in capsys.readouterr().err


def test_a_budget_requires_prices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["run", str(SMOKE), "--llm", "mock", "--budget-usd", "1", "--out", str(tmp_path / "r.json")]
    )
    assert code != 0  # fail closed: the budget cannot be enforced on unpriced calls


# ----------------------------------------------------------------- errors are explained, exit 2
def test_configuration_errors_exit_2_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["run", str(tmp_path / "nope"), "--llm", "mock"]) == 2
    assert "not found" in capsys.readouterr().err

    typo = tmp_path / "typo.yaml"
    typo.write_text(
        "name: t\nversion: 1\ntasks:\n"
        "  - id: a\n    input: hi\n    scorers: [{type: contanis, value: x}]\n"
    )
    assert main(["run", str(typo), "--llm", "mock"]) == 2
    assert "unknown scorer type 'contanis'" in capsys.readouterr().err

    assert main(["run", str(SMOKE), "--llm", "mock", "--target", "nonexistent_mod:f"]) == 2
    assert "nonexistent_mod" in capsys.readouterr().err
    assert main(["run", str(SMOKE), "--llm", "mock", "--target", "no_colon"]) == 2
    assert "module:callable" in capsys.readouterr().err
    assert main(["run", str(SMOKE), "--llm", "mock", "--judge-llm", "none"]) == 2
    assert "judge" in capsys.readouterr().err

    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here either
    assert main(["run", str(SMOKE), "--llm", "openai"]) == 2
    assert "LLM_BASE_URL" in capsys.readouterr().err

    assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 2
    assert "not found" in capsys.readouterr().err


def test_compare_of_different_task_sets_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "taskset.yaml").write_text(
        "name: smoke\nversion: 2\ntasks:\n  - id: only\n    input: hello there\n"
    )
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run_smoke(a)
    assert main(["run", str(other), "--llm", "mock", "--out", str(b)]) == 0
    capsys.readouterr()
    assert main(["compare", str(a), str(b)]) == 2
    text = capsys.readouterr().out
    assert "COMPARE INCONCLUSIVE" in text and "task set changed" in text and "exit code 2" in text


# ----------------------------------------------------------------- the real client path
def test_openai_mode_runs_through_the_real_client_against_a_local_server(
    tmp_path: Path, fake_openai: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", fake_openai.base_url)
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    prices = tmp_path / "prices.yaml"
    prices.write_text("fake-model:\n  input_per_million: 1.0\n  output_per_million: 2.0\n")
    taskset = tmp_path / "ts.yaml"
    taskset.write_text(
        "name: live-ish\nversion: 1\ntasks:\n"
        "  - id: hi\n    input: say hi\n    scorers: [{type: contains, value: hello}]\n"
    )
    fake_openai.reply({"content": "hello there"})
    out = tmp_path / "r.json"
    code = main(
        [
            "run",
            str(taskset),
            "--llm",
            "openai",
            "--prices",
            str(prices),
            "--budget-usd",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    data = load(out)
    assert data["summary"]["passed"] == 1
    assert data["summary"]["usage"]["prompt_tokens"] == 12  # the server's own usage numbers
    assert data["summary"]["cost_usd"] == pytest.approx((12 * 1.0 + 7 * 2.0) / 1_000_000)
    assert data["config"]["model"] == "fake-model" and data["config"]["llm"] == "openai"
    assert fake_openai.headers[0]["authorization"] == "Bearer sk-test"


# ----------------------------------------------------------------- --extra-param
def _live_env(monkeypatch: pytest.MonkeyPatch, fake: FakeOpenAI) -> None:
    monkeypatch.setenv("LLM_BASE_URL", fake.base_url)
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


def _one_task(tmp_path: Path) -> Path:
    path = tmp_path / "ts.yaml"
    path.write_text(
        "name: x\nversion: 1\ntasks:\n"
        "  - id: hi\n    input: say hi\n    scorers: [{type: contains, value: hello}]\n"
    )
    return path


def test_extra_params_reach_the_server_and_are_recorded_in_the_report_config(
    tmp_path: Path, fake_openai: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_env(monkeypatch, fake_openai)
    fake_openai.reply({"content": "hello there"})
    prices = tmp_path / "prices.yaml"
    prices.write_text("fake-model:\n  input_per_million: 1.0\n  output_per_million: 2.0\n")
    out = tmp_path / "r.json"
    argv = ["run", str(_one_task(tmp_path)), "--llm", "openai", "--prices", str(prices)]
    argv += ["--extra-param", "reasoning_effort=low", "--out", str(out)]
    argv += ["--extra-param", 'thinking={"type": "enabled"}']
    assert main(argv) == 0
    body = fake_openai.requests[0]
    assert body["reasoning_effort"] == "low" and body["thinking"] == {"type": "enabled"}
    assert load(out)["config"]["extra_params"] == {
        "reasoning_effort": "low",
        "thinking": {"type": "enabled"},
    }


def test_without_extra_params_the_config_has_no_such_key(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert run_smoke(out) == 0
    assert "extra_params" not in load(out)["config"]  # keeps existing baselines valid


def test_a_mock_run_records_the_extra_params_too(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert run_smoke(out, "--extra-param", "reasoning_effort=low") == 0
    assert load(out)["config"]["extra_params"] == {"reasoning_effort": "low"}


@pytest.mark.parametrize("bad", ["noequals", "=value", "model=other"])
def test_bad_extra_params_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    code = main(["run", str(SMOKE), "--llm", "mock", "--extra-param", bad])
    assert code == 2 and capsys.readouterr().err.startswith("error:")
