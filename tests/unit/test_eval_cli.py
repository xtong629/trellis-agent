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


@pytest.fixture(autouse=True)
def _isolated_cwd(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI reads ./.env in real mode: never let a developer's real .env leak into a test."""
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    for name in ("BUDGET", "BUDGET_USD"):
        monkeypatch.delenv(name, raising=False)


FAKE_PRICES = (
    "models:\n  fake-model:\n    currency: USD\n    per: 1M tokens\n"
    "    input: 1.0\n    output: 2.0\n"
)


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
    assert summary["agent_cost"] > 0 and summary["judge_cost"] > 0
    assert summary["cost"] == pytest.approx(summary["agent_cost"] + summary["judge_cost"])
    assert summary["judge_usage"]["prompt_tokens"] > 0
    assert summary["meter_discrepancy"] == pytest.approx(0.0)  # report cost == meter cost
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
    code = run_smoke(out, "--budget", "0.0003 DEMO", "--concurrency", "1")
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
        [
            "run",
            str(SMOKE),
            "--llm",
            "mock",
            "--budget",
            "1 DEMO",
            "--out",
            str(tmp_path / "r.json"),
        ]
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
    assert main(["run", str(SMOKE), "--llm", "openai", "--budget", "1 USD"]) == 2
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
    prices.write_text(FAKE_PRICES)
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
    assert data["summary"]["cost"] == pytest.approx((12 * 1.0 + 7 * 2.0) / 1_000_000)
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
    prices.write_text(FAKE_PRICES)
    out = tmp_path / "r.json"
    argv = ["run", str(_one_task(tmp_path)), "--llm", "openai", "--prices", str(prices)]
    argv += ["--budget", "1 USD", "--extra-param", "reasoning_effort=low", "--out", str(out)]
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


# --- budgets are amounts in a currency
def test_a_budget_without_a_currency_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(SMOKE), "--llm", "mock", "--prices", str(PRICES), "--budget", "5"]) == 2
    assert "currency" in capsys.readouterr().err


def test_a_budget_in_another_currency_than_the_prices_is_refused_up_front(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["run", str(SMOKE), "--llm", "mock", "--prices", str(PRICES), "--budget", "5 USD"]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "DEMO" in err and "USD" in err and "fx" in err


def test_a_declared_fx_rate_makes_a_cross_currency_budget_work_and_is_recorded(
    tmp_path: Path,
) -> None:
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        PRICES.read_text(encoding="utf-8") + "fx:\n  - from: DEMO\n    to: USD\n    rate: 0.5\n"
        "    source: https://example.test/fx\n    as_of: 2026-11-17\n"
    )
    argv = ["run", str(SMOKE), "--llm", "mock", "--prices", str(prices), "--budget", "100 USD"]
    out = tmp_path / "r.json"
    assert main([*argv, "--out", str(out)]) == 0
    summary = load(out)["summary"]
    assert summary["currency"] == "USD"  # the budget's currency: DEMO costs were converted into it
    (fx,) = summary["fx_applied"]
    assert (fx["from_currency"], fx["to_currency"], fx["rate"]) == ("DEMO", "USD", 0.5)
    assert summary["cost_by_currency"]["DEMO"] > 0  # the native amount is kept apart
    assert summary["cost"] == pytest.approx(summary["cost_by_currency"]["DEMO"] * 0.5)


def test_the_old_usd_flag_and_env_var_still_work_as_deprecated_aliases(
    tmp_path: Path, fake_openai: FakeOpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _live_env(monkeypatch, fake_openai)
    monkeypatch.setenv("BUDGET_USD", "1")
    monkeypatch.delenv("BUDGET", raising=False)
    prices = tmp_path / "prices.yaml"
    prices.write_text(FAKE_PRICES)
    fake_openai.reply({"content": "hello there"})
    argv = ["run", str(_one_task(tmp_path)), "--llm", "openai", "--prices", str(prices)]
    out = tmp_path / "r.json"
    assert main([*argv, "--out", str(out)]) == 0  # budget came from BUDGET_USD = 1 USD
    assert load(out)["summary"]["currency"] == "USD"


def test_the_new_budget_env_var_takes_precedence_and_real_runs_need_some_budget(
    tmp_path: Path,
    fake_openai: FakeOpenAI,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _live_env(monkeypatch, fake_openai)
    prices = tmp_path / "prices.yaml"
    prices.write_text(FAKE_PRICES)
    argv = ["run", str(_one_task(tmp_path)), "--llm", "openai", "--prices", str(prices)]
    monkeypatch.delenv("BUDGET", raising=False)
    monkeypatch.delenv("BUDGET_USD", raising=False)
    assert main(argv) == 2  # no implicit default budget: a currency must be chosen explicitly
    assert "BUDGET" in capsys.readouterr().err
    monkeypatch.setenv("BUDGET_USD", "0.0000001")
    monkeypatch.setenv("BUDGET", "5 USD")  # wins over the alias
    fake_openai.reply({"content": "hello there"})
    assert main([*argv, "--out", str(tmp_path / "r.json")]) == 0


# ----------------------------------------------------------------- the price files in the repo
def test_a_report_records_which_prices_produced_its_numbers(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    assert run_smoke(out, "--md", str(tmp_path / "r.md")) == 0
    data = load(out)
    assert data["summary"]["currency"] == "DEMO" and data["summary"]["fx_applied"] == []
    assert data["config"]["price_table"]["mock"]["currency"] == "DEMO"
    assert set(data["config"]["price_table"]) == {"mock", "mock-judge"}
    md = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## Prices used" in md and "Currency | DEMO" in md and "$" not in md


def test_the_demo_prices_are_in_a_made_up_currency_with_the_banner() -> None:
    from trellis_agent.cost import PriceTable

    text = PRICES.read_text(encoding="utf-8")
    assert "DEMO PRICES ONLY" in text and "仅为示例,非真实价格" in text
    table = PriceTable.from_yaml(PRICES)
    assert {table.get(m).currency for m in ("mock", "mock-judge")} == {"DEMO"}  # type: ignore[union-attr]


def test_the_glm_prices_are_the_official_cny_list_prices_with_provenance() -> None:
    from trellis_agent.cost import PriceTable

    path = ROOT / "evals" / "prices.glm.yaml"
    price = PriceTable.from_yaml(path).get("glm-5.3-flash")
    assert price is not None and price.currency == "CNY" and price.per == "1M tokens"
    assert (price.input_per_million, price.output_per_million, price.cached_input_per_million) == (
        0.8,
        2.8,
        0.23,
    )
    assert price.source == "https://docs.bigmodel.cn/cn/guide/start/pricing.md"
    assert price.checked_on == "2026-11-17"
    assert "resource package" in (price.note or "") and "not actual spend" in (price.note or "")
    banner = path.read_text(encoding="utf-8")
    assert "NOT actual spend" in banner and "不是实际支出" in banner
    assert not {"USD", "0.15", "0.50", "0.03"} & set(
        path.read_text(encoding="utf-8").split()
    )  # old USD figures are gone
