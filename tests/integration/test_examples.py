"""The examples are documentation that must keep working: run each one in --mock mode."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run_example(name: str) -> str:
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), "--mock"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_01_single_agent() -> None:
    out = run_example("01_single_agent.py")
    assert "Answer: I am Trellis demo assistant." in out
    assert "tokens: 10 in / 5 out" in out


def test_02_tools_agent() -> None:
    out = run_example("02_tools_agent.py")
    assert "Answer: Tokyo is 18C and sunny; 18 + 7 = 25." in out
    assert "tool calls: 3 (2 ok)" in out
    assert "error: invalid_arguments" in out  # the first call has a bad argument and is corrected


def test_03_cost_and_budget() -> None:
    out = run_example("03_cost_and_budget.py")
    assert "DEMO prices" in out  # never pass demo prices off as real ones
    assert "stop_reason: budget_exceeded" in out
    assert "spent: $1.800000 of $1.500000" in out


def test_04_planned_agent() -> None:
    out = run_example("04_planned.py")
    assert "Answer: The total price is 60." in out
    for row in ("t1 | research", "t2 | research", "t3 | compute"):
        assert row in out
    assert out.count("| done |") == 3  # every task finished
    assert "workers: researcher, calculator" in out


def test_05_memory() -> None:
    out = run_example("05_memory.py")
    assert "older messages trimmed from the window" in out
    assert "memory note: - 截止日期改到周二了" in out  # the updated value, recalled from memory
    assert "周五" not in out.split("memory note:")[1].split("answer:")[0]  # not the old one
    assert "answer: 我记得:截止日期改到周二了" in out
    assert "superseded (kept as history, never recalled): 1" in out


@pytest.mark.parametrize(
    "name",
    [
        "01_single_agent.py",
        "02_tools_agent.py",
        "03_cost_and_budget.py",
        "04_planned.py",
        "05_memory.py",
    ],
)
def test_examples_without_mock_need_credentials(name: str, tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,  # empty dir: a real .env in the project root must never be picked up here
        env={"PATH": "/usr/bin:/bin"},  # no LLM_* variables
        check=False,
    )
    assert done.returncode != 0
    assert "LLM_BASE_URL" in done.stderr
