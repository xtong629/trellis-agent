"""The experiment task sets: shape, balance, and that the committed files are what the generator
produces (so a hand edit cannot slip in unnoticed)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from experiments.scenarios import SEPARATOR, taskset_a, taskset_b

from trellis_agent.eval import load_taskset, parse_taskset, scorer_types

ROOT = Path(__file__).resolve().parents[2]
A_DIR, B_DIR = ROOT / "evals" / "exp_a_planner", ROOT / "evals" / "exp_b_memory"


def families(doc: dict) -> Counter[str]:  # type: ignore[type-arg]
    return Counter(t["metadata"]["family"] for t in doc["tasks"])


def test_experiment_a_has_thirty_tasks_in_four_families() -> None:
    doc = taskset_a()
    assert (doc["name"], len(doc["tasks"])) == ("exp-a-planner", 30)
    assert families(doc) == {"serial": 8, "fanout": 8, "flaky": 6, "single": 8}


def test_experiment_b_has_twenty_scripts_in_five_families() -> None:
    doc = taskset_b()
    assert (doc["name"], len(doc["tasks"])) == ("exp-b-memory", 20)
    assert families(doc) == {
        "allergy": 6,
        "deadline": 5,
        "refund-zh": 3,
        "fits": 3,
        "irrelevant": 3,
    }


@pytest.mark.parametrize("make", [taskset_a, taskset_b])
def test_each_experiment_has_tasks_that_should_and_should_not_benefit(make) -> None:  # type: ignore[no-untyped-def]
    doc = make()
    benefit = Counter(t["metadata"]["expected"] for t in doc["tasks"])
    assert set(benefit) == {"benefit", "no-benefit"}
    assert min(benefit.values()) >= 3  # neither side is a token gesture
    for t in doc["tasks"]:
        assert t["metadata"]["expected"] in t["tags"] and t["metadata"]["family"] in t["tags"]
        assert t["metadata"]["why"]  # the hypothesis behind each family is written down


def test_the_split_between_benefit_and_no_benefit_tasks() -> None:
    a = Counter(t["metadata"]["expected"] for t in taskset_a()["tasks"])
    b = Counter(t["metadata"]["expected"] for t in taskset_b()["tasks"])
    assert dict(a) == {"benefit": 22, "no-benefit": 8} and dict(b) == {
        "benefit": 14,
        "no-benefit": 6,
    }


@pytest.mark.parametrize("make", [taskset_a, taskset_b])
def test_only_deterministic_rule_scorers_are_used_never_a_judge(make) -> None:  # type: ignore[no-untyped-def]
    doc = make()
    ids = [t["id"] for t in doc["tasks"]]
    assert len(ids) == len(set(ids))
    for t in doc["tasks"]:
        types = {s["type"] for s in t["scorers"]}
        assert types and "llm_judge" not in types and types <= set(scorer_types())
    parse_taskset(doc, known_scorers=scorer_types())  # passes the harness's own validation


def test_experiment_b_scripts_are_multi_turn_conversations() -> None:
    for t in taskset_b()["tasks"]:
        turns = t["input"].split(SEPARATOR)
        assert len(turns) >= 2 and all(turn.strip() for turn in turns)
        assert turns[-1].strip()


def test_the_chinese_refund_scripts_are_in_chinese_and_the_final_turn_asks_for_the_code() -> None:
    zh = [t for t in taskset_b()["tasks"] if t["metadata"]["family"] == "refund-zh"]
    assert len(zh) == 3
    for t in zh:
        assert "退款单号" in t["input"] and any(
            s["type"] == "contains" and s["value"].startswith("RF-") for s in t["scorers"]
        )


def test_the_committed_files_equal_what_the_generator_produces() -> None:
    for directory, make in ((A_DIR, taskset_a), (B_DIR, taskset_b)):
        committed = load_taskset(directory)
        generated = parse_taskset(make())
        assert committed.content_hash == generated.content_hash, (
            f"{directory.name} differs from the generator"
        )
