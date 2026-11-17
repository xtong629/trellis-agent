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
    assert (doc["name"], doc["version"], len(doc["tasks"])) == ("exp-a-planner", "2", 30)
    assert families(doc) == {"serial": 8, "fanout": 8, "flaky": 6, "single": 8}


def test_experiment_b_has_twenty_scripts_in_five_families() -> None:
    doc = taskset_b()
    assert (doc["name"], doc["version"], len(doc["tasks"])) == ("exp-b-memory", "2", 20)
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


def test_the_task_sets_are_frozen_at_v2_and_v1_is_kept_on_record() -> None:
    """Editing a task changes its hash. After the freeze, only bug fixes are allowed and each one
    must update experiments/FROZEN.json in the same commit (and trigger a full re-run)."""
    import json

    frozen = json.loads((ROOT / "experiments" / "FROZEN.json").read_text())
    assert frozen["v1"]["exp_a_planner"]["content_hash"].startswith("25c438fe84a9")
    assert frozen["v1"]["exp_b_memory"]["content_hash"].startswith("76c0056431fc")
    for directory in ("exp_a_planner", "exp_b_memory"):
        entry = frozen["v2"][directory]
        loaded = load_taskset(ROOT / "evals" / directory)
        assert loaded.content_hash == entry["content_hash"], f"{directory} changed after the freeze"
        assert (loaded.version, len(loaded.tasks)) == ("2", entry["tasks"])
        assert entry["content_hash"] != frozen["v1"][directory]["content_hash"]


def test_numeric_answers_are_marked_and_the_prompt_asks_for_the_marker() -> None:
    tasks = [*taskset_a()["tasks"], *[t for t in taskset_b()["tasks"] if "irrelevant" in t["id"]]]
    assert len(tasks) == 33
    for t in tasks:
        number = next(s for s in t["scorers"] if s["type"] == "number_match")
        assert number["which"] == "marked", t["id"]
        assert t["input"].rstrip().endswith("ANSWER: <number>'."), t["id"]


def test_the_memory_window_follows_the_stated_rule() -> None:
    from experiments.targets import WINDOW_TOKENS
    from experiments.window import conversation_tokens, derive_window

    doc = taskset_b()
    window = derive_window(doc)
    assert window == WINDOW_TOKENS  # the value in use is the derived one, not a tuned one
    for t in doc["tasks"]:
        size = conversation_tokens(t["input"])
        if t["metadata"]["family"] == "fits":
            assert size <= window, t["id"]
        else:
            assert size > window, t["id"]  # every other script overflows the window
    assert window % 10 == 0
