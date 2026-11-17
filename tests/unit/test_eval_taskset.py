import json
from pathlib import Path
from typing import Any

import pytest

from trellis_agent.eval.taskset import (
    EvalTask,
    ScorerSpec,
    TaskSet,
    TaskSetError,
    load_taskset,
    parse_taskset,
)

DOC: dict[str, Any] = {
    "name": "demo",
    "version": 3,  # YAML users write numbers; we normalise to str
    "tasks": [
        {
            "id": "t1",
            "input": "What is 2+3?",
            "tags": ["math"],
            "scorers": [{"type": "contains", "value": "5"}, {"type": "tool_called", "name": "add"}],
        },
        {"id": "t2", "input": "Say hi", "scorers": [], "metadata": {"difficulty": "easy"}},
    ],
}

YAML_TEXT = """\
name: demo
version: 3
tasks:
  - id: t1
    input: What is 2+3?
    tags: [math]
    scorers:
      - {type: contains, value: "5"}
      - {type: tool_called, name: add}
  - id: t2
    input: Say hi
    scorers: []
    metadata: {difficulty: easy}
"""


def test_parse_builds_typed_objects() -> None:
    ts = parse_taskset(DOC)
    assert isinstance(ts, TaskSet) and (ts.name, ts.version) == ("demo", "3")
    t1, t2 = ts.tasks
    assert t1 == EvalTask(
        id="t1",
        input="What is 2+3?",
        scorers=(
            ScorerSpec("contains", {"value": "5"}),
            ScorerSpec("tool_called", {"name": "add"}),
        ),
        tags=("math",),
    )
    assert t2.metadata == {"difficulty": "easy"} and t2.expect_success is True


def test_hash_ignores_formatting_but_tracks_content(tmp_path: Path) -> None:
    yaml_path = tmp_path / "a.yaml"
    yaml_path.write_text(YAML_TEXT)
    json_path = tmp_path / "b.json"
    json_path.write_text(json.dumps(DOC, indent=7))
    assert load_taskset(yaml_path).content_hash == load_taskset(json_path).content_hash

    changed = json.loads(json.dumps(DOC))
    changed["tasks"][0]["input"] = "What is 2+4?"
    assert parse_taskset(changed).content_hash != parse_taskset(DOC).content_hash
    reordered = json.loads(json.dumps(DOC))
    reordered["tasks"].reverse()
    assert parse_taskset(reordered).content_hash != parse_taskset(DOC).content_hash
    bumped = {**DOC, "version": 4}
    assert parse_taskset(bumped).content_hash != parse_taskset(DOC).content_hash


def test_hash_is_a_full_sha256_hex_digest() -> None:
    h = parse_taskset(DOC).content_hash
    assert len(h) == 64 and set(h) <= set("0123456789abcdef")


def test_load_directory_finds_the_taskset_file(tmp_path: Path) -> None:
    (tmp_path / "taskset.yaml").write_text(YAML_TEXT)
    assert load_taskset(tmp_path).name == "demo"


def test_load_jsonl_with_header_line(tmp_path: Path) -> None:
    path = tmp_path / "set.jsonl"
    lines = [{"name": "demo", "version": "3"}, *DOC["tasks"]]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    ts = load_taskset(path)
    assert [t.id for t in ts.tasks] == ["t1", "t2"]
    assert ts.content_hash == parse_taskset(DOC).content_hash


def test_jsonl_errors_name_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"name": "x", "version": "1"}\n{oops\n')
    with pytest.raises(TaskSetError, match="line 2"):
        load_taskset(path)


def test_missing_path_and_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(TaskSetError, match="not found"):
        load_taskset(tmp_path / "nope.yaml")
    with pytest.raises(TaskSetError, match="no taskset file"):
        load_taskset(tmp_path)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "x.txt"
    path.write_text("hi")
    with pytest.raises(TaskSetError, match="unsupported"):
        load_taskset(path)


def with_task(**changes: Any) -> dict[str, Any]:
    task = {"id": "t", "input": "q", "scorers": []}
    task.update(changes)
    return {"name": "n", "version": "1", "tasks": [task]}


@pytest.mark.parametrize(
    ("doc", "fragment"),
    [
        ({"version": "1", "tasks": []}, "name"),
        ({"name": "n", "tasks": []}, "version"),
        ({"name": "n", "version": "1"}, "tasks"),
        ({"name": "n", "version": "1", "tasks": []}, "at least one task"),
        (with_task(id=""), "id"),
        (with_task(id=5), "id"),
        (with_task(input=""), "input"),
        (with_task(input=None), "input"),
        (with_task(scorers="contains"), "scorers"),
        (with_task(scorers=[{"value": "x"}]), "type"),
        (with_task(scorers=["contains"]), "scorer"),
        (with_task(tags="math"), "tags"),
        (with_task(metadata=[1]), "metadata"),
        (with_task(expect_success="yes"), "expect_success"),
        (with_task(scores=[]), "unknown"),  # typo for "scorers" must not be silently ignored
        ({"name": "n", "version": "1", "tasks": [], "taks": []}, "unknown"),
    ],
)
def test_invalid_documents_are_rejected_with_a_clear_message(
    doc: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(TaskSetError, match=fragment):
        parse_taskset(doc)


def test_duplicate_ids_rejected() -> None:
    doc = {
        "name": "n",
        "version": "1",
        "tasks": [{"id": "a", "input": "q"}, {"id": "a", "input": "r"}],
    }
    with pytest.raises(TaskSetError, match=r"duplicate.*'a'"):
        parse_taskset(doc)


def test_error_messages_identify_the_task() -> None:
    with pytest.raises(TaskSetError, match="task 'bad'"):
        parse_taskset(with_task(id="bad", input=""))


def test_known_scorer_types_can_be_enforced() -> None:
    doc = with_task(scorers=[{"type": "contanis", "value": "x"}])
    with pytest.raises(TaskSetError, match="unknown scorer type 'contanis'"):
        parse_taskset(doc, known_scorers={"contains", "regex"})
    assert parse_taskset(doc).tasks[0].scorers[0].type == "contanis"  # not enforced by default
