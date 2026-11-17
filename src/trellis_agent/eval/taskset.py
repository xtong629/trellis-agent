"""Versioned, content-hashed evaluation task sets (YAML / JSON / JSONL).

The hash is computed from the *parsed* content, so reformatting a file does not change it but
editing any task does. Reports record the hash, so a comparison against a baseline can detect that
the questions changed ("the tests were edited, not the system").
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_TOP_KEYS = {"name", "version", "tasks"}
_TASK_KEYS = {"id", "input", "scorers", "tags", "metadata", "expect_success"}
_DIR_CANDIDATES = ("taskset.yaml", "taskset.yml", "taskset.json", "taskset.jsonl")


class TaskSetError(ValueError):
    """The task set file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class ScorerSpec:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.params}


@dataclass(frozen=True)
class EvalTask:
    id: str
    input: str
    scorers: tuple[ScorerSpec, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    expect_success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "scorers": [s.to_dict() for s in self.scorers],
            "tags": list(self.tags),
            "metadata": self.metadata,
            "expect_success": self.expect_success,
        }


@dataclass(frozen=True)
class TaskSet:
    name: str
    version: str
    tasks: tuple[EvalTask, ...]
    content_hash: str


def _hash(name: str, version: str, tasks: tuple[EvalTask, ...]) -> str:
    canonical = json.dumps(
        {"name": name, "version": version, "tasks": [t.to_dict() for t in tasks]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_scorer(raw: Any, label: str, known: Collection[str] | None) -> ScorerSpec:
    if not isinstance(raw, Mapping):
        raise TaskSetError(f"{label}: each scorer must be a mapping, got {type(raw).__name__}")
    params = dict(raw)
    kind = params.pop("type", None)
    if not isinstance(kind, str) or not kind:
        raise TaskSetError(f"{label}: scorer is missing its 'type'")
    if known is not None and kind not in known:
        raise TaskSetError(
            f"{label}: unknown scorer type {kind!r} (known: {', '.join(sorted(known))})"
        )
    return ScorerSpec(kind, params)


def _parse_task(raw: Any, index: int, known: Collection[str] | None) -> EvalTask:
    if not isinstance(raw, Mapping):
        raise TaskSetError(f"task #{index}: must be a mapping, got {type(raw).__name__}")
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise TaskSetError(f"task #{index}: 'id' must be a non-empty string")
    label = f"task {task_id!r}"
    unknown = sorted(set(raw) - _TASK_KEYS)
    if unknown:
        raise TaskSetError(f"{label}: unknown field(s) {unknown}; allowed: {sorted(_TASK_KEYS)}")
    text = raw.get("input")
    if not isinstance(text, str) or not text.strip():
        raise TaskSetError(f"{label}: 'input' must be a non-empty string")
    scorers_raw = raw.get("scorers", [])
    if not isinstance(scorers_raw, list):
        raise TaskSetError(f"{label}: 'scorers' must be a list")
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise TaskSetError(f"{label}: 'tags' must be a list of strings")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TaskSetError(f"{label}: 'metadata' must be a mapping")
    expect_success = raw.get("expect_success", True)
    if not isinstance(expect_success, bool):
        raise TaskSetError(f"{label}: 'expect_success' must be true or false")
    return EvalTask(
        id=task_id,
        input=text,
        scorers=tuple(_parse_scorer(s, label, known) for s in scorers_raw),
        tags=tuple(tags),
        metadata=dict(metadata),
        expect_success=expect_success,
    )


def parse_taskset(
    data: Mapping[str, Any], *, known_scorers: Collection[str] | None = None
) -> TaskSet:
    """Validate a parsed document. ``known_scorers`` (if given) rejects unknown scorer types."""
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise TaskSetError(f"unknown top-level field(s) {unknown}; allowed: {sorted(_TOP_KEYS)}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise TaskSetError("task set needs a non-empty 'name'")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, str | int | float) or version == "":
        raise TaskSetError("task set needs a 'version' (string or number)")
    raw_tasks = data.get("tasks")
    if not isinstance(raw_tasks, list):
        raise TaskSetError("task set needs a 'tasks' list")
    if not raw_tasks:
        raise TaskSetError("task set needs at least one task")
    tasks = tuple(_parse_task(raw, i + 1, known_scorers) for i, raw in enumerate(raw_tasks))
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise TaskSetError(f"duplicate task id {task.id!r}")
        seen.add(task.id)
    return TaskSet(name, str(version), tasks, _hash(name, str(version), tasks))


def _read_jsonl(path: Path) -> dict[str, Any]:
    header: dict[str, Any] | None = None
    tasks: list[Any] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TaskSetError(f"{path}: invalid JSON on line {number}: {exc.msg}") from exc
        if header is None:
            if not isinstance(item, dict):
                raise TaskSetError(f"{path}: line {number} must be a header object")
            header = item
        else:
            tasks.append(item)
    if header is None:
        raise TaskSetError(f"{path}: empty file; expected a header line with name and version")
    return {**header, "tasks": tasks}


def load_taskset(path: str | Path, *, known_scorers: Collection[str] | None = None) -> TaskSet:
    """Load ``.yaml/.yml/.json`` (one document), ``.jsonl`` (header line, then one task per line),
    or a directory containing ``taskset.<ext>``."""
    file = Path(path)
    if not file.exists():
        raise TaskSetError(f"task set not found: {file}")
    if file.is_dir():
        found = next((file / n for n in _DIR_CANDIDATES if (file / n).is_file()), None)
        if found is None:
            raise TaskSetError(
                f"no taskset file in {file} (looked for {', '.join(_DIR_CANDIDATES)})"
            )
        file = found
    suffix = file.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(file.read_text(encoding="utf-8"))
        elif suffix == ".json":
            data = json.loads(file.read_text(encoding="utf-8"))
        elif suffix == ".jsonl":
            data = _read_jsonl(file)
        else:
            raise TaskSetError(f"unsupported task set format {suffix!r}: {file}")
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise TaskSetError(f"{file}: cannot parse: {exc}") from exc
    if not isinstance(data, Mapping):
        raise TaskSetError(f"{file}: top level must be a mapping")
    return parse_taskset(data, known_scorers=known_scorers)
