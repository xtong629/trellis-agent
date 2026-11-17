"""Evaluation results: per-case records, run-level summary, and (de)serialisation.

Cost accounting rules (see docs/EVAL.md):
* agent and judge spend are separate fields and both roll up into ``cost_usd``;
* every figure comes from the per-call responses (``LLMResponse.cost_usd``), including the call
  that crossed a budget, and is reconciled against the shared ``CostMeter`` when there is one;
* an unknown cost (model without a price) is ``None`` and flips ``cost_complete`` to False; it is
  never silently treated as zero;
* wall-clock timing is excluded from the saved report unless explicitly requested, so that two runs
  of the same task set with a deterministic (mock) target produce byte-identical files.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import Usage
from .scorers import ScoreResult

SCHEMA_VERSION = 1
CONTENT_CAP = 2000

PASSED, FAILED, ABORTED, NOT_RUN = "passed", "failed", "aborted", "not_run"
COMPLETED = (PASSED, FAILED)


class ReportError(ValueError):
    """A report file is missing, unreadable, or not a report this version understands."""


def short_hash(digest: str) -> str:
    return digest[:12]


def cap_content(text: str) -> str:
    return text if len(text) <= CONTENT_CAP else text[:CONTENT_CAP] + "...[truncated]"


@dataclass
class CaseResult:
    task_id: str
    repeat: int
    status: str  # passed | failed | aborted (budget hit mid-case) | not_run (budget hit earlier)
    content: str = ""
    error: str | None = None
    stop_reason: str = ""
    scores: list[ScoreResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)  # the agent's tokens
    judge_usage: Usage = field(default_factory=Usage)  # LLM judges' tokens, kept apart
    agent_cost_usd: float | None = 0.0
    judge_cost_usd: float | None = 0.0
    judge_calls: int = 0
    model_requests: int = 0
    tool_calls: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.task_id}#{self.repeat}"

    @property
    def cost_usd(self) -> float | None:
        if self.agent_cost_usd is None or self.judge_cost_usd is None:
            return None
        return self.agent_cost_usd + self.judge_cost_usd

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CaseResult:
        for key in ("task_id", "repeat", "status"):
            if key not in raw:
                raise ReportError(f"case is missing {key!r}")
        if raw["status"] not in (PASSED, FAILED, ABORTED, NOT_RUN):
            raise ReportError(f"case has invalid status {raw['status']!r}")
        return cls(
            task_id=str(raw["task_id"]),
            repeat=int(raw["repeat"]),
            status=str(raw["status"]),
            content=str(raw.get("content", "")),
            error=raw.get("error"),
            stop_reason=str(raw.get("stop_reason", "")),
            scores=[ScoreResult.from_dict(x) for x in raw.get("scores", [])],
            usage=Usage.from_dict(raw.get("usage", {})),
            judge_usage=Usage.from_dict(raw.get("judge_usage", {})),
            agent_cost_usd=raw.get("agent_cost_usd"),
            judge_cost_usd=raw.get("judge_cost_usd"),
            judge_calls=int(raw.get("judge_calls", 0)),
            model_requests=int(raw.get("model_requests", 0)),
            tool_calls=list(raw.get("tool_calls", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repeat": self.repeat,
            "status": self.status,
            "content": self.content,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "scores": [s.to_dict() for s in self.scores],
            "usage": self.usage.to_dict(),
            "judge_usage": self.judge_usage.to_dict(),
            "agent_cost_usd": self.agent_cost_usd,
            "judge_cost_usd": self.judge_cost_usd,
            "judge_calls": self.judge_calls,
            "model_requests": self.model_requests,
            "tool_calls": list(self.tool_calls),
        }


def _sum_usage(usages: list[Usage]) -> Usage:
    total = Usage()
    for usage in usages:
        total = total + usage
    return total


@dataclass
class EvalReport:
    taskset_name: str
    taskset_version: str
    taskset_hash: str
    config: dict[str, Any]
    cases: list[CaseResult]
    meter_total_usd: float | None = None
    aborted_by_budget: bool = False
    timing: dict[str, float] | None = None

    def summary(self) -> dict[str, Any]:
        cases = self.cases
        done = [c for c in cases if c.status in COMPLETED]
        passed = sum(1 for c in done if c.status == PASSED)
        counts = {s: sum(1 for c in cases if c.status == s) for s in (FAILED, ABORTED, NOT_RUN)}

        repeats = sorted({c.repeat for c in cases})
        by_repeat: list[float | None] = []
        for r in repeats:
            in_r = [c for c in done if c.repeat == r]
            by_repeat.append(sum(c.status == PASSED for c in in_r) / len(in_r) if in_r else None)
        rates = [x for x in by_repeat if x is not None]

        tasks: list[dict[str, Any]] = []
        for task_id in dict.fromkeys(c.task_id for c in cases):
            mine = [c for c in done if c.task_id == task_id]
            ok = sum(c.status == PASSED for c in mine)
            tasks.append(
                {
                    "task_id": task_id,
                    "repeats_completed": len(mine),
                    "passed": ok,
                    "pass_rate": ok / len(mine) if mine else None,
                }
            )

        ran = [c for c in cases if c.status != NOT_RUN]
        agent_known = [c.agent_cost_usd for c in ran if c.agent_cost_usd is not None]
        judge_known = [c.judge_cost_usd for c in ran if c.judge_cost_usd is not None]
        priced = bool(agent_known) or any(
            c.judge_calls > 0 and c.judge_cost_usd is not None for c in ran
        )
        agent_cost = math.fsum(agent_known)
        judge_cost = math.fsum(judge_known)
        complete = all(c.agent_cost_usd is not None and c.judge_cost_usd is not None for c in ran)
        total_cost = (agent_cost + judge_cost) if priced else None
        cost_per_success = total_cost / passed if total_cost is not None and passed else None
        usage = _sum_usage([c.usage for c in ran])
        judge_usage = _sum_usage([c.judge_usage for c in ran])
        return {
            "planned_jobs": len(cases),
            "completed_jobs": len(done),
            "aborted_jobs": counts[ABORTED],
            "not_run_jobs": counts[NOT_RUN],
            "passed": passed,
            "failed": counts[FAILED],
            "pass_rate": passed / len(done) if done else None,
            "pass_rate_by_repeat": by_repeat,
            "pass_rate_mean": statistics.fmean(rates) if rates else None,
            "pass_rate_std": statistics.stdev(rates) if len(rates) > 1 else None,
            "tasks": tasks,
            "usage": usage.to_dict(),
            "judge_usage": judge_usage.to_dict(),
            "estimated_usage": usage.estimated or judge_usage.estimated,
            "model_requests": sum(c.model_requests for c in ran),
            "agent_cost_usd": agent_cost if priced else None,
            "judge_cost_usd": judge_cost if priced else None,
            "cost_usd": total_cost,
            "cost_complete": complete,
            "cost_per_success": cost_per_success,
            "meter_total_usd": self.meter_total_usd,
            "meter_discrepancy_usd": (
                None
                if self.meter_total_usd is None or total_cost is None
                # rounded: both sums are exact, but they may differ in the last bit
                else round(self.meter_total_usd - total_cost, 12) + 0.0
            ),
            "aborted_by_budget": self.aborted_by_budget,
        }

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "taskset": {
                "name": self.taskset_name,
                "version": self.taskset_version,
                "hash": self.taskset_hash,
            },
            "config": self.config,
            "summary": self.summary(),
            "cases": [c.to_dict() for c in self.cases],
        }
        if self.timing is not None:
            data["timing_ms"] = self.timing
        return data

    def save(self, path: str | Path) -> None:
        """Write deterministic JSON (sorted keys, fixed indent, real unicode, trailing newline)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)
        target.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalReport:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ReportError(
                f"unsupported schema_version {data.get('schema_version')!r} "
                f"(this version reads {SCHEMA_VERSION})"
            )
        for key in ("taskset", "cases"):
            if key not in data:
                raise ReportError(f"report is missing {key!r}")
        taskset = data["taskset"]
        for key in ("name", "version", "hash"):
            if key not in taskset:
                raise ReportError(f"report taskset is missing {key!r}")
        summary = data.get("summary", {})
        return cls(
            taskset_name=str(taskset["name"]),
            taskset_version=str(taskset["version"]),
            taskset_hash=str(taskset["hash"]),
            config=dict(data.get("config", {})),
            cases=[CaseResult.from_dict(c) for c in data["cases"]],
            meter_total_usd=summary.get("meter_total_usd"),
            aborted_by_budget=bool(summary.get("aborted_by_budget", False)),
            timing=data.get("timing_ms"),
        )

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        file = Path(path)
        if not file.is_file():
            raise ReportError(f"report not found: {file}")
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(f"{file} is not valid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise ReportError(f"{file}: top level must be an object")
        return cls.from_dict(data)

    def render_markdown(self) -> str:
        s = self.summary()

        def money(value: float | None) -> str:
            return "unknown (no price)" if value is None else f"${value:.6f}"

        def tokens(u: Mapping[str, Any]) -> str:
            return f"{u['prompt_tokens']} in / {u['completion_tokens']} out"

        rate = "n/a" if s["pass_rate"] is None else f"{s['pass_rate']:.1%}"
        per_success = "n/a" if s["cost_per_success"] is None else money(s["cost_per_success"])
        extra = "".join(
            f", {s[key]} {label}"
            for key, label in (("aborted_jobs", "aborted"), ("not_run_jobs", "not run"))
            if s[key]
        )
        lines = [
            f"# Eval report: {self.taskset_name} v{self.taskset_version}",
            "",
            f"Task set hash: `{short_hash(self.taskset_hash)}`",
            "",
        ]
        if self.aborted_by_budget:
            lines += ["> **ABORTED BY BUDGET**: the results below are partial.", ""]
        lines += ["| Metric | Value |", "| --- | --- |"]
        if s["pass_rate"] is None:
            lines.append(f"| Pass rate | n/a ({s['completed_jobs']} completed{extra}) |")
        else:
            lines.append(
                f"| Pass rate | {rate} ({s['passed']}/{s['completed_jobs']} completed{extra}) |"
            )
        if s["pass_rate_std"] is not None:
            lines.append(
                f"| Pass rate across repeats | "
                f"{s['pass_rate_mean']:.1%} +/- {s['pass_rate_std']:.1%} |"
            )
        lines += [
            f"| Agent tokens | {tokens(s['usage'])} |",
            f"| Judge tokens (separate) | {tokens(s['judge_usage'])} |",
            f"| Agent cost | {money(s['agent_cost_usd'])} |",
            f"| Judge cost (separate) | {money(s['judge_cost_usd'])} |",
            f"| Total cost | {money(s['cost_usd'])} |",
            f"| Cost per success | {per_success} |",
            "",
        ]
        if not s["cost_complete"]:
            lines += [
                "> WARNING: some calls have unknown cost (model has no price); "
                "totals are lower bounds.",
                "",
            ]
        if s["estimated_usage"]:
            lines += [
                "> NOTE: token counts are estimated for some calls (provider reported no usage).",
                "",
            ]
        lines += [
            "## Cases",
            "",
            "| Task | Repeat | Status | Failed checks |",
            "| --- | --- | --- | --- |",
        ]
        for c in self.cases:
            failed = "; ".join(f"{x.name}: {x.reason}" for x in c.scores if not x.passed)
            failed = failed.replace("|", "\\|")[:200]
            lines.append(f"| {c.task_id} | {c.repeat} | {c.status} | {failed} |")
        return "\n".join(lines) + "\n"
