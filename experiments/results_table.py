"""Markdown result tables computed from the committed reports in evals/reports/.

    python -m experiments.results_table

Every number is derived from the report JSON named next to it. Success rates are the mean and the
sample standard deviation (n-1) of the per-repeat pass rate over the repeats."""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"a": "exp_a_planner", "b": "exp_b_memory"}
SYSTEMS = {
    "a_single": ("a", "single agent"), "a_planned": ("a", "planner"),
    "b_window": ("b", "sliding window"), "b_memory": ("b", "tiered memory"),
    "b_full": ("b", "full history (reference)"),
}  # fmt: skip


def _meta(which: str) -> dict[str, tuple[str, str]]:
    doc = yaml.safe_load((ROOT / "evals" / GROUPS[which] / "taskset.yaml").read_text())
    return {t["id"]: (t["metadata"]["family"], t["metadata"]["expected"]) for t in doc["tasks"]}


def _rate(cases: list[dict[str, Any]], repeats: int) -> tuple[float, float]:
    rates = []
    for k in range(repeats):
        mine = [c for c in cases if c["repeat"] == k]
        rates.append(sum(c["status"] == "passed" for c in mine) / len(mine))
    return st.mean(rates), st.stdev(rates)


def load_metrics(system: str) -> dict[str, Any]:
    which = SYSTEMS[system][0]
    path = ROOT / "evals" / "reports" / f"exp_{system}.json"
    report = json.loads(path.read_text())
    cases, meta = report["cases"], _meta(which)
    repeats = len({c["repeat"] for c in cases})
    passed = sum(c["status"] == "passed" for c in cases)
    tokens = sum(c["usage"]["prompt_tokens"] + c["usage"]["completion_tokens"] for c in cases)
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        families[meta[c["task_id"]][0]].append(c)
    by_expected = {
        e: [c for c in cases if meta[c["task_id"]][1] == e] for e in ("benefit", "no-benefit")
    }
    cost = report["summary"]["cost"]
    return {
        "file": f"evals/reports/exp_{system}.json",
        "runs": len(cases), "repeats": repeats, "tasks": len(cases) // repeats,
        "pass": _rate(cases, repeats),
        "benefit": _rate(by_expected["benefit"], repeats),
        "no_benefit": _rate(by_expected["no-benefit"], repeats),
        "families": {f: _rate(v, repeats) for f, v in families.items()},
        "cost": cost, "cost_per_success": cost / passed, "currency": report["summary"]["currency"],
        "tokens": tokens, "calls": sum(c["model_requests"] for c in cases),
        "taskset_hash": report["taskset"]["hash"],
    }  # fmt: skip


def _pm(value: tuple[float, float]) -> str:
    return f"{value[0]:.3f}±{value[1]:.3f}"


def render_tables() -> str:
    metrics = {s: load_metrics(s) for s in SYSTEMS}
    lines = [
        "| system | report | tasks x repeats | all | should benefit | should not benefit "
        "| cost (CNY, list price) | cost per success | tokens | model calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system, (_, label) in SYSTEMS.items():
        m = metrics[system]
        lines.append(
            f"| {label} | `exp_{system}.json` | {m['tasks']} x {m['repeats']} | {_pm(m['pass'])} "
            f"| {_pm(m['benefit'])} | {_pm(m['no_benefit'])} | {m['cost']:.4f} "
            f"| {m['cost_per_success']:.5f} | {m['tokens']:,} | {m['calls']:,} |"
        )
    for which, names in (
        ("a", ["a_single", "a_planned"]),
        ("b", ["b_window", "b_memory", "b_full"]),
    ):
        families = list(metrics[names[0]]["families"])
        header = "| family (tasks) | " + " | ".join(SYSTEMS[n][1] for n in names) + " |"
        counts = {f: sum(1 for t in _meta(which).values() if t[0] == f) for f in families}
        lines += ["", header, "| --- |" + " --- |" * len(names)]
        for f in families:
            cells = " | ".join(_pm(metrics[n]["families"][f]) for n in names)
            lines.append(f"| {f} ({counts[f]}) | {cells} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render_tables(), end="")
