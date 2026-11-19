"""Result tables and pre-registered decisions for the v0.2 experiments, computed only from the
committed reports in evals/reports/ (and the frozen task sets in evals/).

    python -m experiments.v02_results c1|c2|c4

Success rates are the mean and sample standard deviation (n-1) of the per-repeat pass rate.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

import yaml

from experiments.c2_tasks import CELLS

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evals" / "reports"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((REPORTS / name).read_text())
    return data


def _meta(set_dir: str) -> dict[str, dict[str, Any]]:
    doc = yaml.safe_load((ROOT / "evals" / set_dir / "taskset.yaml").read_text())
    return {t["id"]: t["metadata"] for t in doc["tasks"]}


def per_repeat(cases: list[dict[str, Any]]) -> list[float]:
    repeats = sorted({c["repeat"] for c in cases})
    out = []
    for k in repeats:
        mine = [c for c in cases if c["repeat"] == k]
        out.append(sum(c["status"] == "passed" for c in mine) / len(mine))
    return out


def ms(values: list[float]) -> str:
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):.3f}±{sd:.3f}"


def _tokens(cases: list[dict[str, Any]]) -> int:
    return sum(c["usage"]["prompt_tokens"] + c["usage"]["completion_tokens"] for c in cases)


# ------------------------------------------------------------------ C1
def c1() -> dict[str, Any]:
    meta = _meta("exp_c1_reflect")
    out: dict[str, Any] = {"systems": {}}
    for system in ("baseline", "reflect"):
        report = _load(f"exp_c1_{system}.json")
        cases = report["cases"]
        benefit = [c for c in cases if meta[c["task_id"]]["expected"] == "benefit"]
        fams = sorted({m["family"] for m in meta.values()})
        passed = sum(c["status"] == "passed" for c in cases)
        cost = report["summary"]["cost"]
        out["systems"][system] = {
            "runs": len(cases),
            "all": per_repeat(cases),
            "benefit": per_repeat(benefit),
            "families": {f: per_repeat([c for c in cases if meta[c["task_id"]]["family"] == f])
                         for f in fams},
            "cost_cny": cost, "tokens": _tokens(cases),
            "model_calls": sum(c["model_requests"] for c in cases),
            "cost_per_success": cost / passed if passed else None,
        }  # fmt: skip
    base, refl = out["systems"]["baseline"], out["systems"]["reflect"]
    diffs = [r - b for r, b in zip(refl["benefit"], base["benefit"], strict=True)]
    attempts = _load("exp_c1_reflect_attempts.json")["totals"]
    band_ok = 0.3 <= st.mean(base["benefit"]) <= 0.7
    mean_diff = st.mean(diffs)
    if not band_ok:
        decision = "not run as designed (baseline outside [0.3, 0.7]); H1 not decided"
    elif (
        mean_diff >= 0.10 and sum(d > 0 for d in diffs) >= 4 and attempts["harm"] < attempts["fix"]
    ):
        decision = "H1 supported"
    elif mean_diff < 0.05:
        decision = "H1 not supported"
    else:
        decision = "inconclusive"
    easy_drop = st.mean(base["families"]["easy"]) - st.mean(refl["families"]["easy"])
    out |= {"benefit_diff_per_repeat": diffs, "benefit_diff_mean": mean_diff,
            "baseline_benefit_mean": st.mean(base["benefit"]), "band_ok": band_ok,
            "attempts": attempts, "easy_drop": easy_drop, "easy_harm_check_ok": easy_drop <= 0.05,
            "decision": decision}  # fmt: skip
    return out


def c1_markdown(r: dict[str, Any]) -> str:
    fams = list(r["systems"]["baseline"]["families"])
    head = (
        "| system | all | benefit | "
        + " | ".join(fams)
        + " | cost (CNY) | tokens | calls | cost per success |"
    )
    lines = [head, "|" + " --- |" * (len(fams) + 7)]
    for name, s in r["systems"].items():
        cells = [ms(s["all"]), ms(s["benefit"]), *(ms(s["families"][f]) for f in fams)]
        cps = f"{s['cost_per_success']:.5f}" if s["cost_per_success"] else "n/a"
        money = f"{s['cost_cny']:.4f} | {s['tokens']:,} | {s['model_calls']:,} | {cps}"
        lines.append(f"| c1_{name} | " + " | ".join(cells) + f" | {money} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ C2
def c2() -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for cell in CELLS:
        row: dict[str, Any] = {}
        for system in ("single", "planned"):
            report = _load(f"exp_c2_{system}_{cell.label}.json")
            cases, timing = report["cases"], report["timing_ms"]
            secs = {f"{c['task_id']}#{c['repeat']}": timing[f"{c['task_id']}#{c['repeat']}"] / 1000
                    for c in cases}  # fmt: skip
            repeats = sorted({c["repeat"] for c in cases})
            rep_means = [
                st.mean(v for k, v in secs.items() if k.endswith(f"#{r}")) for r in repeats
            ]
            row[system] = {
                "runs": len(cases), "passed": sum(c["status"] == "passed" for c in cases),
                "median_s": st.median(secs.values()), "mean_s": st.mean(secs.values()),
                "repeat_means_s": rep_means, "model_calls": sum(c["model_requests"] for c in cases),
                "tokens": _tokens(cases), "cost_cny": report["summary"]["cost"],
            }  # fmt: skip
        s, p = row["single"], row["planned"]
        ratio = p["median_s"] / s["median_s"]
        lower = sum(
            pm < sm for pm, sm in zip(p["repeat_means_s"], s["repeat_means_s"], strict=True)
        )
        need = 2 if len(s["repeat_means_s"]) == 3 else 4
        wins = ratio < 0.9 and lower >= need
        row |= {"median_ratio": ratio, "planner_lower_repeats": lower, "planner_wins": wins,
                "faults": cell.fault_rate > 0}  # fmt: skip
        cells[cell.label] = row
    wins = [c.latency_s for c in CELLS if not c.fault_rate and cells[c.label]["planner_wins"]]
    decision = (f"H2 supported (smallest winning latency {min(wins):g} s)" if wins
                else "H2 not supported")  # fmt: skip
    return {"cells": cells, "decision": decision}


def c2_markdown(r: dict[str, Any]) -> str:
    head = ["cell", "system", "runs", "passed", "median s", "mean s", "per-repeat mean s",
            "model calls", "tokens", "cost (CNY)"]  # fmt: skip
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for label, row in r["cells"].items():
        for system in ("single", "planned"):
            s = row[system]
            reps = ", ".join(f"{v:.1f}" for v in s["repeat_means_s"])
            cols = [label, system, s["runs"], s["passed"], f"{s['median_s']:.2f}",
                    f"{s['mean_s']:.2f}", reps, s["model_calls"], f"{s['tokens']:,}",
                    f"{s['cost_cny']:.4f}"]  # fmt: skip
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    lines += ["", "| cell | median ratio planner/single | planner lower in repeats | wins |",
              "| --- | --- | --- | --- |"]  # fmt: skip
    for label, row in r["cells"].items():
        lines.append(f"| {label} | {row['median_ratio']:.2f} | {row['planner_lower_repeats']} | "
                     f"{'yes' if row['planner_wins'] else 'no'} |")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["c1", "c2"])
    args = parser.parse_args()
    if args.experiment == "c1":
        result = c1()
        print(c1_markdown(result))
    else:
        result = c2()
        print(c2_markdown(result))
    print(json.dumps({k: v for k, v in result.items() if k not in ("systems", "cells")}, indent=1))


if __name__ == "__main__":
    main()
