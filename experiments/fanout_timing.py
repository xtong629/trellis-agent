"""Supplementary wall-clock measurement of experiment A's fanout family (not a main result).

    python -m experiments.fanout_timing --out evals/reports/exp_a_fanout_timing.json

Runs only the 8 fanout tasks of the frozen v2 task set A, for the single agent and the planner,
3 repeats each, one task at a time (concurrency 1), alternating the two systems repeat by repeat
so that drift in API latency affects both. The planner keeps its own default of up to 4 parallel
workers inside a task: that internal parallelism is part of the system being timed. No task or
scoring rule is changed. The tasks are taken unmodified from the frozen set (a test checks this)."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from experiments.full_run import stop_reason
from experiments.pilot import HttpObserver, summarize_http
from experiments.scenarios import taskset_a
from experiments.targets import a_planned, a_single
from trellis_agent.cost import CostMeter
from trellis_agent.env import load_dotenv
from trellis_agent.eval import EvalRunner, parse_taskset, scorer_types
from trellis_agent.eval.cli import _prices
from trellis_agent.llm.openai_compat import OpenAICompatClient

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {"a_single": a_single, "a_planned": a_planned}
EXIT_STOPPED = 3


def fanout_doc() -> dict[str, Any]:
    doc = taskset_a()
    tasks = [t for t in doc["tasks"] if t["metadata"]["family"] == "fanout"]
    return {**doc, "name": "exp-a-fanout-timing", "tasks": tasks}


def summarize_times(rows: list[dict[str, Any]]) -> dict[str, Any]:
    secs = [r["wall_s"] for r in rows]
    return {
        "runs": len(rows),
        "mean_s": round(st.mean(secs), 2),
        "median_s": round(st.median(secs), 2),
        "min_s": min(secs),
        "max_s": max(secs),
        "sd_s": round(st.stdev(secs), 2) if len(secs) > 1 else None,
    }


async def run_one(
    args: argparse.Namespace, system: str, repeat: int, remaining: float
) -> dict[str, Any]:
    prices, price_info = _prices(str(ROOT / "evals" / "prices.glm.yaml"))
    meter = CostMeter(prices, budget=remaining, currency="CNY")
    observer = HttpObserver(args.max_requests)
    http = httpx.AsyncClient(event_hooks=observer.hooks(), timeout=120.0)
    llm = OpenAICompatClient.from_env(
        meter, extra_params={"reasoning_effort": "low"}, http_client=http, timeout=120.0
    )
    taskset = parse_taskset(fanout_doc(), known_scorers=scorer_types())
    runner = EvalRunner(
        lambda: SYSTEMS[system](llm),  # type: ignore[arg-type,return-value]
        concurrency=1,
        budget_usd=remaining,
        meter=meter,
        include_timing=True,
        config={"llm": "openai", "model": llm.model, "target": f"experiments.targets:{system}",
                **price_info},
    )  # fmt: skip
    try:
        report = await runner.run(taskset)
    finally:
        await llm.aclose()
    rows = [
        {
            "task_id": c.task_id, "repeat": repeat, "status": c.status,
            "wall_s": round((report.timing or {})[c.key] / 1000, 2),
            "model_requests": c.model_requests,
        }
        for c in report.cases
    ]  # fmt: skip
    return {"rows": rows, "spent": meter.total, "http": observer.log, "report": report,
            "taskset_hash": taskset.content_hash}  # fmt: skip


async def run_all(args: argparse.Namespace) -> int:
    load_dotenv(str(ROOT / ".env"))
    arms: dict[str, dict[str, Any]] = {s: {"rows": [], "spent": 0.0, "http": []} for s in SYSTEMS}
    spent, task_hash = 0.0, ""
    for repeat in range(args.repeats):
        for system in SYSTEMS:  # alternate systems within each repeat
            out = await run_one(args, system, repeat, args.budget - spent)
            arm = arms[system]
            arm["rows"] += out["rows"]
            arm["spent"] += out["spent"]
            arm["http"] += out["http"]
            spent += out["spent"]
            task_hash = out["taskset_hash"]
            total = sum(r["wall_s"] for r in out["rows"])
            print(f"r{repeat} {system}: wall={total:.1f}s spent={spent:.4f}/{args.budget:g} CNY",
                  flush=True)  # fmt: skip
            reason = stop_reason(out["http"], spent, args.budget, [out["report"]])
            if reason:
                print(f"STOPPED after r{repeat} {system}: {reason}", flush=True)
                return EXIT_STOPPED
    result: dict[str, Any] = {
        "note": "Supplementary measurement, not part of the main results. Fanout family only.",
        "design": {
            "tasks": "the 8 fanout tasks of frozen task set A v2, unmodified",
            "fanout_subset_hash": task_hash, "repeats": args.repeats, "concurrency": 1,
            "planner_scheduler_max_concurrency": 4, "alternation": "single, planner per repeat",
            "model": "glm-5.3-flash", "reasoning_effort": "low", "run_date": args.run_date,
            "budget_cny": args.budget,
        },
        "arms": {},
    }  # fmt: skip
    for system, arm in arms.items():
        rows = arm["rows"]
        per_task = {}
        for tid in sorted({r["task_id"] for r in rows}):
            per_task[tid] = summarize_times([r for r in rows if r["task_id"] == tid])
        result["arms"][system] = {
            "passed": sum(r["status"] == "passed" for r in rows), "runs": len(rows),
            "wall": summarize_times(rows), "per_task": per_task,
            "spent_cny": round(arm["spent"], 4),
            "http": summarize_http(arm["http"], model_calls=sum(r["model_requests"] for r in rows)),
            "cases": rows,
        }  # fmt: skip
    Path(args.out).write_text(json.dumps(result, indent=1))  # noqa: ASYNC240
    print(json.dumps({s: a["wall"] for s, a in result["arms"].items()}, indent=1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=float, default=1.0, help="CNY cap")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-requests", type=int, default=600)
    parser.add_argument("--run-date", default=time.strftime("%Y-%m-%d"))
    sys.exit(asyncio.run(run_all(parser.parse_args())))


if __name__ == "__main__":
    main()
