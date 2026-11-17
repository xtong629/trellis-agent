"""Pilot run: a ~1/10 scale slice of both experiments against the real API, to measure rate
limits, latency and token use before the full run. Not part of the library.

    python -m experiments.pilot --out-dir ../notes/pilot --budget 10 --concurrency 2

Results go to --out-dir (keep it outside the repository). The API key is never logged: the HTTP
observer records only status, latency, concurrency and the Retry-After value."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from experiments.scenarios import taskset_a, taskset_b
from experiments.targets import a_planned, a_single, b_memory, b_window
from trellis_agent.cost import CostMeter, PriceTable
from trellis_agent.env import load_dotenv
from trellis_agent.eval import EvalRunner, parse_taskset, scorer_types
from trellis_agent.llm.openai_compat import OpenAICompatClient

ROOT = Path(__file__).resolve().parents[1]
PILOT_PICKS = {  # tasks taken (in order) from each family
    "a": {"serial": 4, "fanout": 3, "flaky": 2, "single": 4},
    "b": {"allergy": 3, "deadline": 2, "refund-zh": 1, "fits": 1, "irrelevant": 1},
}
SYSTEMS = [("a_single", "a", a_single), ("a_planned", "a", a_planned),
           ("b_window", "b", b_window), ("b_memory", "b", b_memory)]  # fmt: skip


def pilot_subset(doc: dict[str, Any], which: str) -> dict[str, Any]:
    left = dict(PILOT_PICKS[which])
    tasks = []
    for task in doc["tasks"]:
        family = task["metadata"]["family"]
        if left.get(family, 0) > 0:
            tasks.append(task)
            left[family] -= 1
    return {**doc, "name": f"{doc['name']}-pilot", "tasks": tasks}


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def summarize_http(log: list[dict[str, Any]], *, model_calls: int) -> dict[str, Any]:
    latencies = [float(e["latency_s"]) for e in log]
    statuses = Counter(str(e["status"]) for e in log)
    return {
        "http_requests": len(log),
        "model_calls": model_calls,
        "extra_http_requests": len(log) - model_calls,  # SDK-level retries
        "status_counts": dict(sorted(statuses.items())),
        "rate_limited": statuses.get("429", 0),
        "retry_after_seen": sorted({e["retry_after"] for e in log if e["retry_after"]}),
        "max_in_flight": max((int(e["in_flight"]) for e in log), default=0),
        "latency_s": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p99": percentile(latencies, 99),
            "max": max(latencies, default=None),
        },
    }


class HttpObserver:
    """httpx event hooks that log status/latency per request, with a hard cap on requests."""

    def __init__(self, max_requests: int) -> None:
        self.max_requests = max_requests
        self.log: list[dict[str, Any]] = []
        self._started: dict[int, float] = {}
        self._count = 0
        self._in_flight = 0

    async def _on_request(self, request: httpx.Request) -> None:
        if self._count >= self.max_requests:
            raise RuntimeError(f"hard cap of {self.max_requests} HTTP requests reached")
        self._count += 1
        self._in_flight += 1
        request.extensions["observer"] = (time.perf_counter(), self._in_flight)

    async def _on_response(self, response: httpx.Response) -> None:
        started, in_flight = response.request.extensions["observer"]
        self._in_flight -= 1
        self.log.append(
            {
                "status": response.status_code,
                "latency_s": round(time.perf_counter() - started, 3),
                "in_flight": in_flight,
                "retry_after": response.headers.get("retry-after"),
            }
        )

    def hooks(self) -> dict[str, list[Any]]:
        return {"request": [self._on_request], "response": [self._on_response]}


async def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(str(ROOT / ".env"))
    prices = PriceTable.from_yaml(ROOT / "evals" / "prices.glm.yaml")
    meter = CostMeter(prices, budget=args.budget, currency="CNY")
    observer = HttpObserver(args.max_requests)
    frozen = json.loads((ROOT / "experiments" / "FROZEN.json").read_text())
    http = httpx.AsyncClient(event_hooks=observer.hooks(), timeout=120.0)
    llm = OpenAICompatClient.from_env(
        meter, extra_params={"reasoning_effort": "low"}, http_client=http, timeout=120.0
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    sets = {"a": taskset_a(), "b": taskset_b()}
    parents = {
        "a": frozen["exp_a_planner"]["content_hash"],
        "b": frozen["exp_b_memory"]["content_hash"],
    }
    started = time.perf_counter()
    summary: dict[str, Any] = {"systems": {}}
    try:
        for name, which, make in SYSTEMS:
            if meter.exhausted:
                break
            taskset = parse_taskset(pilot_subset(sets[which], which), known_scorers=scorer_types())
            runner = EvalRunner(
                lambda make=make: make(llm),  # type: ignore[misc]
                concurrency=args.concurrency,
                budget_usd=args.budget,
                meter=meter,
                config={
                    "llm": "openai", "model": llm.model, "target": f"experiments.targets:{name}",
                    "extra_params": {"reasoning_effort": "low"},
                    "budget": {"amount": args.budget, "currency": "CNY"},
                    "meta": {"pilot": "true", "run_date": args.run_date,
                             "parent_taskset_hash": parents[which]},
                },
            )  # fmt: skip
            report = await runner.run(taskset)
            report.save(out / f"{name}.json")
            summary["systems"][name] = report.summary()
    finally:
        await llm.aclose()
    summary["wall_s"] = round(time.perf_counter() - started, 1)
    calls = sum(s["model_requests"] for s in summary["systems"].values())
    summary["http"] = summarize_http(observer.log, model_calls=calls)
    summary["spent_cny"] = meter.total
    (out / "http_log.json").write_text(json.dumps(observer.log))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budget", type=float, default=10.0, help="CNY cap for the whole pilot")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=700)
    parser.add_argument("--run-date", default=time.strftime("%Y-%m-%d"))
    summary = asyncio.run(run_pilot(parser.parse_args()))
    print(json.dumps({k: v for k, v in summary.items() if k != "systems"}, indent=1))


if __name__ == "__main__":
    main()
