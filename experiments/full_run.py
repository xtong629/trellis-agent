"""The full real-API experiment run: both task sets (v2), four systems, N repeats.

    python -m experiments.full_run --chunks-dir ../notes/full --reports-dir evals/reports

Work is split into chunks of one (repeat, system) so the run can stop between chunks when something
unusual happens (a non-200 response, a very slow call, the budget getting close, a failed case) and
be resumed afterwards: finished chunks are kept on disk and skipped. When every chunk is done they
are merged into one report per system (repeats numbered 0..N-1). The API key is never logged."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from experiments.pilot import HttpObserver, summarize_http
from experiments.scenarios import taskset_a, taskset_b
from experiments.targets import a_planned, a_single, b_full, b_memory, b_window
from trellis_agent.cost import CostMeter
from trellis_agent.env import load_dotenv
from trellis_agent.eval import EvalReport, EvalRunner, parse_taskset, scorer_types
from trellis_agent.eval.cli import _prices
from trellis_agent.llm.openai_compat import OpenAICompatClient

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {
    "a_single": ("a", a_single),
    "a_planned": ("a", a_planned),
    "b_window": ("b", b_window),
    "b_memory": ("b", b_memory),
    "b_full": ("b", b_full),  # reference condition, run separately (--systems b_full)
}
DEFAULT_SYSTEMS = ["a_single", "a_planned", "b_window", "b_memory"]
SCHEDULER_MAX = (
    4  # Scheduler's default max_concurrency (planner workers run in parallel up to this)
)
SLOW_CALL_S = 60.0
BUDGET_WARN = 0.8
EXIT_STOPPED = 3


def chunk_plan(
    *, repeats: int, done: set[tuple[int, str]], systems: Sequence[str] = DEFAULT_SYSTEMS
) -> list[tuple[int, str]]:
    """Repeat-major order, so an early stop leaves the most even coverage."""
    return [(r, s) for r in range(repeats) for s in systems if (r, s) not in done]


def stop_reason(
    log: Sequence[dict[str, Any]], spent: float, budget: float, reports: Sequence[EvalReport]
) -> str | None:
    bad = [e for e in log if e["status"] != 200]
    if bad:
        return f"{len(bad)} non-200 response(s), first status {bad[0]['status']} (429 = rate limit)"
    slow = [e for e in log if e["latency_s"] > SLOW_CALL_S]
    if slow:
        worst = max(e["latency_s"] for e in slow)
        return f"{len(slow)} slow call(s) over {SLOW_CALL_S:g}s, slowest {worst}s"
    if spent >= BUDGET_WARN * budget:
        return f"budget close to the cap: spent {spent:.4f} of {budget:g}"
    for report in reports:
        if report.aborted_by_budget or any(
            c.status in ("aborted", "not_run") for c in report.cases
        ):
            return "cases were aborted or not run (budget)"
        errored = [c for c in report.cases if c.error]
        if errored:
            return f"{len(errored)} case(s) ended with a run error, e.g. {errored[0].error[:80]}"  # type: ignore[index]
    return None


def merge_reports(
    chunks: Sequence[EvalReport],
    *,
    concurrency: dict[str, Any],
    spent: Sequence[float],
    extra_meta: dict[str, str],
) -> EvalReport:
    cases = [dataclasses.replace(c, repeat=i) for i, rep in enumerate(chunks) for c in rep.cases]
    first = chunks[0]
    total = math.fsum(spent)
    config = {**first.config, "concurrency": concurrency}
    config["meta"] = {**first.config.get("meta", {}), **extra_meta}
    return dataclasses.replace(
        first,
        config=config,
        cases=cases,
        meter_total_usd=total,
        aborted_by_budget=any(r.aborted_by_budget for r in chunks),
        cost_by_currency={first.currency or "": total},
        fx_applied=[],
    )


# ------------------------------------------------------------------ running (real API)
def _chunk_paths(chunks_dir: Path, system: str, repeat: int) -> tuple[Path, Path]:
    return chunks_dir / f"{system}_r{repeat}.json", chunks_dir / f"{system}_r{repeat}.meta.json"


def _done(chunks_dir: Path) -> set[tuple[int, str]]:
    return {
        (r, s) for r in range(100) for s in SYSTEMS if _chunk_paths(chunks_dir, s, r)[1].exists()
    }


async def run_chunk(
    args: argparse.Namespace, repeat: int, system: str, remaining: float
) -> dict[str, Any]:
    which, make = SYSTEMS[system]
    prices, price_info = _prices(str(ROOT / "evals" / "prices.glm.yaml"))
    meter = CostMeter(prices, budget=remaining, currency="CNY")
    observer = HttpObserver(args.max_requests)
    http = httpx.AsyncClient(event_hooks=observer.hooks(), timeout=120.0)
    llm = OpenAICompatClient.from_env(
        meter, extra_params={"reasoning_effort": "low"}, http_client=http, timeout=120.0
    )
    frozen = json.loads((ROOT / "experiments" / "FROZEN.json").read_text())["v2"]
    doc = {"a": taskset_a, "b": taskset_b}[which]()
    taskset = parse_taskset(doc, known_scorers=scorer_types())
    parent = frozen["exp_a_planner" if which == "a" else "exp_b_memory"]["content_hash"]
    if taskset.content_hash != parent:
        raise SystemExit("task set differs from the frozen v2 hash; refusing to run")
    price = prices.get(llm.model)
    price_table = {llm.model: price.to_dict()} if price else {}
    runner = EvalRunner(
        lambda: make(llm),  # type: ignore[arg-type]
        concurrency=args.concurrency,
        budget_usd=remaining,
        meter=meter,
        config={
            "llm": "openai", "model": llm.model, "target": f"experiments.targets:{system}",
            "extra_params": {"reasoning_effort": "low"},
            "budget": {"amount": args.budget, "currency": "CNY"},
            "price_table": price_table,
            **price_info,
            "meta": {"run_date": args.run_date, "experiment": f"{which}-full", "system": system,
                     "taskset_version": "2"},
        },
    )  # fmt: skip
    started = time.perf_counter()
    try:
        report = await runner.run(taskset)
    finally:
        await llm.aclose()
    report_path, meta_path = _chunk_paths(Path(args.chunks_dir), system, repeat)
    report.save(report_path)
    calls = sum(c.model_requests for c in report.cases)
    meta = {
        "spent": meter.total,
        "wall_s": round(time.perf_counter() - started, 1),
        "model_calls": calls,
        "http": observer.log,
    }
    meta_path.write_text(json.dumps(meta))
    return meta


def merge_all(args: argparse.Namespace) -> dict[str, Any]:
    chunks_dir, out = Path(args.chunks_dir), Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    http_log: list[dict[str, Any]] = []
    calls = wall = spent_all = 0.0
    for system in args.systems:
        reports, spent = [], []
        for r in range(args.repeats):
            report_path, meta_path = _chunk_paths(chunks_dir, system, r)
            reports.append(EvalReport.load(report_path))
            meta = json.loads(meta_path.read_text())
            spent.append(meta["spent"])
            http_log += meta["http"]
            calls, wall = calls + meta["model_calls"], wall + meta["wall_s"]
        theoretical = args.concurrency * (SCHEDULER_MAX if system == "a_planned" else 1)
        merged = merge_reports(
            reports,
            concurrency={"jobs": args.concurrency, "scheduler_max_concurrency":
                         SCHEDULER_MAX if system == "a_planned" else 1,
                         "theoretical_max_in_flight": theoretical},
            spent=spent,
            extra_meta={"repeats": str(args.repeats), "reasoning_effort": "low"},
        )  # fmt: skip
        merged.save(out / f"exp_{system}.json")
        spent_all += merged.meter_total_usd or 0.0
    stats = summarize_http(http_log, model_calls=int(calls))
    stats |= {"observed_max_in_flight": stats["max_in_flight"], "sum_of_chunk_wall_s": wall,
              "spent_cny": spent_all}  # fmt: skip
    name = (
        "exp_full_http.json"
        if args.systems == DEFAULT_SYSTEMS
        else f"exp_{'_'.join(args.systems)}_http.json"
    )
    (out / name).write_text(json.dumps(stats, indent=1))
    return stats


async def run_all(args: argparse.Namespace) -> int:
    load_dotenv(str(ROOT / ".env"))
    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    spent = sum(json.loads(_chunk_paths(chunks_dir, s, r)[1].read_text())["spent"]
                for r, s in _done(chunks_dir))  # fmt: skip
    for repeat, system in chunk_plan(
        repeats=args.repeats, done=_done(chunks_dir), systems=args.systems
    ):
        meta = await run_chunk(args, repeat, system, args.budget - spent)
        spent += meta["spent"]
        print(f"chunk r{repeat} {system}: calls={meta['model_calls']} wall={meta['wall_s']}s "
              f"spent={spent:.4f}/{args.budget:g} CNY", flush=True)  # fmt: skip
        report = EvalReport.load(_chunk_paths(chunks_dir, system, repeat)[0])
        reason = stop_reason(meta["http"], spent, args.budget, [report])
        if reason:
            print(f"STOPPED after r{repeat} {system}: {reason}", flush=True)
            return EXIT_STOPPED
    print(json.dumps(merge_all(args), indent=1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", required=True, help="outside the repository")
    parser.add_argument("--reports-dir", required=True)
    parser.add_argument("--budget", type=float, default=5.0, help="CNY cap for the whole run")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=DEFAULT_SYSTEMS)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=2000, help="hard cap per chunk")
    parser.add_argument("--run-date", default=time.strftime("%Y-%m-%d"))
    sys.exit(asyncio.run(run_all(parser.parse_args())))


if __name__ == "__main__":
    main()
