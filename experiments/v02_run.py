"""Real-API driver for the v0.2 experiments: baseline-only trials and full runs.

    # trial (before the freeze; output stays outside the repository)
    python -m experiments.v02_run c1 --trial --level 1 --systems c1_baseline --repeats 1 \
        --chunks-dir ../notes/v02/c1_trial
    # full run (task set must equal the frozen hash in experiments/FROZEN.json, key "v0.2")
    python -m experiments.v02_run c1 --systems c1_baseline c1_reflect --repeats 5 \
        --chunks-dir ../notes/v02/c1_full --reports-dir evals/reports

Work is split into chunks of one (repeat, system), run repeat-major with the systems alternating, so
API drift affects every system alike and a stop leaves even coverage. Finished chunks are kept and
skipped on resume. The run stops between chunks on: a non-200 response, a call slower than 60 s,
80% of the CNY budget or of the token cap spent, a case aborted by the budget, or a case that ended
with a provider or target error. A case that fails by itself (wrong answer, ``max_steps``) is a
result, not an anomaly, and does not stop the run. The API key is never logged."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from experiments.c1_tasks import taskset_c1
from experiments.full_run import SLOW_CALL_S, merge_reports
from experiments.pilot import HttpObserver, summarize_http
from experiments.targets import c1_baseline, c1_reflect
from trellis_agent.cost import CostMeter
from trellis_agent.env import load_dotenv
from trellis_agent.eval import (
    EvalReport,
    EvalRunner,
    TaskSet,
    build_scorer,
    parse_taskset,
    scorer_types,
)
from trellis_agent.eval.cli import _prices
from trellis_agent.llm.openai_compat import OpenAICompatClient
from trellis_agent.types import Message, RunResult, Usage

ROOT = Path(__file__).resolve().parents[1]
WARN = 0.8
EXIT_STOPPED = 3
INFRA_STOPS = {"llm_error", "target_error"}


@dataclass(frozen=True)
class Experiment:
    key: str  # FROZEN.json "v0.2" key and evals/ directory name
    make: Callable[[int], dict[str, Any]]  # level -> task set document
    systems: dict[str, Callable[..., Any]]
    timing: bool = False
    recording: frozenset[str] = frozenset()  # systems that take a reflection-record sink
    scheduler_max: dict[str, int] = field(default_factory=dict)


EXPERIMENTS: dict[str, Experiment] = {
    "c1": Experiment(
        "exp_c1_reflect",
        taskset_c1,
        {"c1_baseline": c1_baseline, "c1_reflect": c1_reflect},
        recording=frozenset({"c1_reflect"}),
    ),
}


def frozen_entry(key: str) -> dict[str, Any] | None:
    frozen = json.loads((ROOT / "experiments" / "FROZEN.json").read_text())
    entry: dict[str, Any] | None = frozen.get("v0.2", {}).get(key)
    return entry


def tokens_of(report: EvalReport) -> int:
    return sum(c.usage.prompt_tokens + c.usage.completion_tokens for c in report.cases) + sum(
        c.judge_usage.prompt_tokens + c.judge_usage.completion_tokens for c in report.cases
    )


def stop_reason(
    log: Sequence[dict[str, Any]],
    spent: float,
    budget: float,
    tokens: int,
    max_tokens: int,
    report: EvalReport,
) -> str | None:
    bad = [e for e in log if e["status"] != 200]
    if bad:
        return f"{len(bad)} non-200 response(s), first status {bad[0]['status']} (429 = rate limit)"
    slow = [e for e in log if e["latency_s"] > SLOW_CALL_S]
    if slow:
        return f"{len(slow)} call(s) slower than {SLOW_CALL_S:g}s"
    if spent >= WARN * budget:
        return f"budget close to the cap: spent {spent:.4f} of {budget:g} CNY"
    if tokens >= WARN * max_tokens:
        return f"token cap close: {tokens} of {max_tokens}"
    if report.aborted_by_budget or any(c.status in ("aborted", "not_run") for c in report.cases):
        return "cases were aborted or not run (budget)"
    infra = [c for c in report.cases if c.stop_reason in INFRA_STOPS]
    if infra:
        return f"{len(infra)} case(s) ended with a provider/target error, e.g. {infra[0].error}"
    return None


def chunk_plan(
    repeats: int, systems: Sequence[str], done: set[tuple[int, str]]
) -> list[tuple[int, str]]:
    return [(r, s) for r in range(repeats) for s in systems if (r, s) not in done]


def _paths(chunks: Path, system: str, repeat: int) -> tuple[Path, Path]:
    return chunks / f"{system}_r{repeat}.json", chunks / f"{system}_r{repeat}.meta.json"


def _records_path(chunks: Path, system: str, repeat: int) -> Path:
    return chunks / f"{system}_r{repeat}.records.json"


async def _number_ok(task: Any, answer: str) -> bool:
    spec = next(s for s in task.scorers if s.type == "number_match")
    result = RunResult(answer, [Message.assistant(answer)], Usage(), 0.0)
    return (await build_scorer(spec).score(task, result)).passed


async def attempts_report(
    taskset: TaskSet, per_repeat: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    """First answer, verdict and final answer of every reflected run, with the counts the
    pre-registration defines (fix, harm, false accept, false reject), scored by number_match."""
    tasks = {t.id: t for t in taskset.tasks}
    rows = []
    for repeat, records in enumerate(per_repeat):
        for rec in records:
            task = tasks[rec["session_id"].split("#", 1)[0]]
            first = await _number_ok(task, rec["first_answer"])
            final = await _number_ok(task, rec["final_answer"])
            rows.append({"task_id": task.id, "family": task.metadata.get("family"),
                         "repeat": repeat, **rec, "first_correct": first,
                         "final_correct": final})  # fmt: skip

    def counts(group: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "runs": len(group),
            "checked": sum(r["checked"] for r in group),
            "rejected": sum(r["verdict_ok"] is False for r in group),
            "retried": sum(r["retried"] for r in group),
            "unparsed_verdicts": sum(r["parsed"] is False for r in group),
            "first_correct": sum(r["first_correct"] for r in group),
            "final_correct": sum(r["final_correct"] for r in group),
            "fix": sum(not r["first_correct"] and r["final_correct"] for r in group),
            "harm": sum(r["first_correct"] and not r["final_correct"] for r in group),
            "false_accept": sum(r["verdict_ok"] is True and not r["first_correct"] for r in group),
            "false_reject": sum(r["verdict_ok"] is False and r["first_correct"] for r in group),
        }

    families = Counter(r["family"] for r in rows)
    return {
        "note": "Scored with each task's number_match rule only (pre-registered definition).",
        "taskset": {"name": taskset.name, "version": taskset.version,
                    "content_hash": taskset.content_hash},
        "totals": counts(rows),
        "by_family": {f: counts([r for r in rows if r["family"] == f]) for f in sorted(families)},
        "cases": rows,
    }  # fmt: skip


def _done(chunks: Path, systems: Sequence[str], repeats: int) -> set[tuple[int, str]]:
    return {(r, s) for r in range(repeats) for s in systems if _paths(chunks, s, r)[1].exists()}


def load_taskset(exp: Experiment, args: argparse.Namespace) -> tuple[Any, int]:
    """The task set to run: in a trial, the given level; otherwise the frozen one, checked."""
    if args.trial:
        return parse_taskset(exp.make(args.level), known_scorers=scorer_types()), args.level
    entry = frozen_entry(exp.key)
    if entry is None:
        raise SystemExit(f"{exp.key} is not frozen yet; only --trial runs are allowed")
    taskset = parse_taskset(exp.make(int(entry["level"])), known_scorers=scorer_types())
    if taskset.content_hash != entry["content_hash"]:
        raise SystemExit("task set differs from the frozen v0.2 hash; refusing to run")
    return taskset, int(entry["level"])


async def run_chunk(
    args: argparse.Namespace, exp: Experiment, system: str, repeat: int, remaining: float
) -> dict[str, Any]:
    prices, price_info = _prices(str(ROOT / "evals" / "prices.glm.yaml"))
    meter = CostMeter(prices, budget=remaining, currency="CNY")
    observer = HttpObserver(args.max_requests)
    http = httpx.AsyncClient(event_hooks=observer.hooks(), timeout=120.0)
    llm = OpenAICompatClient.from_env(
        meter, extra_params={"reasoning_effort": "low"}, http_client=http, timeout=120.0
    )
    taskset, level = load_taskset(exp, args)
    price = prices.get(llm.model)
    records: list[Any] = []

    def make() -> Any:
        if system in exp.recording:
            return exp.systems[system](llm, records.append)
        return exp.systems[system](llm)

    runner = EvalRunner(
        make,
        concurrency=args.concurrency,
        budget_usd=remaining,
        meter=meter,
        include_timing=exp.timing,
        config={
            "llm": "openai", "model": llm.model, "target": f"experiments.targets:{system}",
            "extra_params": {"reasoning_effort": "low"},
            "budget": {"amount": args.budget, "currency": "CNY"},
            "price_table": {llm.model: price.to_dict()} if price else {},
            **price_info,
            "meta": {"run_date": args.run_date, "experiment": exp.key, "system": system,
                     "trial": str(args.trial).lower(), "level": str(level)},
        },
    )  # fmt: skip
    started = time.perf_counter()
    try:
        report = await runner.run(taskset)
    finally:
        await llm.aclose()
    report_path, meta_path = _paths(Path(args.chunks_dir), system, repeat)
    report.save(report_path)
    if system in exp.recording:
        rows = [asdict(r) for r in records]
        _records_path(Path(args.chunks_dir), system, repeat).write_text(json.dumps(rows, indent=1))
    meta = {
        "spent": meter.total,
        "tokens": tokens_of(report),
        "wall_s": round(time.perf_counter() - started, 1),
        "model_calls": sum(c.model_requests for c in report.cases),
        "http": observer.log,
    }
    meta_path.write_text(json.dumps(meta))
    return meta


def family_rates(report: EvalReport) -> dict[str, str]:
    rows: dict[str, list[int]] = {}
    for case in report.cases:
        row = rows.setdefault(case.task_id.rsplit("-", 1)[0], [0, 0])
        row[0] += case.status == "passed"
        row[1] += 1
    return {k: f"{p}/{n}" for k, (p, n) in rows.items()}


async def merge_all(args: argparse.Namespace, exp: Experiment) -> dict[str, Any]:
    chunks, out = Path(args.chunks_dir), Path(args.reports_dir)
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    http: list[dict[str, Any]] = []
    calls = wall = 0.0
    spent_all = 0.0
    for system in args.systems:
        reports, spent = [], []
        for r in range(args.repeats):
            report_path, meta_path = _paths(chunks, system, r)
            reports.append(EvalReport.load(report_path))
            meta = json.loads(meta_path.read_text())
            spent.append(meta["spent"])
            http += meta["http"]
            calls, wall = calls + meta["model_calls"], wall + meta["wall_s"]
        inner = exp.scheduler_max.get(system, 1)
        merged = merge_reports(
            reports,
            concurrency={"jobs": args.concurrency, "scheduler_max_concurrency": inner,
                         "theoretical_max_in_flight": args.concurrency * inner},
            spent=spent,
            extra_meta={"repeats": str(args.repeats), "reasoning_effort": "low"},
        )  # fmt: skip
        merged.save(out / f"{exp.key.rsplit('_', 1)[0]}_{system.split('_', 1)[1]}.json")
        spent_all += merged.meter_total_usd or 0.0
        if system in exp.recording:
            taskset, _ = load_taskset(exp, args)
            per_repeat = [json.loads(_records_path(chunks, system, r).read_text())
                          for r in range(args.repeats)]  # fmt: skip
            sidecar = await attempts_report(taskset, per_repeat)
            name = f"{exp.key.rsplit('_', 1)[0]}_{system.split('_', 1)[1]}_attempts.json"
            (out / name).write_text(json.dumps(sidecar, indent=1))
    stats = summarize_http(http, model_calls=int(calls))
    stats |= {"sum_of_chunk_wall_s": wall, "spent_cny": spent_all}
    (out / f"{exp.key.rsplit('_', 1)[0]}_http.json").write_text(json.dumps(stats, indent=1))
    return stats


async def run_all(args: argparse.Namespace) -> int:
    exp = EXPERIMENTS[args.experiment]
    unknown = set(args.systems) - set(exp.systems)
    if unknown:
        raise SystemExit(f"unknown systems for {args.experiment}: {sorted(unknown)}")
    load_dotenv(str(ROOT / ".env"))
    chunks = Path(args.chunks_dir)
    chunks.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    done = _done(chunks, args.systems, args.repeats)
    metas = [json.loads(_paths(chunks, s, r)[1].read_text()) for r, s in done]
    spent, tokens = sum(m["spent"] for m in metas), sum(m["tokens"] for m in metas)
    for repeat, system in chunk_plan(args.repeats, args.systems, done):
        meta = await run_chunk(args, exp, system, repeat, args.budget - spent)
        spent, tokens = spent + meta["spent"], tokens + meta["tokens"]
        report = EvalReport.load(_paths(chunks, system, repeat)[0])
        print(f"r{repeat} {system}: {family_rates(report)} calls={meta['model_calls']} "
              f"tokens={meta['tokens']} wall={meta['wall_s']}s total_tokens={tokens} "
              f"spent={spent:.4f}/{args.budget:g} CNY", flush=True)  # fmt: skip
        reason = stop_reason(meta["http"], spent, args.budget, tokens, args.max_tokens, report)
        if reason:
            print(f"STOPPED after r{repeat} {system}: {reason}", flush=True)
            return EXIT_STOPPED
    if args.trial:
        return 0
    print(json.dumps(await merge_all(args, exp), indent=1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=list(EXPERIMENTS))
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--chunks-dir", required=True, help="outside the repository")
    parser.add_argument("--reports-dir", default="evals/reports")
    parser.add_argument("--trial", action="store_true", help="unfrozen task set; no merged report")
    parser.add_argument("--level", type=int, default=1, help="difficulty level (trials only)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budget", type=float, default=5.0, help="CNY cap for this run")
    parser.add_argument("--max-tokens", type=int, default=2_000_000, help="token cap for this run")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=2000, help="hard cap per chunk")
    parser.add_argument("--run-date", default=time.strftime("%Y-%m-%d"))
    sys.exit(asyncio.run(run_all(parser.parse_args())))


if __name__ == "__main__":
    main()
