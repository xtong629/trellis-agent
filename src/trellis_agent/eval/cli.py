"""Command line: ``python -m trellis_agent.eval {run,compare,save-baseline}``.

Exit codes: 0 ok; 1 regression gate failed (compare); 2 configuration / comparison problem
(message on stderr or in the compare output); 3 ``run`` was aborted by its budget (results partial).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..cost import CostMeter, PriceTable, UnpricedModelError
from ..env import load_dotenv
from ..llm import LLMClient, LLMError
from ..llm.openai_compat import OpenAICompatClient, check_extra_params
from . import demo
from .baseline import compare
from .report import EvalReport, ReportError, short_hash
from .runner import EvalRunner, Runnable
from .scorers import ScorerConfigError, scorer_types
from .taskset import TaskSetError, load_taskset

EXIT_OK, EXIT_REGRESSION, EXIT_ERROR, EXIT_BUDGET = 0, 1, 2, 3

TargetMaker = Callable[[LLMClient], Runnable]


class CLIError(Exception):
    """A problem with how the command was used; reported without a traceback."""


def _load_target(spec: str | None) -> TargetMaker:
    if spec is None:
        return demo.make_target
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise CLIError(f"--target must look like module:callable, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CLIError(f"cannot import {module_name!r} for --target: {exc}") from exc
    target = getattr(module, attr, None)
    if not callable(target):
        raise CLIError(f"--target {spec!r}: {attr!r} is not a callable in {module_name!r}")
    return target  # type: ignore[no-any-return]


def _parse_extra_params(items: list[str]) -> dict[str, Any]:
    """KEY=VALUE pairs; VALUE is parsed as JSON when possible (so objects and numbers work)."""
    extra: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise CLIError(f"--extra-param must look like KEY=VALUE, got {item!r}")
        try:
            extra[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            extra[key.strip()] = raw
    return check_extra_params(extra)


def _parse_budget(text: str) -> tuple[float, str]:
    """ "10 CNY" -> (10.0, "CNY"). A budget without a currency is ambiguous, so it is refused."""
    amount, _, currency = text.strip().partition(" ")
    try:
        value = float(amount)
    except ValueError:
        value = -1.0
    if value < 0 or not currency.strip():
        raise CLIError(f"a budget is an amount and a currency, e.g. '10 CNY'; got {text!r}")
    return value, currency.strip()


def _budget(args: argparse.Namespace) -> tuple[float, str] | None:
    """--budget, else --budget-usd (deprecated), else BUDGET, else BUDGET_USD (deprecated)."""
    if args.budget:
        return _parse_budget(args.budget)
    if args.budget_usd is not None:
        return args.budget_usd, "USD"
    if args.llm == "mock":
        return None
    if os.environ.get("BUDGET"):
        return _parse_budget(os.environ["BUDGET"])
    if os.environ.get("BUDGET_USD"):
        return float(os.environ["BUDGET_USD"]), "USD"
    raise CLIError("real runs need a budget: pass --budget '10 CNY' or set BUDGET (e.g. '10 CNY')")


def _prices(path: str | None) -> tuple[PriceTable, dict[str, Any]]:
    if path is None:
        return PriceTable(), {}
    file = Path(path)
    if not file.is_file():
        raise CLIError(f"price file not found: {file}")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    return PriceTable.from_yaml(file), {
        "prices_file": file.name,
        "prices_sha256": short_hash(digest),
    }


async def _run_async(args: argparse.Namespace) -> EvalReport:
    uses_judge = False
    taskset = load_taskset(args.taskset, known_scorers=scorer_types())
    uses_judge = any(s.type == "llm_judge" for t in taskset.tasks for s in t.scorers)

    prices, price_info = _prices(args.prices)
    if args.llm == "openai":
        load_dotenv()  # BUDGET may live in .env
    limit = _budget(args)
    budget = None if limit is None else limit[0]
    meter = CostMeter(prices, budget=budget, currency=None if limit is None else limit[1])
    extra = _parse_extra_params(args.extra_param)

    clients: list[LLMClient] = []
    judge: LLMClient | None = None
    if args.llm == "mock":
        llm: LLMClient = demo.make_mock_llm(meter)
    else:
        load_dotenv()
        llm = OpenAICompatClient.from_env(meter, extra_params=extra)
    clients.append(llm)
    judge_kind = args.judge_llm if args.judge_llm != "auto" else args.llm
    if uses_judge and judge_kind != "none":
        if judge_kind == "mock":
            judge = demo.make_mock_judge(meter)
        else:
            load_dotenv()
            judge = OpenAICompatClient.from_env(meter, model=os.environ.get("LLM_JUDGE_MODEL"))
        clients.append(judge)

    if (
        limit is not None
    ):  # a budget cannot be enforced on unpriced calls or unconvertible currencies
        for client in clients:
            price = prices.get(client.model)
            if price is None:
                raise CLIError(
                    f"cannot enforce the budget: no price for model {client.model!r} "
                    "(pass --prices FILE with that model's prices)"
                )
            if price.currency != limit[1] and prices.fx_rate(price.currency, limit[1]) is None:
                raise CLIError(
                    f"model {client.model!r} is priced in {price.currency} but the budget is in "
                    f"{limit[1]}; use a budget in {price.currency} or declare an fx rate "
                    f"{price.currency}->{limit[1]} (with a source and date) in the price file"
                )

    make = _load_target(args.target)
    runner = EvalRunner(
        lambda: make(llm),
        repeats=args.repeats,
        concurrency=args.concurrency,
        budget_usd=budget,
        meter=meter,
        judge_llm=judge,
        include_timing=args.include_timing,
        config={
            "llm": args.llm,
            "model": llm.model,
            "judge_model": None if judge is None else judge.model,
            "target": args.target or "trellis_agent.eval.demo:make_target",
            **({"extra_params": extra} if extra else {}),
            **price_info,
        },
    )
    try:
        return await runner.run(taskset)
    finally:
        for client in clients:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()


def _cmd_run(args: argparse.Namespace) -> int:
    report = asyncio.run(_run_async(args))
    text = report.render_markdown()
    print(text, end="")
    if args.out:
        report.save(args.out)
    if args.md:
        Path(args.md).write_text(text, encoding="utf-8")
    return EXIT_BUDGET if report.aborted_by_budget else EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    result = compare(
        EvalReport.load(args.baseline),
        EvalReport.load(args.current),
        max_pass_drop=args.max_pass_drop,
        max_regressed_tasks=args.max_regressed_tasks,
        max_cost_increase_pct=args.max_cost_increase_pct,
    )
    print(result.render(), end="")
    return result.exit_code


def _cmd_save_baseline(args: argparse.Namespace) -> int:
    report = EvalReport.load(args.report)
    s = report.summary()
    if report.aborted_by_budget or s["aborted_jobs"] or s["not_run_jobs"]:
        raise CLIError(
            f"report is incomplete ({s['aborted_jobs']} aborted, {s['not_run_jobs']} not run); "
            "refusing to save a partial run as a baseline"
        )
    out = Path(args.out or f"evals/baselines/{report.taskset_name}.json")
    report.save(out)
    label = f"{report.taskset_name} v{report.taskset_version} [{short_hash(report.taskset_hash)}]"
    print(f"saved baseline: {label} -> {out}")
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trellis_agent.eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task set and write a report")
    run.add_argument("taskset", help="task set file (.yaml/.json/.jsonl) or directory")
    run.add_argument("--llm", choices=["mock", "openai"], default="mock")
    run.add_argument("--judge-llm", choices=["auto", "mock", "openai", "none"], default="auto")
    run.add_argument("--target", help="module:callable taking an LLMClient, returning a runnable")
    run.add_argument("--prices", help="YAML of per-model prices (USD per million tokens)")
    run.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra request field for the main model, e.g. reasoning_effort=low (repeatable); "
        "recorded in the report config",
    )
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--budget", help="hard cap: an amount and a currency, e.g. '10 CNY'")
    run.add_argument("--budget-usd", type=float, help="deprecated: same as --budget 'N USD'")
    run.add_argument("--out", help="write the JSON report here")
    run.add_argument("--md", help="write the Markdown report here")
    run.add_argument("--include-timing", action="store_true", help="record wall-clock per case")
    run.set_defaults(func=_cmd_run)

    cmp_ = sub.add_parser("compare", help="gate a report against a baseline")
    cmp_.add_argument("baseline")
    cmp_.add_argument("current")
    cmp_.add_argument("--max-pass-drop", type=float, default=0.0, help="fraction, e.g. 0.05")
    cmp_.add_argument("--max-regressed-tasks", type=int, default=0)
    cmp_.add_argument("--max-cost-increase-pct", type=float, default=None)
    cmp_.set_defaults(func=_cmd_compare)

    save = sub.add_parser("save-baseline", help="store a complete report as the baseline")
    save.add_argument("report")
    save.add_argument("--out", help="default: evals/baselines/<task set name>.json")
    save.set_defaults(func=_cmd_save_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (
        TaskSetError,
        ScorerConfigError,
        ReportError,
        CLIError,
        LLMError,
        UnpricedModelError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
