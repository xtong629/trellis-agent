"""C4 (v0.2): calibrate the LLM judge against the rule scorers on answers already committed.

    python -m experiments.c4_judge build                      # answer set from committed reports
    python -m experiments.c4_judge trial --n 20 --out ../notes/v02/c4_trial.json
    python -m experiments.c4_judge run --out evals/reports/exp_c4_judge.json

No task is added and no task set is changed: the answers are the final replies stored in committed
reports, and the reference for each answer comes from its task's own content scorers. Runs that
ended with an error are left out (the eval runner never sends those to a judge either). The judge
is the same model as the one that wrote the answers, so self-preference is possible; there are no
human labels.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from experiments.pilot import HttpObserver, summarize_http
from trellis_agent.cost import CostMeter
from trellis_agent.env import load_dotenv
from trellis_agent.eval import EvalReport, EvalTask, LLMJudge, build_scorer, load_taskset
from trellis_agent.eval.calibrate import agreement
from trellis_agent.eval.cli import _prices
from trellis_agent.llm.openai_compat import OpenAICompatClient
from trellis_agent.types import Message, RunResult, Usage

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {  # committed report -> task set directory
    "exp_a_single": "exp_a_planner", "exp_a_planned": "exp_a_planner",
    "exp_b_window": "exp_b_memory", "exp_b_memory": "exp_b_memory", "exp_b_full": "exp_b_memory",
    "exp_c1_baseline": "exp_c1_reflect", "exp_c1_reflect": "exp_c1_reflect",
}  # fmt: skip
CONTENT_RULES = ("number_match", "contains", "not_contains")
ANSWERS = ROOT / "evals" / "exp_c4_judge" / "answers.json"
RUBRIC = (
    "Judge only whether the final answer is correct; ignore formatting, length and style. "
    "Expected: {expected}. Score 1 if the answer's final answer meets this, otherwise 0."
)
CONSISTENCY_N = 100
CONSISTENCY_PASSES = 2  # extra judge passes on the consistency subset (3 verdicts in total)


def expected_text(task: EvalTask) -> str:
    parts = []
    for spec in task.scorers:
        p = spec.params
        if spec.type == "number_match":
            tol = float(p.get("tolerance", 0))
            parts.append(f"the number {p['value']:g}" + (f" (within {tol:g})" if tol else ""))
        elif spec.type == "contains":
            parts.append(f"it states {p['value']!r} (letter case does not matter)")
        elif spec.type == "not_contains":
            parts.append(f"it does not give {p['value']!r}")
    return "; ".join(parts)


async def content_rule(task: EvalTask, content: str) -> tuple[bool, list[str]]:
    """The task's content scorers on a reply: passed, and the reasons of those that failed."""
    result = RunResult(content, [Message.assistant(content)], Usage(), 0.0)
    failed = []
    for spec in task.scorers:
        if spec.type in CONTENT_RULES:
            score = await build_scorer(spec).score(task, result)
            if not score.passed:
                failed.append(f"{spec.type}: {score.reason}")
    return not failed, failed


def _hash(rows: list[dict[str, Any]]) -> str:
    canon = json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


async def build_answers() -> dict[str, Any]:
    rows, excluded = [], Counter()
    for report_name, set_dir in SOURCES.items():
        path = ROOT / "evals" / "reports" / f"{report_name}.json"
        if not path.exists():
            raise SystemExit(f"missing committed report {path.name}")
        report = EvalReport.load(path)
        tasks = {t.id: t for t in load_taskset(ROOT / "evals" / set_dir).tasks}
        for case in report.cases:
            if case.error is not None or case.stop_reason != "end_turn":
                excluded[report_name] += 1
                continue
            task = tasks[case.task_id]
            passed, reasons = await content_rule(task, case.content)
            rows.append({
                "id": f"{report_name}:{case.task_id}#{case.repeat}", "report": report_name,
                "task_set": set_dir, "task_id": case.task_id, "repeat": case.repeat,
                "family": task.metadata.get("family"), "content": case.content,
                "rule_content_pass": passed, "rule_failures": reasons,
                "case_status": case.status,
            })  # fmt: skip
    return {"rows": rows, "excluded_error_runs": dict(excluded), "content_hash": _hash(rows)}


def consistency_subset(rows: list[dict[str, Any]]) -> list[str]:
    """Up to half rule failures, the rest passes, in a fixed hash order (no randomness)."""

    def order(row: dict[str, Any]) -> str:
        return hashlib.sha256(row["id"].encode()).hexdigest()

    fails = sorted((r for r in rows if not r["rule_content_pass"]), key=order)
    passes = sorted((r for r in rows if r["rule_content_pass"]), key=order)
    picked = fails[: CONSISTENCY_N // 2]
    picked += passes[: CONSISTENCY_N - len(picked)]
    return [r["id"] for r in picked]


def category(row: dict[str, Any], judge_pass: bool) -> str | None:
    rule = row["rule_content_pass"]
    if judge_pass == rule:
        return None
    if not judge_pass:
        return "judge rejected an answer the rule accepts"
    reasons = " ".join(row["rule_failures"])
    if "no number after" in reasons:
        return "format: right or wrong value, but no ANSWER line"
    if reasons.startswith("not_contains") or " not_contains" in reasons:
        return "mentions a value the rule forbids (e.g. a superseded date)"
    return "value: the judge accepted what the rule rejects"


class Judging:
    def __init__(self, budget: float, max_requests: int) -> None:
        prices, self.price_info = _prices(str(ROOT / "evals" / "prices.glm.yaml"))
        self.meter = CostMeter(prices, budget=budget, currency="CNY")
        self.observer = HttpObserver(max_requests)
        http = httpx.AsyncClient(event_hooks=self.observer.hooks(), timeout=120.0)
        self.llm = OpenAICompatClient.from_env(
            self.meter, extra_params={"reasoning_effort": "low"}, http_client=http, timeout=120.0
        )
        self.tasks = {d: {t.id: t for t in load_taskset(ROOT / "evals" / d).tasks}
                      for d in set(SOURCES.values())}  # fmt: skip
        self.gate = asyncio.Semaphore(4)

    async def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        task = self.tasks[row["task_set"]][row["task_id"]]
        judge = LLMJudge(self.llm, RUBRIC.format(expected=expected_text(task)), pass_threshold=0.5)
        result = RunResult(row["content"], [Message.assistant(row["content"])], Usage(), 0.0)
        async with self.gate:
            score = await judge.score(task, result)
        usage = score.usage or Usage()
        return {"passed": None if score.error else score.passed, "score": score.score,
                "reason": score.reason, "error": score.error,
                "tokens": usage.prompt_tokens + usage.completion_tokens}  # fmt: skip


def analyse(rows: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if verdicts[r["id"]]["passed"] is not None]
    primary = [(verdicts[r["id"]]["passed"], r["rule_content_pass"]) for r in valid]
    secondary = [(verdicts[r["id"]]["passed"], r["case_status"] == "passed") for r in valid]
    by_report = {
        name: agreement([(verdicts[r["id"]]["passed"], r["rule_content_pass"])
                         for r in valid if r["report"] == name], bootstrap=0).to_dict()
        for name in SOURCES
    }  # fmt: skip
    disagreements = []
    for r in valid:
        cat = category(r, verdicts[r["id"]]["passed"])
        if cat:
            disagreements.append({"id": r["id"], "category": cat, "content": r["content"][-300:],
                                  "rule_failures": r["rule_failures"],
                                  "judge_reason": verdicts[r["id"]]["reason"]})  # fmt: skip
    return {
        "answers": len(rows), "judged": len(valid), "judge_errors": len(rows) - len(valid),
        "primary_judge_vs_content_rule": agreement(primary, seed=20261119).to_dict(),
        "secondary_judge_vs_case_status": agreement(secondary, seed=20261119).to_dict(),
        "by_report": by_report,
        "disagreement_categories": dict(Counter(d["category"] for d in disagreements)),
        "disagreements": disagreements,
    }  # fmt: skip


async def judge_all(rows: list[dict[str, Any]], judging: Judging) -> dict[str, dict[str, Any]]:
    out = await asyncio.gather(*(judging.judge(r) for r in rows))
    return {r["id"]: v for r, v in zip(rows, out, strict=True)}


async def run(args: argparse.Namespace) -> int:
    load_dotenv(str(ROOT / ".env"))
    data = json.loads(ANSWERS.read_text())
    frozen = json.loads((ROOT / "experiments" / "FROZEN.json").read_text())
    entry = frozen.get("v0.2", {}).get("exp_c4_judge")
    if args.command == "run" and (entry is None or entry["content_hash"] != _hash(data["rows"])):
        raise SystemExit("the answer set is not frozen or differs from the frozen hash")
    rows = data["rows"]
    if args.command == "trial":
        ids = set(
            consistency_subset(rows)[: args.n // 2] + consistency_subset(rows)[-args.n // 2 :]
        )
        rows = [r for r in rows if r["id"] in ids]
    judging = Judging(args.budget, args.max_requests)
    started = time.perf_counter()
    try:
        verdicts = await judge_all(rows, judging)
        extra: list[dict[str, dict[str, Any]]] = []
        if args.command == "run":
            subset = [r for r in rows if r["id"] in set(consistency_subset(rows))]
            for _ in range(CONSISTENCY_PASSES):
                extra.append(await judge_all(subset, judging))
    finally:
        await judging.llm.aclose()
    bad = [e for e in judging.observer.log if e["status"] != 200]
    result = analyse(rows, verdicts)
    if extra:
        same = sum(len({verdicts[i]["passed"], *(p[i]["passed"] for p in extra)}) == 1
                   for i in extra[0])  # fmt: skip
        result["consistency"] = {"answers": len(extra[0]), "verdicts_per_answer": 1 + len(extra),
                                 "all_identical": same}  # fmt: skip
    result |= {
        "note": ("Judge = the same model that wrote the answers (glm-5.3-flash, reasoning_effort "
                 "low): self-preference is possible. No human labels."),
        "rubric": RUBRIC, "answer_set_hash": _hash(data["rows"]),
        "excluded_error_runs": data["excluded_error_runs"],
        "tokens": judging.meter.usage.prompt_tokens + judging.meter.usage.completion_tokens,
        "cost_cny": judging.meter.total, "http": summarize_http(judging.observer.log,
                                                                model_calls=judging.meter.calls),
        "wall_s": round(time.perf_counter() - started, 1), "run_date": args.run_date,
        "verdicts": verdicts,
    }  # fmt: skip
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))  # noqa: ASYNC240
    summary = {k: result[k] for k in ("answers", "judged", "judge_errors", "tokens", "cost_cny")}
    print(json.dumps(summary | {"kappa": result["primary_judge_vs_content_rule"]["kappa"],
                                "non_200": len(bad)}, indent=1))  # fmt: skip
    return 3 if bad else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "trial", "run"])
    parser.add_argument("--out", default="evals/reports/exp_c4_judge.json")
    parser.add_argument("--n", type=int, default=20, help="trial size")
    parser.add_argument("--budget", type=float, default=1.0, help="CNY cap")
    parser.add_argument("--max-requests", type=int, default=1500)
    parser.add_argument("--run-date", default=time.strftime("%Y-%m-%d"))
    args = parser.parse_args()
    if args.command == "build":
        data = asyncio.run(build_answers())
        ANSWERS.parent.mkdir(parents=True, exist_ok=True)
        ANSWERS.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        print(json.dumps({"answers": len(data["rows"]), "hash": data["content_hash"],
                          "excluded": data["excluded_error_runs"]}))  # fmt: skip
        return
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
