"""The regression gate: compare a run against a saved baseline.

Exit codes: 0 = no regression; 1 = quality or cost regression; 2 = cannot compare (different task
set, incomplete run, cost needed but unknown). The distinction matters in CI: 1 means "the system
got worse", 2 means "fix the evaluation setup first". Rendered text always names the failing task,
the failing check and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .report import FAILED, CaseResult, EvalReport, short_hash

_EPS = 1e-9
_SNIPPET = 80


@dataclass(frozen=True)
class Finding:
    kind: str  # regression | improvement | pass_rate | cost | incomparable
    message: str
    task_id: str | None = None
    fatal: bool = False
    code: int = 1  # exit code this finding forces when fatal
    detail: tuple[str, ...] = ()


@dataclass
class Comparison:
    baseline: EvalReport
    current: EvalReport
    findings: list[Finding] = field(default_factory=list)
    max_pass_drop: float = 0.0
    max_cost_increase_pct: float | None = None

    @property
    def exit_code(self) -> int:
        fatal = [f.code for f in self.findings if f.fatal]
        return max(fatal) if fatal else 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    # ------------------------------------------------------------- rendering
    def render(self) -> str:
        b, c = self.baseline.summary(), self.current.summary()
        out = [self._headline(), ""]
        out.append(f"baseline: {_label(self.baseline)}")
        out.append(f"current:  {_label(self.current)}")
        if any(f.kind == "incomparable" for f in self.findings):
            out.append("")
            out += [f"  ! {f.message}" for f in self.findings if f.kind == "incomparable"]
            out += ["", f"exit code {self.exit_code}"]
            return "\n".join(out) + "\n"

        out.append("")
        out.append(
            f"pass rate : {_pct(b['pass_rate'])} -> {_pct(c['pass_rate'])}{self._drop_text(b, c)}"
        )
        out.append(f"cost      : {self._cost_text(b, c)}")
        regs = [f for f in self.findings if f.kind == "regression"]
        imps = [f for f in self.findings if f.kind == "improvement"]
        out.append(
            f"tasks     : {len(b['tasks'])} compared, {len(regs)} regressed, {len(imps)} improved"
        )
        if regs:
            allowed = "" if any(f.fatal for f in regs) else " (allowed by --max-regressed-tasks)"
            out += ["", f"REGRESSIONS{allowed}"]
            for f in regs:
                out.append(f"  - {f.message}")
                out += [f"      {line}" for line in f.detail]
        if imps:
            out += ["", "IMPROVEMENTS"]
            out += [f"  + {f.message}" for f in imps]
        others = [f for f in self.findings if f.fatal and f.kind in ("pass_rate", "cost")]
        if others:
            out += ["", "LIMITS EXCEEDED"]
            out += [f"  ! {f.message}" for f in others]
        out += ["", f"exit code {self.exit_code}"]
        return "\n".join(out) + "\n"

    def _headline(self) -> str:
        fatal = [f for f in self.findings if f.fatal]
        if not fatal:
            c = self.current.summary()
            return f"COMPARE OK: no regressions (pass rate {_pct(c['pass_rate'])})"
        if self.exit_code == 2:
            return "COMPARE INCONCLUSIVE: " + "; ".join(f.message for f in fatal if f.code == 2)
        parts: list[str] = []
        n = sum(1 for f in fatal if f.kind == "regression")
        if n:
            parts.append(f"{n} task{'s' if n != 1 else ''} regressed")
        parts += [f.message for f in fatal if f.kind in ("pass_rate", "cost")]
        return "COMPARE FAILED: " + "; ".join(parts)

    def _drop_text(self, b: dict[str, Any], c: dict[str, Any]) -> str:
        if b["pass_rate"] is None or c["pass_rate"] is None:
            return ""
        diff = (c["pass_rate"] - b["pass_rate"]) * 100
        return f" ({diff:+.1f} pp; limit {self.max_pass_drop * 100:.1f} pp)"

    def _cost_text(self, b: dict[str, Any], c: dict[str, Any]) -> str:
        bc, cc = b["cost"], c["cost"]
        pct = self.max_cost_increase_pct
        limit = "none" if pct is None else f"{pct:.1f}%"
        if bc is None or cc is None:
            return f"unknown (no price); limit {limit}"
        change = f"{(cc - bc) / bc * 100:+.1f}%; " if bc > 0 else ""
        cur = self.current.currency or "?"
        return f"{bc:.6f} {cur} -> {cc:.6f} {cur} ({change}limit {limit})"


def _label(report: EvalReport) -> str:
    return f"{report.taskset_name} v{report.taskset_version} [{short_hash(report.taskset_hash)}]"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _incomplete(report: EvalReport) -> str | None:
    s = report.summary()
    if report.aborted_by_budget or s["aborted_jobs"] or s["not_run_jobs"]:
        return (
            f"aborted by budget ({s['aborted_jobs']} aborted, {s['not_run_jobs']} not run "
            f"of {s['planned_jobs']} jobs)"
        )
    return None


def _failure_detail(cases: list[CaseResult]) -> tuple[str, ...]:
    failing = [c for c in cases if c.status == FAILED]
    if not failing:
        return ()
    first = failing[0]
    lines = []
    for score in first.scores:
        if not score.passed:
            lines.append(f"(repeat {first.repeat}) failed check {score.name}: {score.reason}")
    snippet = first.content.replace("\n", " ")
    if len(snippet) > _SNIPPET:
        snippet = snippet[:_SNIPPET] + "..."
    lines.append(f"(repeat {first.repeat}) output: {snippet!r}")
    if len(failing) > 1:
        lines.append(f"({len(failing) - 1} more failing repeat(s) not shown)")
    return tuple(lines)


def compare(
    baseline: EvalReport,
    current: EvalReport,
    *,
    max_pass_drop: float = 0.0,
    max_regressed_tasks: int = 0,
    max_cost_increase_pct: float | None = None,
) -> Comparison:
    """Gate ``current`` against ``baseline``.

    ``max_pass_drop`` is a fraction (0.05 = five percentage points); ``max_regressed_tasks`` is how
    many individual tasks may get worse; ``max_cost_increase_pct`` is optional (cost is only gated
    when you ask for it).
    """
    result = Comparison(
        baseline, current, max_pass_drop=max_pass_drop, max_cost_increase_pct=max_cost_increase_pct
    )
    found = result.findings

    def stop(message: str) -> None:
        found.append(Finding("incomparable", message, fatal=True, code=2))

    if baseline.taskset_name != current.taskset_name:
        stop(
            f"different task sets: baseline is {baseline.taskset_name!r}, "
            f"current is {current.taskset_name!r}"
        )
    elif baseline.taskset_hash != current.taskset_hash:
        stop(
            f"task set changed: baseline {_label(baseline)} vs current {_label(current)}; "
            "the tasks are not the same, so results are not comparable. "
            "Re-run the baseline on the new task set and save it before comparing"
        )
    if baseline.currency and current.currency and baseline.currency != current.currency:
        stop(
            f"cost currencies differ: baseline {baseline.currency}, current {current.currency}; "
            "costs in different currencies are never compared"
        )
    if (why := _incomplete(baseline)) is not None:
        stop(f"baseline is incomplete: {why}; save a complete baseline")
    if (why := _incomplete(current)) is not None:
        stop(f"current run is incomplete: {why}; no verdict can be given")
    if found:
        return result

    b, c = baseline.summary(), current.summary()
    cur_tasks = {t["task_id"]: t for t in c["tasks"]}
    cur_cases: dict[str, list[CaseResult]] = {}
    for case in current.cases:
        cur_cases.setdefault(case.task_id, []).append(case)

    regressions: list[Finding] = []
    for bt in b["tasks"]:
        ct = cur_tasks.get(bt["task_id"])
        if ct is None or bt["pass_rate"] is None or ct["pass_rate"] is None:
            continue
        was = f"{bt['passed']}/{bt['repeats_completed']}"
        now = f"{ct['passed']}/{ct['repeats_completed']}"
        label = f"{bt['task_id']}: passed {was} -> {now}"
        if ct["pass_rate"] < bt["pass_rate"] - _EPS:
            regressions.append(
                Finding(
                    "regression",
                    label,
                    bt["task_id"],
                    detail=_failure_detail(cur_cases[bt["task_id"]]),
                )
            )
        elif ct["pass_rate"] > bt["pass_rate"] + _EPS:
            found.append(Finding("improvement", label, bt["task_id"]))
    blocked = len(regressions) > max_regressed_tasks
    for f in regressions:
        found.append(Finding(f.kind, f.message, f.task_id, fatal=blocked, code=1, detail=f.detail))

    if b["pass_rate"] is not None and c["pass_rate"] is not None:
        drop = b["pass_rate"] - c["pass_rate"]
        if drop > max_pass_drop + _EPS:
            found.append(
                Finding(
                    "pass_rate",
                    f"pass rate dropped {drop * 100:.1f} pp (limit {max_pass_drop * 100:.1f} pp)",
                    fatal=True,
                )
            )

    if max_cost_increase_pct is not None:
        bc, cc = b["cost"], c["cost"]
        if bc is None or cc is None:
            found.append(
                Finding(
                    "incomparable",
                    "cost is unknown for one of the runs (model has no price), "
                    "so the cost limit cannot be checked",
                    fatal=True,
                    code=2,
                )
            )
        elif bc > 0:
            pct = (cc - bc) / bc * 100
            if pct > max_cost_increase_pct + _EPS:
                found.append(
                    Finding(
                        "cost",
                        f"cost rose {pct:.1f}% (limit {max_cost_increase_pct:.1f}%)",
                        fatal=True,
                    )
                )
    return result
