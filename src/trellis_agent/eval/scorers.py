"""Scorers decide whether one run of a task is acceptable.

Rule scorers are deterministic and free. ``llm_judge`` (see ``judge.py``) can grade open-ended
answers but costs tokens, so its usage is reported separately. Scorers that cannot verify their
condition (e.g. a cost limit when the model has no price) fail with an explanation rather than
passing silently.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..types import RunResult, Usage
from .taskset import EvalTask, ScorerSpec

if TYPE_CHECKING:
    from ..llm import LLMClient

_MISSING: Any = object()
_NUMBER = re.compile(r"(?<!\w)-?\d+(?:,\d{3})*(?:\.\d+)?")


class ScorerConfigError(ValueError):
    """A scorer was configured with missing, mistyped or unknown parameters."""


@dataclass(frozen=True)
class ScoreResult:
    name: str
    passed: bool
    score: float
    reason: str
    judge: bool = False  # True when the score came from a model (its cost is reported separately)
    error: bool = False  # the scorer itself failed (e.g. unparsable judge output)
    usage: Usage | None = None
    cost_usd: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ScoreResult:
        usage = raw.get("usage")
        return cls(
            name=str(raw["name"]),
            passed=bool(raw["passed"]),
            score=float(raw["score"]),
            reason=str(raw.get("reason", "")),
            judge=bool(raw.get("judge", False)),
            error=bool(raw.get("error", False)),
            usage=None if usage is None else Usage.from_dict(usage),
            cost_usd=raw.get("cost", raw.get("cost_usd")),  # old key: cost_usd
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "judge": self.judge,
            "error": self.error,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "cost": self.cost_usd,
        }


class Scorer(Protocol):
    name: str

    async def score(self, task: EvalTask, result: RunResult) -> ScoreResult: ...


# ----------------------------------------------------------------- parameter validation
class Params:
    """Typed access to a scorer's parameters; ``finish`` rejects typos (unknown parameters)."""

    def __init__(self, kind: str, raw: Mapping[str, Any]) -> None:
        self.kind = kind
        self._raw = dict(raw)
        self._used: set[str] = set()

    def _get(self, key: str, default: Any) -> Any:
        self._used.add(key)
        if key in self._raw:
            return self._raw[key]
        if default is _MISSING:
            raise ScorerConfigError(f"scorer {self.kind!r}: missing required parameter {key!r}")
        return default

    def _bad(self, key: str, expected: str) -> ScorerConfigError:
        return ScorerConfigError(f"scorer {self.kind!r}: parameter {key!r} must be {expected}")

    def string(self, key: str, default: Any = _MISSING) -> str:
        value = self._get(key, default)
        if not isinstance(value, str):
            raise self._bad(key, "a string")
        return value

    def number(self, key: str, default: Any = _MISSING, *, minimum: float | None = None) -> float:
        value = self._get(key, default)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._bad(key, "a number")
        if minimum is not None and value < minimum:
            raise self._bad(key, f"a number >= {minimum:g}")
        return value

    def integer(self, key: str, default: Any = _MISSING, *, minimum: int | None = None) -> int:
        value = self._get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise self._bad(key, "an integer")
        if minimum is not None and value < minimum:
            raise self._bad(key, f"an integer >= {minimum}")
        return value

    def boolean(self, key: str, default: Any = _MISSING) -> bool:
        value = self._get(key, default)
        if not isinstance(value, bool):
            raise self._bad(key, "true or false")
        return value

    def choice(self, key: str, options: tuple[str, ...], default: Any = _MISSING) -> str:
        value = self._get(key, default)
        if value not in options:
            raise self._bad(key, f"one of {list(options)}")
        return str(value)

    def finish(self) -> None:
        unknown = sorted(set(self._raw) - self._used)
        if unknown:
            raise ScorerConfigError(f"scorer {self.kind!r}: unknown parameter(s) {unknown}")


# ----------------------------------------------------------------- rule scorers
Check = Callable[[EvalTask, RunResult], tuple[bool, str]]


class _Rule:
    def __init__(self, name: str, check: Check) -> None:
        self.name = name
        self._check = check

    async def score(self, task: EvalTask, result: RunResult) -> ScoreResult:
        ok, reason = self._check(task, result)
        return ScoreResult(self.name, ok, 1.0 if ok else 0.0, reason)


def called_tools(result: RunResult) -> list[str]:
    return [str(e.data.get("name")) for e in result.trace if e.type == "tool_call"]


def _fold(text: str, casefold: bool) -> str:
    return text.casefold() if casefold else text


def _exact_match(p: Params) -> Scorer:
    value, strip, fold = p.string("value"), p.boolean("strip", True), p.boolean("casefold", False)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        got = result.content.strip() if strip else result.content
        want = value.strip() if strip else value
        ok = _fold(got, fold) == _fold(want, fold)
        return ok, "ok" if ok else f"expected exactly {value!r}, got {got[:80]!r}"

    return _Rule(f"exact_match({value!r})", check)


def _contains(p: Params) -> Scorer:
    value, fold = p.string("value"), p.boolean("casefold", True)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        ok = _fold(value, fold) in _fold(result.content, fold)
        return ok, "ok" if ok else f"output does not contain {value!r}"

    return _Rule(f"contains({value!r})", check)


def _not_contains(p: Params) -> Scorer:
    value, fold = p.string("value"), p.boolean("casefold", True)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        found = _fold(value, fold) in _fold(result.content, fold)
        return (not found), f"output contains forbidden text {value!r}" if found else "ok"

    return _Rule(f"not_contains({value!r})", check)


def _regex(p: Params) -> Scorer:
    pattern, ignore = p.string("pattern"), p.boolean("ignore_case", False)
    p.finish()
    try:
        compiled = re.compile(pattern, re.IGNORECASE if ignore else 0)
    except re.error as exc:
        raise ScorerConfigError(f"scorer 'regex': invalid regex {pattern!r}: {exc}") from exc

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        ok = compiled.search(result.content) is not None
        return ok, "ok" if ok else f"output does not match /{pattern}/"

    return _Rule(f"regex({pattern!r})", check)


def _numbers(text: str) -> list[float]:
    return [float(m.group().replace(",", "")) for m in _NUMBER.finditer(text)]


def _number_match(p: Params) -> Scorer:
    value = p.number("value")
    tolerance = p.number("tolerance", 0.0, minimum=0.0)
    which = p.choice("which", ("any", "last", "marked"), "any")
    marker = p.string("marker", "ANSWER:")
    p.finish()
    marked = re.compile(rf"^\W*{re.escape(marker)}(.*)$", re.IGNORECASE | re.MULTILINE)

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        found = _numbers(result.content)
        pool = found[-1:] if which == "last" else found
        if which == "marked":  # the first number on the last marked line is the answer
            lines = marked.findall(result.content)
            found = _numbers(lines[-1]) if lines else []
            pool = found[:1]
            if not pool:
                return False, f"no number after the {marker!r} answer line"
        ok = any(abs(n - value) <= tolerance for n in pool)
        if ok:
            return True, "ok"
        seen = [f"{n:g}" for n in found[:5]] or "none"
        kind = {"last": "final number", "marked": f"number on the {marker!r} line"}.get(
            which, "number"
        )
        return (
            False,
            f"no {kind} equal to {value:g} (+/-{tolerance:g}); numbers in output: {seen}",
        )

    return _Rule(f"number_match({value!r})", check)


def _tool_called(p: Params) -> Scorer:
    name, minimum = p.string("name"), p.integer("min_times", 1, minimum=1)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        calls = called_tools(result)
        n = calls.count(name)
        if n >= minimum:
            return True, "ok"
        seen = sorted(set(calls)) or "none"
        return (
            False,
            f"tool {name!r} called {n} time(s), expected at least {minimum}; tools called: {seen}",
        )

    return _Rule(f"tool_called({name!r})", check)


def _tool_forbidden(p: Params) -> Scorer:
    name = p.string("name")
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        n = called_tools(result).count(name)
        return n == 0, "ok" if n == 0 else f"forbidden tool {name!r} was called {n} time(s)"

    return _Rule(f"tool_forbidden({name!r})", check)


def _max_tokens(p: Params) -> Scorer:
    limit = p.integer("value", minimum=0)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        used = result.usage.total_tokens
        return used <= limit, "ok" if used <= limit else f"used {used} tokens, limit {limit}"

    return _Rule(f"max_tokens({limit})", check)


def _max_cost(p: Params) -> Scorer:
    limit = p.number("value", minimum=0.0)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        if result.cost_usd is None:
            return False, "cost unknown (the model has no price), so the limit cannot be verified"
        ok = result.cost_usd <= limit
        return ok, "ok" if ok else f"cost ${result.cost_usd:.6f} exceeds limit ${limit:.6f}"

    return _Rule(f"max_cost_usd({limit:g})", check)


def _max_latency(p: Params) -> Scorer:
    limit = p.number("value", minimum=0.0)
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        ends = [e for e in result.trace if e.type == "run_end" and "duration_ms" in e.data]
        if not ends:
            return False, "no latency data in the trace, so the limit cannot be verified"
        took = float(ends[-1].data["duration_ms"])
        return took <= limit, "ok" if took <= limit else f"took {took:.1f}ms, limit {limit:g}ms"

    return _Rule(f"max_latency_ms({limit:g})", check)


def _stop_reason(p: Params) -> Scorer:
    want = p.string("value", "end_turn")
    p.finish()

    def check(task: EvalTask, result: RunResult) -> tuple[bool, str]:
        ok = result.stop_reason == want
        return ok, "ok" if ok else f"stop_reason was {result.stop_reason!r}, expected {want!r}"

    return _Rule(f"stop_reason({want!r})", check)


_RULES: dict[str, Callable[[Params], Scorer]] = {
    "exact_match": _exact_match,
    "contains": _contains,
    "not_contains": _not_contains,
    "regex": _regex,
    "number_match": _number_match,
    "tool_called": _tool_called,
    "tool_forbidden": _tool_forbidden,
    "max_tokens": _max_tokens,
    "max_cost_usd": _max_cost,
    "max_latency_ms": _max_latency,
    "stop_reason": _stop_reason,
}


def scorer_types() -> list[str]:
    return sorted([*_RULES, "llm_judge"])


def build_scorer(spec: ScorerSpec, *, judge_llm: LLMClient | None = None) -> Scorer:
    params = Params(spec.type, spec.params)
    if spec.type == "llm_judge":
        if judge_llm is None:
            raise ScorerConfigError("scorer 'llm_judge' needs a judge model (none configured)")
        from .judge import LLMJudge  # local import: judge builds on this module

        return LLMJudge.from_params(params, judge_llm)
    factory = _RULES.get(spec.type)
    if factory is None:
        raise ScorerConfigError(
            f"unknown scorer type {spec.type!r} (known: {', '.join(scorer_types())})"
        )
    return factory(params)


# ----------------------------------------------------------------- built-in success check, custom
def check_success(task: EvalTask, result: RunResult) -> ScoreResult:
    """Every run must end without error, unless the task says an error is the expected outcome."""
    failed = result.error is not None
    if task.expect_success:
        reason = "ok" if not failed else f"run ended with {result.stop_reason}: {result.error}"
        return ScoreResult("run_succeeded", not failed, 0.0 if failed else 1.0, reason)
    reason = "ok" if failed else "expected the run to fail, but it succeeded"
    return ScoreResult("run_fails_as_expected", failed, 1.0 if failed else 0.0, reason)


CustomReturn = bool | str | None | tuple[bool, str] | ScoreResult


class CustomScorer:
    """Wrap a Python callable ``(task, result) -> True | False | None | "reason" | (ok, reason)``.

    ``True``/``None`` pass; ``False`` fails; a non-empty string fails *with that message*.
    """

    def __init__(self, name: str, fn: Callable[[EvalTask, RunResult], Any]) -> None:
        self.name = name
        self._fn = fn

    async def score(self, task: EvalTask, result: RunResult) -> ScoreResult:
        try:
            outcome = self._fn(task, result)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception as exc:
            return ScoreResult(
                self.name,
                False,
                0.0,
                f"custom scorer raised {type(exc).__name__}: {exc}",
                error=True,
            )
        if isinstance(outcome, ScoreResult):
            return outcome
        if outcome is None or outcome is True:
            return ScoreResult(self.name, True, 1.0, "ok")
        if outcome is False:
            return ScoreResult(self.name, False, 0.0, "custom check returned False")
        if isinstance(outcome, str):
            return ScoreResult(self.name, not outcome, 0.0 if outcome else 1.0, outcome or "ok")
        if isinstance(outcome, tuple) and outcome:
            ok = bool(outcome[0])
            reason = (
                str(outcome[1]) if len(outcome) > 1 else ("ok" if ok else "custom check failed")
            )
            return ScoreResult(self.name, ok, 1.0 if ok else 0.0, reason)
        return ScoreResult(
            self.name, False, 0.0, f"unsupported custom scorer result {outcome!r}", error=True
        )
