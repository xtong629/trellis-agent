"""Evaluation: versioned task sets, scorers (rule + LLM judge), a cost-aware runner, and a
baseline regression gate. See docs/EVAL.md."""

from .baseline import Comparison, Finding, compare
from .judge import LLMJudge
from .report import CaseResult, EvalReport, ReportError
from .runner import EvalRunner, Runnable
from .scorers import CustomScorer, ScorerConfigError, ScoreResult, build_scorer, scorer_types
from .taskset import EvalTask, ScorerSpec, TaskSet, TaskSetError, load_taskset, parse_taskset

__all__ = [
    "CaseResult",
    "Comparison",
    "CustomScorer",
    "EvalReport",
    "EvalRunner",
    "EvalTask",
    "Finding",
    "LLMJudge",
    "ReportError",
    "Runnable",
    "ScoreResult",
    "ScorerConfigError",
    "ScorerSpec",
    "TaskSet",
    "TaskSetError",
    "build_scorer",
    "compare",
    "load_taskset",
    "parse_taskset",
    "scorer_types",
]
