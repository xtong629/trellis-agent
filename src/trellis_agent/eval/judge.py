"""LLM-as-judge scoring.

A judge grades open-ended answers against a rubric, but it is a second model with its own cost and
noise. So: temperature 0, a fixed rubric, strict output parsing, the judge's tokens and cost are
returned on the ``ScoreResult`` (reported separately from the agent's), and an unusable judgement
is a *scorer error*, never a silent pass. Prefer rule scorers wherever a rule can decide.
"""

from __future__ import annotations

from typing import Any

from ..jsonutil import extract_json
from ..llm import LLMClient, LLMError
from ..types import Message, RunResult
from .scorers import Params, ScorerConfigError, ScoreResult
from .taskset import EvalTask

_SYSTEM = (
    "You are a strict, consistent grader. Grade the ANSWER to the TASK against the RUBRIC. "
    'Reply with ONLY a JSON object: {"score": <number from 0 to 1>, "reason": "<one sentence>"}. '
    "1 means the rubric is fully satisfied, 0 means not at all."
)


class LLMJudge:
    judge = True  # the runner skips judges for runs that already failed (nothing to grade)

    def __init__(
        self,
        llm: LLMClient,
        rubric: str,
        *,
        pass_threshold: float = 0.7,
        tag: str = "judge",
        name: str | None = None,
        max_answer_chars: int = 4000,
    ) -> None:
        if not rubric.strip():
            raise ScorerConfigError("scorer 'llm_judge': parameter 'rubric' must be non-empty")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ScorerConfigError(
                "scorer 'llm_judge': parameter 'pass_threshold' must be in [0, 1]"
            )
        self.llm = llm
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.tag = tag
        self.max_answer_chars = max_answer_chars
        self.name = name or f"llm_judge({rubric[:40]!r})"

    @classmethod
    def from_params(cls, p: Params, llm: LLMClient) -> LLMJudge:
        rubric = p.string("rubric")
        threshold = p.number("pass_threshold", 0.7)
        name = p.string("name", "") or None
        p.finish()
        return cls(llm, rubric, pass_threshold=threshold, name=name)

    def _prompt(self, task: EvalTask, answer: str) -> list[Message]:
        if not answer.strip():
            shown = "(empty answer)"
        elif len(answer) > self.max_answer_chars:
            shown = answer[: self.max_answer_chars] + "\n...[answer truncated for grading]"
        else:
            shown = answer
        body = f"TASK:\n{task.input}\n\nRUBRIC:\n{self.rubric}\n\nANSWER:\n{shown}"
        return [Message.system(_SYSTEM), Message.user(body)]

    async def score(self, task: EvalTask, result: RunResult) -> ScoreResult:
        try:
            response = await self.llm.chat(
                self._prompt(task, result.content), temperature=0.0, tag=self.tag
            )
        except LLMError as exc:  # BudgetExceeded is deliberately NOT caught: the runner aborts
            return self._error(f"judge model failed: {exc}")
        verdict = extract_json(response.message.content or "")
        raw = None if verdict is None else verdict.get("score")
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            return self._error("judge output had no numeric 'score'", response)
        if not 0.0 <= raw <= 1.0:
            return self._error(f"judge score {raw} is outside [0, 1]", response)
        assert verdict is not None
        score = float(raw)
        return ScoreResult(
            self.name,
            passed=score >= self.pass_threshold,
            score=score,
            reason=str(verdict.get("reason", "")),
            judge=True,
            usage=response.usage,
            cost_usd=response.cost_usd,
        )

    def _error(self, reason: str, response: Any = None) -> ScoreResult:
        return ScoreResult(
            self.name,
            passed=False,
            score=0.0,
            reason=reason,
            judge=True,
            error=True,
            usage=None if response is None else response.usage,
            cost_usd=None if response is None else response.cost_usd,
        )
