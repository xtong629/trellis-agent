import pytest

from trellis_agent.cost import BudgetExceeded, CostMeter, ModelPrice, PriceTable
from trellis_agent.eval.judge import LLMJudge
from trellis_agent.eval.scorers import ScorerConfigError, build_scorer
from trellis_agent.eval.taskset import EvalTask, ScorerSpec
from trellis_agent.llm import LLMError, RuleLLM, ScriptedLLM
from trellis_agent.types import Message, RunResult, Usage

TASK = EvalTask(id="t", input="Summarise the refund policy.")
RUBRIC = "Mentions the three-day review window."
JUDGE_USAGE = Usage(200, 20)


def result(content: str = "Refunds are reviewed within three days.") -> RunResult:
    return RunResult(content, [], Usage(1, 1), None)


def judge_llm(*replies: str, meter: CostMeter | None = None) -> ScriptedLLM:
    return ScriptedLLM(
        [Message.assistant(r) for r in replies],
        model="judge-model",
        usage_per_call=JUDGE_USAGE,
        meter=meter,
    )


async def test_passing_grade_carries_the_judges_own_usage_and_cost() -> None:
    meter = CostMeter(PriceTable({"judge-model": ModelPrice(1.0, 1.0)}))
    llm = judge_llm('{"score": 0.9, "reason": "mentions three days"}', meter=meter)
    scored = await LLMJudge(llm, RUBRIC).score(TASK, result())
    assert (scored.passed, scored.score, scored.judge, scored.error) == (True, 0.9, True, False)
    assert scored.reason == "mentions three days"
    assert scored.usage == JUDGE_USAGE
    assert scored.cost_usd == pytest.approx(220 / 1_000_000)
    assert meter.by_tag == {
        "judge": pytest.approx(220 / 1_000_000)
    }  # tagged separately in the meter


async def test_threshold_decides_pass_or_fail() -> None:
    reply = '{"score": 0.5, "reason": "partial"}'
    assert not (await LLMJudge(judge_llm(reply), RUBRIC).score(TASK, result())).passed
    assert (
        await LLMJudge(judge_llm(reply), RUBRIC, pass_threshold=0.5).score(TASK, result())
    ).passed


async def test_prompt_contains_task_rubric_and_answer_at_temperature_zero() -> None:
    llm = judge_llm('{"score": 1, "reason": "ok"}')
    await LLMJudge(llm, RUBRIC).score(TASK, result("the answer text"))
    call = llm.calls[0]
    assert call.temperature == 0.0 and call.tag == "judge" and call.tools is None
    text = "\n".join(m.content or "" for m in call.messages)
    assert TASK.input in text and RUBRIC in text and "the answer text" in text
    assert "json" in (call.messages[0].content or "").lower()


async def test_long_answers_are_truncated_and_empty_answers_are_marked() -> None:
    llm = judge_llm('{"score": 1, "reason": "ok"}', '{"score": 0, "reason": "empty"}')
    await LLMJudge(llm, RUBRIC, max_answer_chars=10).score(TASK, result("x" * 50))
    await LLMJudge(llm, RUBRIC).score(TASK, result(""))
    first = "\n".join(m.content or "" for m in llm.calls[0].messages)
    assert "x" * 10 in first and "x" * 11 not in first and "truncated" in first
    assert "(empty answer)" in "\n".join(m.content or "" for m in llm.calls[1].messages)


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n{"score": 0.8, "reason": "fenced"}\n```',
        'Sure! Here is my grade: {"score": 0.8, "reason": "chatty"} Hope that helps.',
        '{"score": 0.8, "reason": "plain"}',
    ],
)
async def test_tolerates_fences_and_chatter_around_the_json(reply: str) -> None:
    scored = await LLMJudge(judge_llm(reply), RUBRIC).score(TASK, result())
    assert scored.passed and scored.score == 0.8 and not scored.error


@pytest.mark.parametrize(
    "reply",
    [
        "I think it is good.",
        '{"reason": "no score"}',
        '{"score": "high", "reason": "x"}',
        '{"score": 1.5, "reason": "out of range"}',
        '{"score": -0.1, "reason": "out of range"}',
        '{"score": true, "reason": "bool"}',
        "[0.9]",
    ],
)
async def test_unusable_output_is_a_scorer_error_but_the_spend_is_kept(reply: str) -> None:
    scored = await LLMJudge(judge_llm(reply), RUBRIC).score(TASK, result())
    assert not scored.passed and scored.error and scored.judge and scored.score == 0.0
    assert "judge" in scored.reason
    assert scored.usage == JUDGE_USAGE  # we paid for that call even though it was useless


async def test_judge_model_failure_is_a_scorer_error() -> None:
    def fail(messages: list[Message], tools: object) -> Message:
        raise LLMError("judge provider down")

    scored = await LLMJudge(RuleLLM(fail), RUBRIC).score(TASK, result())
    assert not scored.passed and scored.error and "provider down" in scored.reason
    assert scored.usage is None and scored.cost_usd is None


async def test_budget_exceeded_propagates_with_the_paid_response() -> None:
    meter = CostMeter(PriceTable({"judge-model": ModelPrice(1.0, 1.0)}), budget_usd=0.0001)
    llm = ScriptedLLM(
        [Message.assistant('{"score": 1, "reason": "ok"}')],
        model="judge-model",
        usage_per_call=Usage(1_000_000, 0),
        meter=meter,
    )
    with pytest.raises(BudgetExceeded) as info:
        await LLMJudge(llm, RUBRIC).score(TASK, result())
    assert info.value.response is not None and info.value.response.cost_usd == pytest.approx(1.0)


def test_built_from_a_task_spec() -> None:
    llm = judge_llm()
    spec = ScorerSpec("llm_judge", {"rubric": RUBRIC, "pass_threshold": 0.5})
    scorer = build_scorer(spec, judge_llm=llm)
    assert isinstance(scorer, LLMJudge) and scorer.pass_threshold == 0.5
    assert scorer.name.startswith("llm_judge(") and scorer.judge is True


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ({}, "rubric"),
        ({"rubric": "  "}, "rubric"),
        ({"rubric": "r", "pass_threshold": 1.5}, "pass_threshold"),
        ({"rubric": "r", "pass_threshold": -0.1}, "pass_threshold"),
        ({"rubric": "r", "rubrik": "typo"}, "rubrik"),
    ],
)
def test_bad_judge_config_rejected(params: dict[str, object], fragment: str) -> None:
    with pytest.raises(ScorerConfigError, match=fragment):
        build_scorer(ScorerSpec("llm_judge", params), judge_llm=judge_llm())
