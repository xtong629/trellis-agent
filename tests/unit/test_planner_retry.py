import random

import pytest

from trellis_agent.llm import LLMError
from trellis_agent.planner.retry import RetryPolicy, TransientError
from trellis_agent.types import RunResult, Usage


def failed(stop_reason: str) -> RunResult:
    return RunResult("", [], Usage(), None, [], "boom", stop_reason)


def test_delays_double_each_attempt_without_jitter() -> None:
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, factor=2.0, jitter=0.0)
    rng = random.Random(0)
    assert [policy.delay(n, rng) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 4.0]


def test_delay_is_capped() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=10.0, max_delay=5.0, jitter=0.0)
    assert policy.delay(3, random.Random(0)) == 5.0


def test_jitter_stays_within_its_band_and_is_reproducible_with_a_seed() -> None:
    policy = RetryPolicy(base_delay=1.0, factor=2.0, jitter=0.2)
    draws = [policy.delay(2, random.Random(seed)) for seed in range(200)]
    assert all(1.6 <= d <= 2.4 for d in draws)  # 2.0 +/- 20%
    assert len(set(draws)) > 50  # it really varies
    assert policy.delay(2, random.Random(7)) == policy.delay(2, random.Random(7))


@pytest.mark.parametrize(
    "exc",
    [TransientError("x"), TimeoutError(), ConnectionError(), LLMError("provider down")],
)
def test_transient_errors_are_retryable(exc: Exception) -> None:
    assert RetryPolicy().is_retryable_error(exc)


@pytest.mark.parametrize("exc", [ValueError("bad"), KeyError("k"), RuntimeError("bug")])
def test_programming_and_validation_errors_are_not_retried(exc: Exception) -> None:
    assert not RetryPolicy().is_retryable_error(exc)


def test_custom_retry_on() -> None:
    policy = RetryPolicy(retry_on=(KeyError,))
    assert policy.is_retryable_error(KeyError("k")) and not policy.is_retryable_error(
        TimeoutError()
    )


def test_only_provider_failures_are_retryable_run_results() -> None:
    policy = RetryPolicy()
    assert policy.is_retryable_result(failed("llm_error"))
    for reason in ("max_steps", "budget_exceeded", "end_turn", "target_error"):
        assert not policy.is_retryable_result(failed(reason))
    assert RetryPolicy(retry_stop_reasons=frozenset({"max_steps"})).is_retryable_result(
        failed("max_steps")
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"base_delay": -1.0},
        {"factor": 0.5},
        {"jitter": 1.0},
        {"jitter": -0.1},
        {"max_delay": 0.0},
    ],
)
def test_invalid_policy_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)  # type: ignore[arg-type]
