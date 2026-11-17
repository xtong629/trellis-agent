"""Retry policy: exponential backoff with jitter, and which failures deserve a retry.

Only *transient* failures are retried (timeouts, connection problems, provider errors). A bug, a
validation error, running out of steps or running out of budget would fail again, so retrying them
only wastes time and money.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..llm import LLMError
from ..types import RunResult


class TransientError(Exception):
    """Raise this from a worker or tool to say "trying again may help"."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    factor: float = 2.0
    max_delay: float = 30.0
    jitter: float = 0.2  # +/- fraction of the delay, so many retries do not stampede together
    retry_on: tuple[type[BaseException], ...] = (
        TransientError,
        TimeoutError,
        ConnectionError,
        LLMError,
    )
    retry_stop_reasons: frozenset[str] = frozenset({"llm_error"})

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay <= 0:
            raise ValueError("delays must be positive")
        if self.factor < 1:
            raise ValueError("factor must be at least 1")
        if not 0 <= self.jitter < 1:
            raise ValueError("jitter must be in [0, 1)")

    def delay(self, attempt: int, rng: random.Random) -> float:
        """Seconds to wait after failed attempt number ``attempt`` (1-based)."""
        base = min(self.base_delay * self.factor ** (attempt - 1), self.max_delay)
        return base * (1 + rng.uniform(-self.jitter, self.jitter)) if self.jitter else base

    def is_retryable_error(self, exc: BaseException) -> bool:
        return isinstance(exc, self.retry_on)

    def is_retryable_result(self, result: RunResult) -> bool:
        return result.error is not None and result.stop_reason in self.retry_stop_reasons
