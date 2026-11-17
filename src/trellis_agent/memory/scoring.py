"""Memory scoring: relevance + recency (exponential decay) + importance."""

from __future__ import annotations

from dataclasses import dataclass

_DAY = 86_400.0


@dataclass(frozen=True)
class ScoringConfig:
    half_life_days: float = 14.0  # recency halves every this many days
    w_relevance: float = 1.0
    w_recency: float = 0.5
    w_importance: float = 0.5

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        if min(self.w_relevance, self.w_recency, self.w_importance) < 0:
            raise ValueError("weights must be non-negative")


def recency(age_seconds: float, half_life_days: float) -> float:
    return float(0.5 ** (max(age_seconds, 0.0) / (half_life_days * _DAY)))


def retention(importance: float, age_seconds: float, cfg: ScoringConfig) -> float:
    """How worth keeping a memory is (no query involved): used for forgetting."""
    return cfg.w_recency * recency(age_seconds, cfg.half_life_days) + cfg.w_importance * importance


def retrieval_score(
    relevance: float, importance: float, age_seconds: float, cfg: ScoringConfig
) -> float:
    return cfg.w_relevance * relevance + retention(importance, age_seconds, cfg)
