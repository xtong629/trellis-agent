import pytest

from trellis_agent.memory.scoring import ScoringConfig, recency, retention, retrieval_score

DAY = 86_400.0


def test_recency_halves_every_half_life() -> None:
    assert recency(0, 14) == 1.0
    assert recency(14 * DAY, 14) == pytest.approx(0.5)
    assert recency(28 * DAY, 14) == pytest.approx(0.25)
    assert recency(7 * DAY, 14) == pytest.approx(0.5**0.5)


def test_a_clock_that_runs_backwards_does_not_boost_recency() -> None:
    assert recency(-5 * DAY, 14) == 1.0


def test_hand_calculated_scores() -> None:
    cfg = ScoringConfig(half_life_days=10, w_relevance=1.0, w_recency=0.5, w_importance=0.5)
    # age 20 days = 2 half-lives -> recency 0.25; importance 0.8
    assert retention(0.8, 20 * DAY, cfg) == pytest.approx(0.5 * 0.25 + 0.5 * 0.8)  # 0.525
    assert retrieval_score(1.0, 0.8, 20 * DAY, cfg) == pytest.approx(1.0 + 0.525)
    assert retrieval_score(0.0, 0.8, 20 * DAY, cfg) == pytest.approx(0.525)  # no relevance term


def test_score_falls_with_age_and_rises_with_importance() -> None:
    cfg = ScoringConfig()
    ages = [0, 1, 7, 30, 365]
    scores = [retention(0.5, a * DAY, cfg) for a in ages]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1]
    assert retention(0.9, 30 * DAY, cfg) > retention(0.1, 30 * DAY, cfg)


def test_an_old_important_memory_outlasts_a_fresh_trivial_one_only_by_its_importance() -> None:
    cfg = ScoringConfig(half_life_days=1, w_recency=0.5, w_importance=0.5)
    old_important = retention(0.9, 100 * DAY, cfg)  # recency ~ 0, so ~0.45 from importance
    fresh_trivial = retention(0.0, 0, cfg)  # 0.5 from recency alone
    assert old_important == pytest.approx(0.45, abs=1e-6) and fresh_trivial == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [{"half_life_days": 0}, {"half_life_days": -1}, {"w_recency": -0.1}, {"w_importance": -1}],
)
def test_invalid_config_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ScoringConfig(**kwargs)
