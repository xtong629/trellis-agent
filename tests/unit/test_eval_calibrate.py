"""Agreement between two pass/fail graders: counts, Cohen's kappa, a bootstrap interval."""

from __future__ import annotations

import pytest

from trellis_agent.eval.calibrate import agreement, cohen_kappa


def pairs(tt: int, ff: int, tf: int, ft: int) -> list[tuple[bool, bool]]:
    return [(True, True)] * tt + [(False, False)] * ff + [(True, False)] * tf + [(False, True)] * ft


def test_kappa_on_a_textbook_table() -> None:
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa 0.4
    assert cohen_kappa(pairs(20, 15, 5, 10)) == pytest.approx(0.4)


def test_perfect_and_chance_agreement() -> None:
    assert cohen_kappa(pairs(30, 20, 0, 0)) == pytest.approx(1.0)
    assert cohen_kappa(pairs(25, 25, 25, 25)) == pytest.approx(0.0)


def test_kappa_is_undefined_when_both_graders_never_vary() -> None:
    assert cohen_kappa(pairs(10, 0, 0, 0)) is None
    assert cohen_kappa([]) is None


def test_agreement_report() -> None:
    result = agreement(pairs(20, 15, 5, 10), bootstrap=500, seed=1)
    assert (result.n, result.both_pass, result.both_fail) == (50, 20, 15)
    assert (result.first_only, result.second_only) == (5, 10)
    assert result.agree_rate == pytest.approx(0.7) and result.kappa == pytest.approx(0.4)
    assert result.kappa_ci is not None
    low, high = result.kappa_ci
    assert low < 0.4 < high and low >= -1.0 and high <= 1.0
    again = agreement(pairs(20, 15, 5, 10), bootstrap=500, seed=1)
    assert again.kappa_ci == result.kappa_ci  # deterministic for a seed
    assert set(result.to_dict()) >= {"n", "agree_rate", "kappa", "kappa_ci"}


def test_no_bootstrap_and_empty_input() -> None:
    assert agreement(pairs(3, 3, 1, 1), bootstrap=0).kappa_ci is None
    empty = agreement([])
    assert empty.n == 0 and empty.agree_rate is None and empty.kappa is None
    with pytest.raises(ValueError):
        agreement(pairs(1, 1, 0, 0), level=1.5)
