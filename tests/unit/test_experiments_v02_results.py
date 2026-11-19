"""The v0.2 result tables are computed from the committed reports, and the numbers quoted in
docs/EXPERIMENTS_V02.md follow from them (pre-registered decision rules applied by code)."""

from __future__ import annotations

import pytest
from experiments import v02_results


def test_c1_numbers_and_decision() -> None:
    r = v02_results.c1()
    base, refl = r["systems"]["baseline"], r["systems"]["reflect"]
    assert v02_results.ms(base["benefit"]) == "1.000±0.000"
    assert v02_results.ms(refl["benefit"]) == "0.978±0.030"
    assert v02_results.ms(refl["all"]) == "0.975±0.037"
    assert (base["tokens"], refl["tokens"], base["model_calls"], refl["model_calls"]) == (
        272_797, 335_647, 391, 524
    )  # fmt: skip
    assert round(base["cost_cny"], 4) == 0.2538 and round(refl["cost_cny"], 4) == 0.3133
    assert r["benefit_diff_mean"] == pytest.approx(-1 / 45)
    assert not r["band_ok"] and r["decision"].startswith("not run as designed")
    a = r["attempts"]
    assert (a["rejected"], a["fix"], a["harm"], a["false_accept"], a["false_reject"]) == (
        5,
        2,
        2,
        1,
        3,
    )
    assert r["easy_harm_check_ok"]
