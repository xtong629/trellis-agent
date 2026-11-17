from experiments.results_table import ROOT, load_metrics, render_tables


def test_metrics_come_from_the_committed_reports() -> None:
    m = load_metrics("a_planned")
    assert m["file"] == "evals/reports/exp_a_planned.json"
    assert (m["runs"], m["repeats"]) == (150, 5)
    assert round(m["pass"][0], 3) == 0.993 and round(m["pass"][1], 3) == 0.015
    assert m["calls"] == 1223 and m["tokens"] == 506477


def test_the_tables_name_their_source_and_cover_every_system() -> None:
    text = render_tables()
    for name in ("a_single", "a_planned", "b_window", "b_memory", "b_full"):
        assert f"exp_{name}.json" in text
    assert "0.993±0.015" in text and "0.980±0.027" in text
    assert (ROOT / "evals" / "reports" / "exp_b_full.json").exists()
