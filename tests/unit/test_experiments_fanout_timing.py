from experiments.fanout_timing import fanout_doc, summarize_times
from experiments.scenarios import taskset_a


def test_fanout_doc_keeps_the_frozen_fanout_tasks_unchanged() -> None:
    full = {t["id"]: t for t in taskset_a()["tasks"]}
    doc = fanout_doc()
    assert len(doc["tasks"]) == 8
    assert all(t["metadata"]["family"] == "fanout" and t == full[t["id"]] for t in doc["tasks"])


def test_summarize_times() -> None:
    rows = [{"wall_s": 1.0}, {"wall_s": 3.0}]
    assert summarize_times(rows)["mean_s"] == 2.0
    assert summarize_times(rows[:1])["sd_s"] is None
