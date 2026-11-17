"""Forgetting and consolidation, with an injected clock (no real time passes)."""

from trellis_agent.memory.consolidate import consolidate, forget
from trellis_agent.memory.scoring import ScoringConfig, retention
from trellis_agent.memory.store import MemoryItem, MemoryStore

DAY = 86_400.0
CFG = ScoringConfig(half_life_days=1.0, w_recency=0.5, w_importance=0.5)  # 1-day half-life


def put(store: MemoryStore, id_: str, importance: float, at: float = 0.0, **kw: object) -> None:
    tier = str(kw.pop("tier", "semantic"))
    store.add(MemoryItem(id_, f"text of {id_}", tier, importance, at, at, **kw))  # type: ignore[arg-type]


# ----------------------------------------------------------------- forgetting
def test_a_trivial_memory_is_kept_while_fresh_and_forgotten_once_it_has_decayed() -> None:
    store = MemoryStore()
    put(store, "trivial", importance=0.1)
    assert forget(store, CFG, now=0.0, threshold=0.1, max_items=100) == []  # fresh: 0.5 + 0.05
    # 5 days = 5 half-lives: recency 1/32 -> 0.5/32 + 0.05 = 0.0656 < 0.1
    assert retention(0.1, 5 * DAY, CFG) < 0.1
    assert forget(store, CFG, now=5 * DAY, threshold=0.1, max_items=100) == ["trivial"]
    assert store.get("trivial") is None


def test_an_important_memory_survives_any_amount_of_time() -> None:
    store = MemoryStore()
    put(store, "allergy", importance=0.9)
    assert forget(store, CFG, now=1000 * DAY, threshold=0.1, max_items=100) == []
    assert store.get("allergy") is not None  # retention floor = 0.5 * 0.9 = 0.45


def test_only_the_low_scoring_items_go_and_the_rest_stay() -> None:
    store = MemoryStore()
    put(store, "keep", importance=0.8)
    put(store, "drop_a", importance=0.05)
    put(store, "drop_b", importance=0.0)
    removed = forget(store, CFG, now=10 * DAY, threshold=0.1, max_items=100)
    assert sorted(removed) == ["drop_a", "drop_b"]
    assert [i.id for i in store.items()] == ["keep"]


def test_recent_access_resets_the_clock() -> None:
    store = MemoryStore()
    put(store, "used", importance=0.1)
    put(store, "ignored", importance=0.1)
    store.touch(["used"], now=9.5 * DAY)  # recalled recently
    removed = forget(store, CFG, now=10 * DAY, threshold=0.1, max_items=100)
    assert removed == ["ignored"]


def test_the_item_cap_evicts_the_lowest_scores_first() -> None:
    store = MemoryStore()
    for i, imp in enumerate([0.9, 0.7, 0.5, 0.3]):
        put(store, f"m{i}", importance=imp)
    removed = forget(store, CFG, now=0.0, threshold=0.0, max_items=2)
    assert sorted(removed) == ["m2", "m3"] and {i.id for i in store.items()} == {"m0", "m1"}


def test_superseded_history_is_forgotten_by_the_same_rule() -> None:
    store = MemoryStore()
    put(store, "old", importance=0.1, key="k")
    put(store, "new", importance=0.9, key="k")
    store.supersede("old", "new")
    assert forget(store, CFG, now=10 * DAY, threshold=0.1, max_items=100) == ["old"]


def test_forgetting_an_empty_store_is_a_noop() -> None:
    assert forget(MemoryStore(), CFG, now=0.0, threshold=0.1, max_items=10) == []


# ----------------------------------------------------------------- consolidation
def episodes(store: MemoryStore, session: str, n: int, at: float, importance: float = 0.3) -> None:
    for i in range(n):
        put(store, f"{session}-{i}", importance + i * 0.01, at, tier="episodic", session=session)


def test_old_episodes_of_one_session_become_one_semantic_summary() -> None:
    store = MemoryStore()
    episodes(store, "s1", 4, at=0.0)
    report = consolidate(store, now=3 * DAY, min_age_days=1.0, min_group=3, new_id=lambda: "sum1")
    assert report.groups == 1 and report.merged == 4 and report.created == ["sum1"]
    (summary,) = store.items()
    assert summary.tier == "semantic" and summary.session == "s1" and summary.created_at == 3 * DAY
    assert summary.importance == max(
        0.3 + i * 0.01 for i in range(4)
    )  # keeps the highest importance
    assert all(f"text of s1-{i}" in summary.text for i in range(4))


def test_young_episodes_small_groups_and_semantic_items_are_left_alone() -> None:
    store = MemoryStore()
    episodes(store, "young", 5, at=2.9 * DAY)  # too recent
    episodes(store, "few", 2, at=0.0)  # below min_group
    put(store, "fact", 0.5, at=0.0)  # already semantic
    report = consolidate(store, now=3 * DAY, min_age_days=1.0, min_group=3)
    assert report.groups == 0 and store.count() == 8


def test_each_session_is_summarised_separately() -> None:
    store = MemoryStore()
    episodes(store, "a", 3, at=0.0)
    episodes(store, "b", 3, at=0.0)
    ids = iter(["x", "y"])
    report = consolidate(
        store, now=3 * DAY, min_age_days=1.0, min_group=3, new_id=lambda: next(ids)
    )
    assert report.groups == 2 and {i.session for i in store.items()} == {"a", "b"}
    assert store.count() == 2


def test_a_custom_summariser_can_be_plugged_in() -> None:
    store = MemoryStore()
    episodes(store, "s", 3, at=0.0)
    consolidate(
        store, now=3 * DAY, min_age_days=1.0, min_group=3, summarize=lambda t: f"{len(t)} episodes"
    )
    assert store.items()[0].text == "3 episodes"
