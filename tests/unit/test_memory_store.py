from pathlib import Path

from trellis_agent.memory.store import MemoryItem, MemoryStore


def item(id_: str, text: str, tier: str = "semantic", **kw: object) -> MemoryItem:
    base: dict[str, object] = {"importance": 0.5, "created_at": 100.0, "last_access": 100.0}
    base.update(kw)
    return MemoryItem(id=id_, text=text, tier=tier, **base)  # type: ignore[arg-type]


def test_add_get_list_and_count() -> None:
    store = MemoryStore()
    store.add(item("a", "user likes tea", key="drink"))
    store.add(item("b", "met on monday", tier="episodic", session="s1"))
    assert store.get("a") == item("a", "user likes tea", key="drink")
    assert [i.id for i in store.items(tier="episodic")] == ["b"]
    assert {i.id for i in store.items()} == {"a", "b"} and store.count() == 2
    assert store.get("missing") is None


def test_search_finds_live_items_in_both_languages() -> None:
    store = MemoryStore()
    store.add(item("zh", "退款流程:三个工作日内审核"))
    store.add(item("en", "The deadline is Friday"))
    assert [i.id for i, _ in store.search("怎么申请退款?")] == ["zh"]
    hits = store.search("when is the deadline")
    assert [i.id for i, _ in hits] == ["en"] and hits[0][1] > 0


def test_superseding_hides_the_old_item_from_search_and_default_listing_but_keeps_history() -> None:
    store = MemoryStore()
    store.add(item("old", "deadline is Friday", key="deadline"))
    store.add(item("new", "deadline is Tuesday", key="deadline"))
    store.supersede("old", "new")
    assert [i.id for i, _ in store.search("deadline")] == ["new"]
    assert [i.id for i in store.items()] == ["new"]
    assert [i.id for i in store.items(include_superseded=True)] == ["old", "new"]
    old = store.get("old")
    assert old is not None and old.superseded_by == "new"


def test_find_live_by_key_ignores_superseded_items() -> None:
    store = MemoryStore()
    store.add(item("old", "deadline is Friday", key="deadline"))
    store.add(item("new", "deadline is Tuesday", key="deadline"))
    store.supersede("old", "new")
    assert [i.id for i in store.find_by_key("deadline")] == ["new"]
    assert store.find_by_key("nothing") == []


def test_duplicate_detection_normalises_case_and_whitespace() -> None:
    store = MemoryStore()
    store.add(item("a", "Deadline is   Friday"))
    dup = store.find_duplicate("  deadline IS friday ")
    assert dup is not None and dup.id == "a"
    assert store.find_duplicate("deadline is monday") is None


def test_delete_removes_from_listing_and_search() -> None:
    store = MemoryStore()
    store.add(item("a", "refund policy"))
    store.add(item("b", "refund window"))
    store.delete(["a"])
    assert [i.id for i, _ in store.search("refund")] == ["b"] and store.count() == 1


def test_touch_records_access() -> None:
    store = MemoryStore()
    store.add(item("a", "x y z"))
    store.touch(["a"], now=500.0)
    got = store.get("a")
    assert got is not None and (got.last_access, got.access_count) == (500.0, 1)


def test_a_file_backed_store_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "mem.db"
    MemoryStore(path).add(item("a", "退款流程审核"))
    assert [i.id for i, _ in MemoryStore(path).search("退款")] == ["a"]
