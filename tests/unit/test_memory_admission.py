from trellis_agent.memory.admission import Admission
from trellis_agent.memory.extract import Fact
from trellis_agent.memory.store import MemoryItem, MemoryStore


def store_with(*texts: str) -> MemoryStore:
    store = MemoryStore()
    for i, text in enumerate(texts):
        store.add(MemoryItem(f"m{i}", text, "semantic"))
    return store


def test_a_new_fact_is_admitted() -> None:
    assert Admission(store_with()).check(Fact("User likes tea"), written_this_turn=0) is None


def test_duplicates_are_rejected_with_the_reason_and_the_existing_id() -> None:
    adm = Admission(store_with("User likes   TEA"))
    reason = adm.check(Fact("user likes tea"), written_this_turn=0)
    assert reason is not None and "duplicate" in reason and "m0" in reason


def test_the_per_turn_write_limit_is_enforced() -> None:
    adm = Admission(store_with(), max_writes_per_turn=2)
    assert adm.check(Fact("a fact"), written_this_turn=1) is None
    reason = adm.check(Fact("another fact"), written_this_turn=2)
    assert reason is not None and "limit" in reason and "2" in reason


def test_too_short_facts_are_rejected() -> None:
    reason = Admission(store_with(), min_length=4).check(Fact("ab"), written_this_turn=0)
    assert reason is not None and "short" in reason


def test_a_changed_value_under_the_same_key_is_not_a_duplicate() -> None:
    adm = Admission(store_with("deadline is Friday"))
    assert adm.check(Fact("deadline is Tuesday", key="deadline"), written_this_turn=0) is None
