from __future__ import annotations

from collections.abc import Sequence

import pytest

from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import ScriptedLLM
from trellis_agent.memory.extract import Extraction, Fact, LLMExtractor, RuleExtractor
from trellis_agent.memory.manager import MemoryConfig, MemoryManager
from trellis_agent.memory.scoring import ScoringConfig
from trellis_agent.memory.store import MemoryItem, MemoryStore
from trellis_agent.types import Message, Usage

DAY = 86_400.0


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FixedFacts:
    """An extractor that returns a prepared list of facts (no model involved)."""

    def __init__(self, *facts: Fact) -> None:
        self.facts = list(facts)

    async def extract(self, turn: Sequence[Message]) -> Extraction:
        return Extraction(list(self.facts))


def manager(extractor=None, clock: Clock | None = None, **cfg: object) -> MemoryManager:
    return MemoryManager(
        MemoryStore(), extractor or RuleExtractor(), MemoryConfig(**cfg), clock=clock or Clock()
    )  # type: ignore[arg-type]


def turn(user: str, assistant: str = "ok") -> list[Message]:
    return [Message.user(user), Message.assistant(assistant)]


# ----------------------------------------------------------------- recall
async def test_facts_from_one_turn_are_recalled_in_a_later_turn_in_either_language() -> None:
    m = manager()
    await m.observe("s", turn("我的截止日期是周五"))
    ctx = await m.build_context("s", "截止日期是哪天?", [])
    assert len(ctx.system) == 1 and ctx.system[0].role == "system"
    assert "周五" in (ctx.system[0].content or "")
    assert ctx.history == []


async def test_nothing_relevant_means_no_memory_message() -> None:
    m = manager()
    await m.observe("s", turn("我的截止日期是周五"))
    assert (await m.build_context("s", "quantum chromodynamics", [])).system == []


async def test_importance_and_recency_reorder_equally_relevant_memories() -> None:
    # Same length and same query terms, so BM25 relevance is equal and only importance/recency
    # differ. Half-life 1 day, now = day 30. retention = 0.5*recency + 0.5*importance:
    #   charlie: used just now   -> 0.5*1.0 + 0.5*0.50 = 0.750
    #   bravo:   important, stale -> 0.5*0.0 + 0.5*0.95 = 0.475
    #   alpha:   ordinary, stale  -> 0.5*0.0 + 0.5*0.50 = 0.250
    clock = Clock(0.0)
    store = MemoryStore()
    store.add(MemoryItem("a", "tea note alpha", "semantic", 0.5, 0.0, 0.0))
    store.add(MemoryItem("b", "tea note bravo", "semantic", 0.95, 0.0, 0.0))
    store.add(MemoryItem("c", "tea note charlie", "semantic", 0.5, 0.0, 0.0))
    cfg = MemoryConfig(scoring=ScoringConfig(half_life_days=1))
    m = MemoryManager(store, RuleExtractor(), cfg, clock=clock)
    clock.now = 30 * DAY
    store.touch(["c"], now=30 * DAY)
    text = (await m.build_context("s", "tea", [])).system[0].content or ""
    order = [line for line in text.splitlines() if line.startswith("- ")]
    assert [line.split()[-1] for line in order] == ["charlie", "bravo", "alpha"]


async def test_recalled_memories_are_marked_as_used() -> None:
    clock = Clock(5000.0)
    m = manager(clock=clock)
    await m.observe("s", turn("我的截止日期是周五"))
    clock.now = 9000.0
    await m.build_context("s", "截止日期", [])
    fact = next(i for i in m.store.items(tier="semantic"))
    assert fact.last_access == 9000.0 and fact.access_count == 1


# ----------------------------------------------------------------- updated facts
async def test_a_newer_value_supersedes_the_old_one_and_only_the_new_one_is_recalled() -> None:
    m = manager()
    await m.observe("s", turn("我的截止日期是周五"))
    result = await m.observe("s", turn("截止日期改到周二了"))
    assert len(result.superseded) == 1
    text = (await m.build_context("s", "截止日期是哪天?", [])).system[0].content or ""
    assert "周二" in text and "周五" not in text
    live = [i.text for i in m.store.items(tier="semantic")]
    assert live == ["截止日期改到周二了"]
    history = {i.text: i for i in m.store.items(tier="semantic", include_superseded=True)}
    assert history["我的截止日期是周五"].superseded_by is not None  # kept as history, marked


# ----------------------------------------------------------------- working memory
async def test_the_working_window_drops_old_messages_but_keeps_the_newest() -> None:
    m = manager(working_token_budget=60)
    history: list[Message] = []
    for i in range(10):
        history += turn(f"question number {i} " + "pad " * 10, f"answer {i} " + "pad " * 10)
    ctx = await m.build_context("s", "latest question", history)
    assert 0 < len(ctx.history) < len(history) and ctx.history == history[-len(ctx.history) :]
    assert all(msg.role != "tool" for msg in ctx.history[:1])


# ----------------------------------------------------------------- write admission
async def test_duplicate_facts_are_rejected_and_the_reason_is_recorded() -> None:
    m = manager(FixedFacts(Fact("User likes tea")))
    first = await m.observe("s", turn("hello there"))
    second = await m.observe("s", turn("hello again"))
    assert len(first.written) == 1 and second.written == []
    ((text, reason),) = second.rejected
    assert text == "User likes tea" and "duplicate" in reason
    event = next(e for e in second.overhead.events if e[0] == "memory_reject")
    assert event[1]["reason"] == reason


async def test_the_per_turn_write_limit_rejects_the_excess_with_a_reason() -> None:
    facts = [Fact(f"distinct fact number {i}") for i in range(7)]
    m = manager(FixedFacts(*facts), max_writes_per_turn=5)
    result = await m.observe("s", turn("lots of facts"))
    assert len(result.written) == 5 and len(result.rejected) == 2
    assert all("limit" in reason for _, reason in result.rejected)
    assert len(m.store.items(tier="semantic")) == 5


async def test_every_turn_is_also_kept_as_an_episode() -> None:
    m = manager(FixedFacts())
    await m.observe("s1", turn("what a day", "indeed"))
    (episode,) = m.store.items(tier="episodic")
    assert episode.session == "s1" and "what a day" in episode.text and "indeed" in episode.text


# ----------------------------------------------------------------- the cost of remembering
async def test_llm_extraction_is_billed_and_visible_as_a_model_request() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 0.0)}))
    llm = ScriptedLLM(
        [
            Message.assistant(
                '{"facts": [{"text": "User likes tea", "key": "drink", "importance": 0.6}]}'
            )
        ],
        usage_per_call=Usage(1_000_000, 0),
        meter=meter,
    )
    m = manager(LLMExtractor(llm))
    result = await m.observe("s", turn("I like tea"))
    ov = result.overhead
    assert ov.usage == Usage(1_000_000, 0) and ov.costs == [pytest.approx(1.0)]
    request = [e for e in ov.events if e[0] == "model_request"]
    assert request[0][1]["phase"] == "memory_extract" and request[0][1][
        "cost_usd"
    ] == pytest.approx(1.0)
    assert meter.by_tag == {"memory_extract": pytest.approx(1.0)}


async def test_rule_extraction_costs_nothing_and_adds_no_model_request() -> None:
    result = await manager().observe("s", turn("我的截止日期是周五"))
    assert result.overhead.usage == Usage() and not any(
        e[0] == "model_request" for e in result.overhead.events
    )


async def test_running_out_of_budget_while_extracting_is_reported_not_raised() -> None:
    meter = CostMeter(PriceTable({"mock": ModelPrice(1.0, 0.0)}), budget_usd=0.1)
    llm = ScriptedLLM(
        [Message.assistant('{"facts": []}')], usage_per_call=Usage(1_000_000, 0), meter=meter
    )
    result = await manager(LLMExtractor(llm)).observe("s", turn("hello"))
    assert result.written == [] and result.overhead.usage == Usage(1_000_000, 0)
    assert any(
        e[0] == "error" and e[1]["category"] == "memory_budget" for e in result.overhead.events
    )


# ----------------------------------------------------------------- upkeep with an injected clock
async def test_forget_and_consolidate_use_the_injected_clock() -> None:
    clock = Clock(0.0)
    m = manager(FixedFacts(), clock, scoring=ScoringConfig(half_life_days=1), forget_threshold=0.1)
    for i in range(4):
        await m.observe("s", turn(f"chat number {i}"))
    clock.now = 3 * DAY
    report = await m.consolidate()
    assert report.groups == 1 and report.merged == 4
    assert [i.tier for i in m.store.items()] == ["semantic"]
    clock.now = 60 * DAY  # the summary keeps the episodes' low importance (0.1) and is now ancient
    removed = m.forget()
    assert len(removed) == 1 and m.store.count() == 0
