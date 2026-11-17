"""MemoryManager: recall before a turn, remember after it, consolidate and forget over time."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..cost import BudgetExceeded
from ..retrieval.lexical import Hit
from ..types import Message, Usage
from .admission import Admission
from .consolidate import ConsolidationReport, consolidate, forget
from .extract import FactExtractor
from .scoring import ScoringConfig, retrieval_score
from .store import MemoryItem, MemoryStore
from .tiers import WorkingMemory


@dataclass(frozen=True)
class MemoryConfig:
    working_token_budget: int = 2000
    retrieve_k: int = 5
    max_writes_per_turn: int = 5
    forget_threshold: float = 0.1
    max_items: int = 1000
    consolidate_min_age_days: float = 1.0
    consolidate_min_group: int = 3
    episode_importance: float = 0.1  # low on purpose: raw chat should fade unless consolidated
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


@dataclass
class Overhead:
    """What remembering cost: model usage/cost plus trace events (type, data) to emit."""

    usage: Usage = field(default_factory=Usage)
    costs: list[float | None] = field(default_factory=list)
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def call(self, phase: str, usage: Usage, cost: float | None, **extra: Any) -> None:
        self.usage = self.usage + usage
        self.costs.append(cost)
        data = {"phase": phase, "usage": usage.to_dict(), "cost_usd": cost, "latency_ms": 0.0}
        self.events.append(("model_request", {**data, **extra}))

    def note(self, kind: str, **data: Any) -> None:
        self.events.append((kind, data))


@dataclass
class ContextResult:
    system: list[Message]  # the "relevant memories" note (empty if nothing relevant)
    history: list[Message]  # the part of the conversation that fits the working window
    overhead: Overhead


@dataclass
class ObserveResult:
    overhead: Overhead
    written: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (fact text, reason)


class Retriever(Protocol):
    """Optional secondary index (e.g. vectors). Absent: BM25 only."""

    async def index(self, item_id: str, text: str) -> None: ...
    async def search(self, query: str, k: int) -> list[Hit]: ...
    def remove(self, item_id: str) -> None: ...


class MemoryManager:
    def __init__(
        self,
        store: MemoryStore,
        extractor: FactExtractor,
        config: MemoryConfig | None = None,
        *,
        retriever: Retriever | None = None,
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self.store, self.extractor, self.retriever = store, extractor, retriever
        self.config = config or MemoryConfig()
        self.clock, self.new_id = clock, new_id
        self.admission = Admission(store, max_writes_per_turn=self.config.max_writes_per_turn)
        self.working = WorkingMemory(self.config.working_token_budget)
        if retriever is not None:
            store.on_remove = retriever.remove

    # ------------------------------------------------------------ before a turn
    async def build_context(
        self, session_id: str, query: str, history: Sequence[Message]
    ) -> ContextResult:
        cfg, now, ov = self.config, self.clock(), Overhead()
        fitted, _ = self.working.fit([*history, Message.user(query)])
        ranked = sorted(
            (
                (retrieval_score(rel, i.importance, now - i.last_access, cfg.scoring), i)
                for i, rel in await self._search(query, ov)
            ),
            key=lambda pair: -pair[0],
        )[: cfg.retrieve_k]
        ov.note("memory_retrieve", query=query, ids=[i.id for _, i in ranked])
        note: list[Message] = []
        if ranked:
            self.store.touch([i.id for _, i in ranked], now)
            lines = "\n".join(f"- {i.text}" for _, i in ranked)
            note = [
                Message.system(f"Relevant memories from earlier (most relevant first):\n{lines}")
            ]
        return ContextResult(note, fitted[:-1], ov)

    async def _search(self, query: str, ov: Overhead) -> list[tuple[MemoryItem, float]]:
        k = self.config.retrieve_k * 3
        if self.retriever is not None:
            hits = await self.retriever.search(query, k)
        else:
            hits = self.store.index.search(query, k)
        self._drain(ov)
        found = [(self.store.get(h.doc_id), h.score) for h in hits]
        live = [(i, s) for i, s in found if i and i.tier == "semantic" and not i.superseded_by]
        best = max((s for _, s in live), default=0.0)
        return [(i, s / best if best > 0 else 0.0) for i, s in live]

    # ------------------------------------------------------------ after a turn
    async def observe(self, session_id: str, turn: Sequence[Message]) -> ObserveResult:
        ov, now = Overhead(), self.clock()
        result = ObserveResult(ov)
        said = " | ".join(f"{m.role}: {m.content}" for m in turn if m.role != "tool" and m.content)
        if said:
            episode = MemoryItem(
                self.new_id(),
                said[:300],
                "episodic",
                self.config.episode_importance,
                now,
                now,
                session_id,
            )
            self.store.add(episode)
        try:
            extraction = await self.extractor.extract(turn)
        except BudgetExceeded as exc:
            if exc.response is not None:
                ov.call(
                    "memory_extract", exc.response.usage, exc.response.cost_usd, over_budget=True
                )
            ov.note("error", category="memory_budget", message=str(exc))
            return result
        if extraction.usage.total_tokens:
            ov.call("memory_extract", extraction.usage, extraction.cost_usd)
        for fact in extraction.facts:
            reason = self.admission.check(fact, len(result.written))
            if reason is not None:
                result.rejected.append((fact.text, reason))
                ov.note("memory_reject", text=fact.text, reason=reason)
                continue
            item = MemoryItem(
                self.new_id(),
                fact.text,
                "semantic",
                fact.importance,
                now,
                now,
                session_id,
                fact.key,
            )
            older = self.store.find_by_key(fact.key) if fact.key else []
            self.store.add(item)
            for old in older:  # a newer value for the same key replaces the old one
                self.store.supersede(old.id, item.id)
                result.superseded.append(old.id)
            result.written.append(item.id)
            ov.note("memory_write", id=item.id, key=fact.key, superseded=[o.id for o in older])
            await self._index(item, ov)
        return result

    # ------------------------------------------------------------ upkeep
    async def consolidate(self) -> ConsolidationReport:
        cfg = self.config
        report = consolidate(
            self.store,
            now=self.clock(),
            min_age_days=cfg.consolidate_min_age_days,
            min_group=cfg.consolidate_min_group,
            new_id=self.new_id,
        )
        for item_id in report.created:
            summary = self.store.get(item_id)
            if summary is not None:
                await self._index(summary, Overhead())  # embedding cost surfaces on the next drain
        return report

    def forget(self) -> list[str]:
        cfg = self.config
        return forget(
            self.store,
            cfg.scoring,
            now=self.clock(),
            threshold=cfg.forget_threshold,
            max_items=cfg.max_items,
        )

    # ------------------------------------------------------------ secondary index plumbing
    async def _index(self, item: MemoryItem, ov: Overhead) -> None:
        if self.retriever is not None:
            await self.retriever.index(item.id, item.text)
            self._drain(ov)

    def _drain(self, ov: Overhead) -> None:
        drain = getattr(self.retriever, "drain", None)
        for usage, cost in drain() if drain else []:
            ov.call("memory_embed", usage, cost)
