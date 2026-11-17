"""Consolidation (old episodes -> a semantic summary) and forgetting (drop low-retention items)."""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from .scoring import ScoringConfig, retention
from .store import MemoryItem, MemoryStore

_DAY = 86_400.0


def default_summary(texts: list[str]) -> str:
    return " | ".join(t[:80] for t in texts)


@dataclass
class ConsolidationReport:
    groups: int = 0
    merged: int = 0
    created: list[str] = field(default_factory=list)


def consolidate(
    store: MemoryStore,
    *,
    now: float,
    min_age_days: float = 1.0,
    min_group: int = 3,
    summarize: Callable[[list[str]], str] = default_summary,
    new_id: Callable[[], str] = lambda: uuid.uuid4().hex,
) -> ConsolidationReport:
    """Merge each session's old episodes (at least ``min_group`` of them) into one semantic item."""
    old = [e for e in store.items(tier="episodic") if now - e.created_at >= min_age_days * _DAY]
    by_session: dict[str, list[MemoryItem]] = defaultdict(list)
    for episode in old:
        by_session[episode.session].append(episode)
    report = ConsolidationReport()
    for session, group in by_session.items():
        if len(group) < min_group:
            continue
        top = max(e.importance for e in group)
        summary = MemoryItem(
            new_id(), summarize([e.text for e in group]), "semantic", top, now, now, session
        )
        store.delete(e.id for e in group)
        store.add(summary)
        report.groups, report.merged = report.groups + 1, report.merged + len(group)
        report.created.append(summary.id)
    return report


def forget(
    store: MemoryStore, cfg: ScoringConfig, *, now: float, threshold: float, max_items: int
) -> list[str]:
    """Delete items whose retention is below ``threshold``, then the lowest-scoring ones beyond
    ``max_items``. Returns the removed ids."""
    items = store.items(include_superseded=True)
    scored = sorted((retention(i.importance, now - i.last_access, cfg), i.id) for i in items)
    removed = [item_id for score, item_id in scored if score < threshold]
    survivors = [item_id for score, item_id in scored if score >= threshold]
    removed += survivors[: max(0, len(survivors) - max_items)]  # lowest first
    store.delete(removed)
    return removed
