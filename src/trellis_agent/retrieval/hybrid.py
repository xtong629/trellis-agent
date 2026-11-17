"""Hybrid retrieval: BM25 + vectors fused with Reciprocal Rank Fusion (RRF).

With no vector index it is plain BM25, so this module can be dropped without losing anything else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..cost import CostMeter
from ..types import Usage
from .lexical import Hit, LexicalIndex

if TYPE_CHECKING:
    from ..llm import Embedder
    from .vector import VectorIndex


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """score(d) = sum over rankings of 1 / (k + rank), rank starting at 1. Ranks, not scores."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        lexical: LexicalIndex,
        vector: VectorIndex | None = None,
        embedder: Embedder | None = None,
        *,
        rrf_k: int = 60,
        meter: CostMeter | None = None,
        model: str = "embedder",
    ) -> None:
        self.lexical, self.vector, self.embedder = lexical, vector, embedder
        self.rrf_k, self.meter, self.model = rrf_k, meter, model
        self._pending: list[tuple[Usage, float | None]] = []

    async def _embed(self, text: str) -> list[float]:
        assert self.embedder is not None
        (vec,) = await self.embedder.embed([text])
        usage = Usage(max(1, len(text) // 4), estimated=True)  # embedders report no usage
        cost = self.meter.record(self.model, usage, tag="embed") if self.meter else None
        self._pending.append((usage, cost))
        return vec

    async def index(self, item_id: str, text: str) -> None:
        """Add the vector side (the BM25 side is indexed by the store)."""
        if self.vector is not None and self.embedder is not None:
            self.vector.add(item_id, await self._embed(text))

    async def search(self, query: str, k: int) -> list[Hit]:
        lexical = self.lexical.search(query, k)
        if self.vector is None or self.embedder is None:
            return lexical
        semantic = self.vector.search(await self._embed(query), k)
        fused = rrf([[h.doc_id for h in lexical], [h.doc_id for h in semantic]], self.rrf_k)
        return [Hit(d, s) for d, s in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]

    def remove(self, item_id: str) -> None:
        if self.vector is not None:
            self.vector.remove(item_id)

    def drain(self) -> list[tuple[Usage, float | None]]:
        """Embedding calls made since the last drain, for cost accounting."""
        pending, self._pending = self._pending, []
        return pending
