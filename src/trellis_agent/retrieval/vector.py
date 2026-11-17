"""Tiny in-memory vector index (cosine similarity). Optional: nothing else depends on it."""

from __future__ import annotations

import math

from .lexical import Hit


class VectorIndex:
    def __init__(self) -> None:
        self._vecs: dict[str, list[float]] = {}

    def add(self, doc_id: str, vec: list[float]) -> None:
        self._vecs[doc_id] = vec

    def remove(self, doc_id: str) -> None:
        self._vecs.pop(doc_id, None)

    def search(self, vec: list[float], k: int = 10) -> list[Hit]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return []
        scored = []
        for doc_id, other in self._vecs.items():
            size = math.sqrt(sum(x * x for x in other))
            if size > 0:
                scored.append(
                    Hit(doc_id, sum(a * b for a, b in zip(vec, other, strict=True)) / (norm * size))
                )
        return sorted((h for h in scored if h.score > 0), key=lambda h: -h.score)[:k]
