"""Write admission: decide whether a fact may enter long-term memory."""

from __future__ import annotations

from .extract import Fact
from .store import MemoryStore


class Admission:
    def __init__(
        self, store: MemoryStore, *, max_writes_per_turn: int = 5, min_length: int = 2
    ) -> None:
        self.store = store
        self.max_writes_per_turn = max_writes_per_turn
        self.min_length = min_length

    def check(self, fact: Fact, written_this_turn: int) -> str | None:
        """None if admitted, otherwise the reason for rejecting it."""
        if len(fact.text.strip()) < self.min_length:
            return f"too short (fewer than {self.min_length} characters)"
        if written_this_turn >= self.max_writes_per_turn:
            return f"write limit reached ({self.max_writes_per_turn} per turn)"
        duplicate = self.store.find_duplicate(fact.text)
        if duplicate is not None:
            return f"duplicate of existing memory {duplicate.id}"
        return None
