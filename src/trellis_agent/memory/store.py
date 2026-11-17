"""SQLite store for episodic and semantic memories, indexed by the BM25 index."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Literal

from ..retrieval.lexical import LexicalIndex
from ..retrieval.tokenize import Tokenizer

Tier = Literal["episodic", "semantic"]


@dataclass
class MemoryItem:
    id: str
    text: str
    tier: Tier
    importance: float = 0.5
    created_at: float = 0.0
    last_access: float = 0.0
    session: str = ""
    key: str | None = None  # same key = same fact (a newer value supersedes the older one)
    superseded_by: str | None = None
    access_count: int = 0


_COLS = "id,text,tier,importance,created_at,last_access,session,key,superseded_by,access_count"


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


class MemoryStore:
    def __init__(
        self, path: str | Path = ":memory:", *, tokenizer: Tokenizer | None = None
    ) -> None:
        self.conn = sqlite3.connect(str(path))
        self.index = LexicalIndex(conn=self.conn, tokenizer=tokenizer)
        self.on_remove: Callable[[str], None] | None = None  # lets a secondary index stay in sync
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS items(id TEXT PRIMARY KEY, text TEXT, tier TEXT, "
            "importance REAL, created_at REAL, last_access REAL, session TEXT, key TEXT, "
            "superseded_by TEXT, access_count INTEGER, norm TEXT)"
        )
        self.conn.commit()

    def _rows(self, where: str, args: tuple[object, ...] = ()) -> list[MemoryItem]:
        sql = f"SELECT {_COLS} FROM items WHERE {where} ORDER BY created_at, rowid"
        return [MemoryItem(*row) for row in self.conn.execute(sql, args).fetchall()]

    def add(self, item: MemoryItem) -> None:
        self.conn.execute(
            f"INSERT INTO items({_COLS}, norm) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (*astuple(item), normalise(item.text)),
        )
        self.conn.commit()
        if item.superseded_by is None:
            self.index.add(item.id, item.text)

    def get(self, item_id: str) -> MemoryItem | None:
        rows = self._rows("id = ?", (item_id,))
        return rows[0] if rows else None

    def items(self, tier: Tier | None = None, include_superseded: bool = False) -> list[MemoryItem]:
        clauses = (
            ["1"]
            + (["tier = ?"] if tier else [])
            + ([] if include_superseded else ["superseded_by IS NULL"])
        )
        return self._rows(" AND ".join(clauses), (tier,) if tier else ())

    def count(self) -> int:
        return len(self.items())

    def find_by_key(self, key: str) -> list[MemoryItem]:
        return self._rows("key = ? AND superseded_by IS NULL", (key,))

    def find_duplicate(self, text: str) -> MemoryItem | None:
        rows = self._rows("norm = ? AND superseded_by IS NULL", (normalise(text),))
        return rows[0] if rows else None

    def search(self, query: str, k: int = 10) -> list[tuple[MemoryItem, float]]:
        out: list[tuple[MemoryItem, float]] = []
        for hit in self.index.search(query, k):
            found = self.get(hit.doc_id)
            if found is not None and found.superseded_by is None:
                out.append((found, hit.score))
        return out

    def touch(self, ids: Iterable[str], now: float) -> None:
        for item_id in ids:
            self.conn.execute(
                "UPDATE items SET last_access = ?, access_count = access_count + 1 WHERE id = ?",
                (now, item_id),
            )
        self.conn.commit()

    def supersede(self, old_id: str, new_id: str) -> None:
        self.conn.execute("UPDATE items SET superseded_by = ? WHERE id = ?", (new_id, old_id))
        self.conn.commit()
        self.index.remove(old_id)  # history is kept, but it can no longer be retrieved
        if self.on_remove:
            self.on_remove(old_id)

    def delete(self, ids: Iterable[str]) -> None:
        for item_id in ids:
            self.conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
            self.index.remove(item_id)
            if self.on_remove:
                self.on_remove(item_id)
        self.conn.commit()
