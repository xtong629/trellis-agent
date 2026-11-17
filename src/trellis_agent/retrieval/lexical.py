"""BM25 retrieval on SQLite FTS5 (porter stemming). Query terms are OR-ed, best matches first."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .tokenize import Tokenizer


class RetrievalError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float  # higher is better


class LexicalIndex:
    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        tokenizer: Tokenizer | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self.conn = conn or sqlite3.connect(str(path))
        self.tokenizer = tokenizer or Tokenizer()
        try:
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
                "text, doc_id UNINDEXED, tokenize='porter unicode61')"
            )
        except sqlite3.OperationalError as exc:
            raise RetrievalError(
                f"SQLite FTS5 with the porter tokenizer is required but unavailable ({exc}); "
                f"this SQLite is version {sqlite3.sqlite_version}"
            ) from exc

    def add(self, doc_id: str, text: str) -> None:
        self.remove(doc_id)
        indexed = " ".join(self.tokenizer.tokens(text))
        self.conn.execute("INSERT INTO docs(text, doc_id) VALUES (?, ?)", (indexed, doc_id))
        self.conn.commit()

    def remove(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
        self.conn.commit()

    def search(self, query: str, k: int = 10) -> list[Hit]:
        terms = list(dict.fromkeys(self.tokenizer.tokens(query)))
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms)  # tokens are alphanumeric: safe to quote
        rows = self.conn.execute(
            "SELECT doc_id, bm25(docs) FROM docs WHERE docs MATCH ? "
            "ORDER BY bm25(docs), rowid LIMIT ?",
            (match, k),
        ).fetchall()
        return [Hit(doc_id, -float(score)) for doc_id, score in rows]
