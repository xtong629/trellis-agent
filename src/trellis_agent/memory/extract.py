"""Turn a conversation turn into durable facts (never store raw chat lines as memories)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..jsonutil import extract_json
from ..llm import LLMClient
from ..types import Message, Usage


@dataclass(frozen=True)
class Fact:
    text: str
    key: str | None = None  # facts with the same key describe the same thing (newer wins)
    importance: float = 0.5


@dataclass
class Extraction:
    facts: list[Fact]
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = 0.0


class FactExtractor(Protocol):
    async def extract(self, turn: Sequence[Message]) -> Extraction: ...


# Full-width sentence punctuation is written as \u escapes: 3002, ff01, ff1f, ff1b.
_SENTENCE = re.compile(r"[^.\u3002!?\uff01\uff1f;\uff1b\n]+")
_QUESTION = re.compile(r"[?\uff1f]|哪|什么|吗|几|\b(?:what|when|where|who|how|why)\b", re.I)
_VERB = r"\bis\b|\bare\b|是|改到|改为|改成|变成|moved to|changed to"
_STATEMENT = re.compile(rf"^(?:my\s+|the\s+|我的)?(.{{1,20}}?)\s*(?:{_VERB})\s*(.+)$", re.I)


class RuleExtractor:
    """Free and deterministic: "X is Y" / "X是Y" / "X改到Y" statements in user messages."""

    async def extract(self, turn: Sequence[Message]) -> Extraction:
        facts: list[Fact] = []
        for message in turn:
            if message.role != "user":
                continue
            for raw in _SENTENCE.findall(message.content or ""):
                sentence = raw.strip()
                match = _STATEMENT.match(sentence)
                if match and not _QUESTION.search(sentence):
                    facts.append(Fact(sentence, match.group(1).strip().lower(), 0.7))
        return Extraction(facts)


_SYSTEM = (
    "Extract durable facts about the user or task from this conversation turn. Reply with ONLY "
    'JSON: {"facts": [{"text": "...", "key": "<short id, null if none>", "importance": 0-1}]}. '
    "Reuse the same key when a fact updates an earlier one. Return an empty list if nothing lasts."
)


class LLMExtractor:
    def __init__(self, llm: LLMClient, tag: str = "memory_extract") -> None:
        self.llm = llm
        self.tag = tag

    async def extract(self, turn: Sequence[Message]) -> Extraction:
        convo = "\n".join(f"{m.role}: {m.content}" for m in turn if m.role != "tool" and m.content)
        reply = await self.llm.chat(
            [Message.system(_SYSTEM), Message.user(convo)], temperature=0.0, tag=self.tag
        )
        data = extract_json(reply.message.content or "")
        raw_facts = data.get("facts", []) if data else []
        facts: list[Fact] = []
        for raw in raw_facts if isinstance(raw_facts, list) else []:
            text = raw.get("text") if isinstance(raw, dict) else None
            if not isinstance(text, str) or not text.strip():
                continue
            key = raw.get("key") if isinstance(raw.get("key"), str) else None
            weight = raw.get("importance", 0.5)
            weight = (
                0.5 if isinstance(weight, bool) or not isinstance(weight, int | float) else weight
            )
            facts.append(Fact(text.strip(), key, min(1.0, max(0.0, float(weight)))))
        return Extraction(facts, reply.usage, reply.cost_usd)
