"""Working memory: the part of the conversation that fits the token budget."""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..types import Message


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Rough chars/4 estimate per message (additive), including tool-call arguments."""
    total = 0
    for m in messages:
        chars = len(m.content or "")
        chars += sum(len(c.name) + len(json.dumps(c.arguments)) for c in m.tool_calls)
        total += chars // 4 + 4
    return total


class WorkingMemory:
    def __init__(self, token_budget: int) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget

    def fit(self, messages: Sequence[Message]) -> tuple[list[Message], list[Message]]:
        """Keep system messages and the newest messages that fit; return (kept, dropped)."""
        system = [m for m in messages if m.role == "system"]
        rest = [m for m in messages if m.role != "system"]
        room = self.token_budget - estimate_tokens(system)
        kept: list[Message] = []
        used = 0
        for m in reversed(rest):
            cost = estimate_tokens([m])
            if kept and used + cost > room:
                break  # the newest message is always kept, even if it alone is too big
            kept.insert(0, m)
            used += cost
        while kept and kept[0].role == "tool":
            kept.pop(0)  # a tool result without its tool call would be rejected by providers
        return [*system, *kept], rest[: len(rest) - len(kept)]
