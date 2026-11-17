"""The working-window size for experiment B, derived from the task set alone.

Rule (fixed before any real result is used, and not adjustable afterwards): the window must hold
the *complete* conversation of every ``fits`` script and be too small for the complete conversation
of every other script. A conversation is measured with the library's own token estimate as

    system prompt + every user turn + REPLY_ALLOWANCE per reply

``REPLY_ALLOWANCE`` is the size the system prompt asks replies to stay within ("at most 20 words",
about 27 tokens, rounded up to 40). The window is the largest ``fits`` conversation rounded up to a
multiple of 10 tokens; ``derive_window`` fails if that is not smaller than every other script."""

from __future__ import annotations

import math
from typing import Any

from experiments.scenarios import SEPARATOR
from trellis_agent.memory.tiers import estimate_tokens
from trellis_agent.types import Message

REPLY_ALLOWANCE = 40
B_SYSTEM_PROMPT = "You are a helpful assistant. Reply in at most 20 words."


def conversation_tokens(script: str) -> int:
    turns = script.split(SEPARATOR)
    users = sum(estimate_tokens([Message.user(t)]) for t in turns)
    return (
        estimate_tokens([Message.system(B_SYSTEM_PROMPT)])
        + users
        + REPLY_ALLOWANCE * (len(turns) - 1)
    )


def derive_window(doc: dict[str, Any]) -> int:
    sizes = {True: [], False: []}  # type: dict[bool, list[int]]
    for task in doc["tasks"]:
        sizes[task["metadata"]["family"] == "fits"].append(conversation_tokens(task["input"]))
    window = math.ceil(max(sizes[True]) / 10) * 10
    if window >= min(sizes[False]):
        raise ValueError(
            f"no window separates fits ({max(sizes[True])}) from overflow ({min(sizes[False])})"
        )
    return window
