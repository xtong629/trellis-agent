"""Pulling JSON out of model replies that may be fenced, chatty, or both."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, *, kinds: tuple[type, ...] = (dict,)) -> Any | None:
    """First JSON value of one of ``kinds`` (default: object) found in ``text``, else None."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[{\[]", candidate):
        try:
            value, _ = decoder.raw_decode(candidate[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, kinds):
            return value
    return None
