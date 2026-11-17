"""Tokenizer for the full-text index: lowercase words; jieba search mode for Chinese (lazy)."""

from __future__ import annotations

import functools
import logging
import re
import warnings
from pathlib import Path
from typing import Any

_PIECE = re.compile(r"[一-鿿]+|[^\W_一-鿿]+")
_CJK = re.compile(r"[一-鿿]")


@functools.lru_cache(maxsize=4)
def _load_segmenter(user_dict: str | None) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # some setuptools versions warn when jieba is imported
        import jieba
    jieba.setLogLevel(logging.WARNING)
    segmenter = jieba.Tokenizer()  # a private instance: a user dictionary never leaks globally
    if user_dict:
        with open(user_dict, "rb") as handle:  # jieba leaks the handle if given a path
            segmenter.load_userdict(handle)
    return segmenter


class Tokenizer:
    def __init__(self, *, user_dict: str | Path | None = None) -> None:
        self._user_dict = None if user_dict is None else str(user_dict)
        self._segmenter: Any = None

    def tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for piece in _PIECE.findall(text):
            if _CJK.match(piece):
                if self._segmenter is None:
                    self._segmenter = _load_segmenter(self._user_dict)
                out += [w for w in self._segmenter.cut_for_search(piece) if w.strip()]
            else:
                out.append(piece.lower())
        return out
