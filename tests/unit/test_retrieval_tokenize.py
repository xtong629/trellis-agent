from pathlib import Path

import pytest

from trellis_agent.retrieval import tokenize
from trellis_agent.retrieval.tokenize import Tokenizer


def test_english_is_lowercased_and_stripped_of_punctuation() -> None:
    got = Tokenizer().tokens("How does the Scheduler handle retries?!")
    assert got == ["how", "does", "the", "scheduler", "handle", "retries"]


def test_chinese_uses_search_mode_so_sub_words_are_indexed_too() -> None:
    got = Tokenizer().tokens("退款流程")
    assert "退款" in got and "流程" in got  # sub-words, not only the whole phrase


def test_mixed_text_keeps_codes_and_chinese_words() -> None:
    got = Tokenizer().tokens("订单 RF-20931 的退款")
    assert {"rf", "20931", "退款", "订单"} <= set(got)


@pytest.mark.parametrize("text", ["", "   ", "?!.,;:", "—— ……"])
def test_nothing_to_index_gives_no_tokens(text: str) -> None:
    assert Tokenizer().tokens(text) == []


def test_jieba_is_loaded_lazily_and_only_when_chinese_appears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str | None] = []
    real = tokenize._load_segmenter

    def counting(user_dict: str | None) -> object:
        loads.append(user_dict)
        return real(user_dict)

    monkeypatch.setattr(tokenize, "_load_segmenter", counting)
    t = Tokenizer()
    t.tokens("english only text")
    assert loads == []  # constructing and using it for English never touches jieba
    t.tokens("退款")
    t.tokens("退款流程")
    assert len(loads) == 1  # loaded once, then reused


def test_a_user_dictionary_keeps_domain_terms_whole(tmp_path: Path) -> None:
    dictionary = tmp_path / "terms.txt"
    dictionary.write_text("云账户 1000000 n\n", encoding="utf-8")
    assert "云账户" in Tokenizer(user_dict=dictionary).tokens("开通云账户")
