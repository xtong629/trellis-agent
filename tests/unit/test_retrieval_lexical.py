"""Lexical (BM25) retrieval. The regression cases come from the stage-2 analysis of LightAgent,
whose FTS search AND-ed whitespace-split terms and had no Chinese segmentation or stemming."""

from pathlib import Path
from typing import Any

import pytest

from trellis_agent.retrieval.lexical import LexicalIndex, RetrievalError

DOCS = {
    "en": "LightDAG schedules tasks with retries and verification.",
    "zh": "退款流程:用户提交申请后三个工作日内审核。",
    "weather": "The weather is sunny today in Oslo.",
    "phone": "苹果手机的电池续航很长。",
}


def index() -> LexicalIndex:
    idx = LexicalIndex()
    for doc_id, text in DOCS.items():
        idx.add(doc_id, text)
    return idx


def top(idx: LexicalIndex, query: str) -> str | None:
    hits = idx.search(query)
    return hits[0].doc_id if hits else None


@pytest.mark.parametrize(
    "query",
    ["退款", "怎么申请退款?", "退款流程 审核", "退款流程", "审核"],
)
def test_chinese_queries_that_the_old_search_missed_now_hit_the_refund_document(query: str) -> None:
    assert top(index(), query) == "zh"


@pytest.mark.parametrize(
    "query",
    [
        "retries",
        "how does the scheduler handle retries?",  # natural-language question, OR semantics
        "task retries verification",  # "task" vs "tasks": stemming
    ],
)
def test_english_queries_hit_the_english_document(query: str) -> None:
    assert top(index(), query) == "en"


@pytest.mark.xfail(
    strict=True,
    reason="Known limitation: the Porter stemmer maps 'verify' to 'verifi' but 'verification' to "
    "'verif', so the two never meet. Prefix matching would not fix 'retries'/'retry' and would add "
    "noise, so the limitation is accepted and documented (see README, Limitations).",
)
def test_verify_does_not_match_verification() -> None:
    assert top(index(), "verify") == "en"


def test_terms_are_or_ed_so_one_unknown_word_does_not_kill_the_query() -> None:
    assert top(index(), "退款 火星") == "zh"
    assert top(index(), "refund scheduler zzzz") == "en"


def test_documents_matching_more_query_terms_rank_higher_and_scores_descend() -> None:
    idx = LexicalIndex()
    idx.add("one", "refund")
    idx.add("three", "refund policy window")
    idx.add("none", "completely unrelated words")
    hits = idx.search("refund policy window")
    assert [h.doc_id for h in hits] == ["three", "one"]
    assert hits[0].score > hits[1].score > 0


def test_no_match_empty_and_punctuation_only_queries_return_nothing() -> None:
    idx = index()
    assert idx.search("quantum") == [] and idx.search("") == [] and idx.search("?!") == []


@pytest.mark.parametrize("query", ['refund AND ("', "NEAR(a b", 'a" OR "b', "*", "col:val", "-x"])
def test_fts_syntax_in_user_text_is_never_interpreted(query: str) -> None:
    index().search(query)  # must not raise


def test_limit_remove_and_replace() -> None:
    idx = LexicalIndex()
    for i in range(5):
        idx.add(f"d{i}", "refund policy")
    assert len(idx.search("refund", k=3)) == 3
    idx.remove("d0")
    assert "d0" not in [h.doc_id for h in idx.search("refund", k=10)]
    idx.add("d1", "something else entirely")  # same id again replaces the text
    assert "d1" not in [h.doc_id for h in idx.search("refund", k=10)]
    assert [h.doc_id for h in idx.search("entirely")] == ["d1"]


def test_a_file_backed_index_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "idx.db"
    LexicalIndex(path).add("zh", DOCS["zh"])
    assert top(LexicalIndex(path), "退款") == "zh"


def test_missing_fts5_or_porter_gives_a_clear_error() -> None:
    class NoFts:
        def execute(self, *a: Any, **k: Any) -> Any:
            import sqlite3

            raise sqlite3.OperationalError("no such tokenizer: porter")

    with pytest.raises(RetrievalError, match=r"FTS5.*porter"):
        LexicalIndex(conn=NoFts())  # type: ignore[arg-type]
