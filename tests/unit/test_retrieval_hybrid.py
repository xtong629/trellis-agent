from __future__ import annotations

from collections.abc import Sequence

import pytest

from trellis_agent.agent import Agent
from trellis_agent.cost import CostMeter, ModelPrice, PriceTable
from trellis_agent.llm import HashEmbedder, RuleLLM
from trellis_agent.memory.extract import RuleExtractor
from trellis_agent.memory.manager import MemoryConfig, MemoryManager
from trellis_agent.memory.store import MemoryItem, MemoryStore
from trellis_agent.retrieval.hybrid import HybridRetriever, rrf
from trellis_agent.retrieval.lexical import LexicalIndex
from trellis_agent.retrieval.vector import VectorIndex
from trellis_agent.types import Message, Usage


# ----------------------------------------------------------------- vector index
def test_cosine_search_orders_by_similarity_and_supports_remove() -> None:
    idx = VectorIndex()
    idx.add("x", [1.0, 0.0])
    idx.add("diag", [1.0, 1.0])
    idx.add("y", [0.0, 1.0])
    assert [h.doc_id for h in idx.search([1.0, 0.1], k=3)] == ["x", "diag", "y"]
    assert idx.search([1.0, 0.1], k=1)[0].doc_id == "x"
    idx.remove("x")
    assert [h.doc_id for h in idx.search([1.0, 0.1], k=3)] == ["diag", "y"]
    assert idx.search([0.0, 0.0], k=3) == []  # a zero vector matches nothing


# ----------------------------------------------------------------- rank fusion, by hand
def test_rrf_matches_a_hand_calculation() -> None:
    # lexical ranks d1 > d2 > d3; vector ranks d3 > d1. With k = 60 (rank is 1-based):
    #   d1 = 1/61 + 1/62 = 0.032522   d3 = 1/63 + 1/61 = 0.032266   d2 = 1/62 = 0.016129
    scores = rrf([["d1", "d2", "d3"], ["d3", "d1"]], k=60)
    assert scores["d1"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["d3"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["d2"] == pytest.approx(1 / 62)
    assert sorted(scores, key=lambda d: -scores[d]) == ["d1", "d3", "d2"]


def test_rrf_uses_ranks_not_scores_and_rewards_agreement() -> None:
    assert rrf([["a"], ["a"]], k=1)["a"] == pytest.approx(1.0)  # 1/2 + 1/2
    assert rrf([["a", "b"], ["c"]], k=60)["b"] < rrf([["a", "b"], ["b"]], k=60)["b"]
    assert rrf([], k=60) == {}


# ----------------------------------------------------------------- the two retrieval paths
def lexical_with(texts: dict[str, str]) -> LexicalIndex:
    idx = LexicalIndex()
    for doc_id, text in texts.items():
        idx.add(doc_id, text)
    return idx


TEXTS = {"en": "refund policy three days", "zh": "退款流程 审核", "other": "sunny weather today"}


async def test_without_a_vector_index_it_is_plain_bm25() -> None:
    idx = lexical_with(TEXTS)
    hybrid = HybridRetriever(idx)
    for query in ("refund policy", "退款", "weather"):
        assert [h.doc_id for h in await hybrid.search(query, 5)] == [
            h.doc_id for h in idx.search(query, 5)
        ]
    assert hybrid.drain() == []  # no embeddings, no overhead


class FakeEmbedder:
    """Maps known texts to chosen vectors so the vector ranking can be controlled exactly."""

    dim = 2

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.table[t] for t in texts]


async def test_hybrid_fuses_the_lexical_and_vector_rankings_with_rrf() -> None:
    docs = {"a": "refund policy window extra", "b": "refund policy extra", "c": "refund extra"}
    idx = lexical_with(docs)
    query = "refund policy window"
    # BM25 ranks a > b > c for this query. The vectors point the other way: the query vector is
    # closest to c, then b, and orthogonal to a (so a is not a vector hit at all).
    table = {docs["a"]: [0.0, 1.0], docs["b"]: [0.5, 1.0], docs["c"]: [1.0, 0.0], query: [1.0, 0.0]}
    hybrid = HybridRetriever(idx, VectorIndex(), FakeEmbedder(table))
    for doc_id, text in docs.items():
        await hybrid.index(doc_id, text)
    lexical_ranking = [h.doc_id for h in idx.search(query, 3)]
    assert lexical_ranking == ["a", "b", "c"]
    fused = await hybrid.search(query, 3)
    expected = rrf([lexical_ranking, ["c", "b"]], k=60)  # a: 1/61  b: 1/62+1/62  c: 1/63+1/61
    assert [h.doc_id for h in fused] == ["c", "b", "a"]
    for hit in fused:
        assert hit.score == pytest.approx(expected[hit.doc_id])


async def test_embedding_calls_are_estimated_billed_and_drained() -> None:
    meter = CostMeter(PriceTable({"embedder": ModelPrice(1_000_000.0, 0.0)}))  # cost == tokens
    hybrid = HybridRetriever(
        LexicalIndex(), VectorIndex(), HashEmbedder(dim=16), meter=meter, model="embedder"
    )
    await hybrid.index("d1", "x" * 400)  # ~100 estimated tokens
    ((usage, cost),) = hybrid.drain()
    assert usage.estimated is True and usage.prompt_tokens == 100
    assert cost == pytest.approx(100.0) == pytest.approx(meter.total_usd)
    assert hybrid.drain() == []  # drained once


async def test_removing_an_item_removes_its_vector_too() -> None:
    embedder = HashEmbedder(dim=32)
    hybrid = HybridRetriever(lexical_with({"a": "refund policy"}), VectorIndex(), embedder)
    await hybrid.index("a", "refund policy")
    (vec,) = await embedder.embed(["refund policy"])
    assert [h.doc_id for h in hybrid.vector.search(vec, 3)] == ["a"]  # type: ignore[union-attr]
    hybrid.remove("a")
    assert hybrid.vector.search(vec, 3) == []  # type: ignore[union-attr]


# ----------------------------------------------------------------- inside the memory system
def hybrid_manager(meter: CostMeter | None = None) -> MemoryManager:
    store = MemoryStore()
    retriever = HybridRetriever(
        store.index, VectorIndex(), HashEmbedder(dim=64), meter=meter, model="embedder"
    )
    return MemoryManager(store, RuleExtractor(), MemoryConfig(), retriever=retriever)


async def test_the_memory_system_works_with_the_hybrid_retriever_and_reports_embedding_cost() -> (
    None
):
    meter = CostMeter(PriceTable({"embedder": ModelPrice(1_000_000.0, 0.0)}))
    m = hybrid_manager(meter)
    written = await m.observe("s", [Message.user("我的截止日期是周五"), Message.assistant("好")])
    assert any(
        e[0] == "model_request" and e[1]["phase"] == "memory_embed" for e in written.overhead.events
    )
    ctx = await m.build_context("s", "截止日期", [])
    assert "周五" in (ctx.system[0].content or "")
    assert any(
        e[1].get("phase") == "memory_embed" for e in ctx.overhead.events if e[0] == "model_request"
    )
    assert meter.total_usd > 0


async def test_superseded_and_forgotten_items_leave_the_vector_index() -> None:
    m = hybrid_manager()
    await m.observe("s", [Message.user("我的截止日期是周五")])
    await m.observe("s", [Message.user("截止日期改到周二了")])
    vectors = m.retriever.vector._vecs  # type: ignore[union-attr]
    live = {i.id for i in m.store.items(tier="semantic")}
    assert set(vectors) <= live | {i.id for i in m.store.items(tier="episodic")}
    assert len(live) == 1


async def test_an_agent_with_hybrid_memory_bills_embeddings_in_the_run_cost() -> None:
    meter = CostMeter(
        PriceTable({"mock": ModelPrice(1.0, 1.0), "embedder": ModelPrice(1_000_000.0, 0.0)})
    )
    llm = RuleLLM(
        lambda msgs, tools: Message.assistant("ok"), usage_per_call=Usage(10, 5), meter=meter
    )
    agent = Agent(llm, memory=hybrid_manager(meter))
    result = await agent.arun("我的截止日期是周五", session_id="s")
    phases = [e.data.get("phase") for e in result.trace if e.type == "model_request"]
    assert "memory_embed" in phases
    assert result.cost_usd == pytest.approx(meter.total_usd)


async def test_a_failing_embedder_degrades_to_no_memory_not_a_failed_run() -> None:
    class Down:
        dim = 2

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise ConnectionError("embedding service down")

    store = MemoryStore()
    manager = MemoryManager(
        store, RuleExtractor(), retriever=HybridRetriever(store.index, VectorIndex(), Down())
    )
    agent = Agent(RuleLLM(lambda m, t: Message.assistant("ok")), memory=manager)
    result = await agent.arun("hello there", session_id="s")
    assert result.error is None and result.content == "ok"
    assert "memory_error" in [e.data.get("category") for e in result.trace if e.type == "error"]


def test_the_item_type_is_unchanged_by_the_vector_module() -> None:
    assert MemoryItem("i", "t", "semantic").importance == 0.5
