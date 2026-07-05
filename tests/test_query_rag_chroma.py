from rag_runtime import query_rag_chroma
from rag_runtime.query_chroma import ChromaSearchResult


def make_search_result(distance: float) -> ChromaSearchResult:
    return ChromaSearchResult(
        chunk_id="chunk_1",
        document_id="doc_1",
        title="Doc 1",
        source_path="experiments/docs/doc_1.md",
        chunk_index=0,
        content="Useful support content.",
        distance=distance,
        tenant_id="tenant_demo",
        category="it",
    )


def test_ask_rag_returns_no_context_without_results(monkeypatch):
    monkeypatch.setattr(query_rag_chroma, "search_chroma", lambda **kwargs: [])

    result = query_rag_chroma.ask_rag("missing answer")

    assert result.retrieval_status == "no_context"
    assert result.top_distance is None
    assert result.sources == []


def test_ask_rag_returns_low_relevance_when_top_distance_exceeds_threshold(monkeypatch):
    monkeypatch.setattr(
        query_rag_chroma,
        "search_chroma",
        lambda **kwargs: [make_search_result(distance=1.2)],
    )

    result = query_rag_chroma.ask_rag("weak answer", max_distance=0.9)

    assert result.retrieval_status == "refused_low_relevance"
    assert result.top_distance == 1.2
    assert len(result.sources) == 1


def test_ask_rag_returns_ok_when_retrieval_is_relevant(monkeypatch):
    monkeypatch.setattr(
        query_rag_chroma,
        "search_chroma",
        lambda **kwargs: [make_search_result(distance=0.4)],
    )
    monkeypatch.setattr(
        query_rag_chroma,
        "generate_answer",
        lambda question, context: "Answer from context.",
    )

    result = query_rag_chroma.ask_rag("good answer", max_distance=0.9)

    assert result.retrieval_status == "ok"
    assert result.answer == "Answer from context."
    assert result.top_distance == 0.4