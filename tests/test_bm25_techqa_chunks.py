from importlib import import_module

import pytest

from experiments.evals.adapters.techqa import TechQADocument


def _load_module():
    try:
        return import_module("experiments.evals.retrievers.bm25_techqa_chunks")
    except ModuleNotFoundError:
        pytest.fail("chunk BM25 retriever is not implemented yet")


def test_build_techqa_chunks_is_deterministic_across_document_order() -> None:
    module = _load_module()
    documents = [
        TechQADocument("b", "beta paragraph"),
        TechQADocument("a", "alpha paragraph"),
    ]

    forward = module.build_techqa_chunks(documents)
    reversed_input = module.build_techqa_chunks(list(reversed(documents)))

    assert forward == reversed_input
    assert [
        (chunk.document_id, chunk.chunk_index, chunk.chunk_id)
        for chunk in forward
    ] == [
        ("a", 0, "a_chunk_0"),
        ("b", 0, "b_chunk_0"),
    ]

def test_chunk_bm25_returns_specific_chunk_identity() -> None:
    module = _load_module()
    chunks = [
        module.TechQAChunk(
            "a_chunk_0",
            "a",
            0,
            "printer configuration",
        ),
        module.TechQAChunk(
            "b_chunk_0",
            "b",
            0,
            "permission denied 0x80070005",
        ),
        module.TechQAChunk(
            "b_chunk_1",
            "b",
            1,
            "unrelated follow up",
        ),
    ]

    retriever = module.TechQAChunkBM25Retriever(chunks)
    results = retriever.search("0x80070005", top_k=3)

    assert results[0].chunk_id == "b_chunk_0"
    assert results[0].document_id == "b"
    assert results[0].chunk_index == 0

def test_chunk_bm25_returns_empty_for_nonpositive_top_k() -> None:
    module = _load_module()
    retriever = module.TechQAChunkBM25Retriever(
        [
            module.TechQAChunk(
                "a_chunk_0",
                "a",
                0,
                "permission denied",
            )
        ]
    )

    assert retriever.search("permission", top_k=0) == []
    assert retriever.search("permission", top_k=-1) == []


def test_chunk_bm25_returns_empty_for_empty_token_query() -> None:
    module = _load_module()
    retriever = module.TechQAChunkBM25Retriever(
        [
            module.TechQAChunk(
                "a_chunk_0",
                "a",
                0,
                "permission denied",
            )
        ]
    )

    assert retriever.search("--- /// ...", top_k=100) == []