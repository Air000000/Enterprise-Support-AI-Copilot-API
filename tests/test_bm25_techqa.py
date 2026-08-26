from __future__ import annotations

from importlib import import_module

import pytest

from experiments.evals.adapters.techqa import TechQADocument


def _load_module():
    try:
        return import_module("experiments.evals.retrievers.bm25_techqa")
    except ModuleNotFoundError:
        pytest.fail("bm25_techqa retriever is not implemented yet")


def test_technical_tokenizer_preserves_error_codes_and_versions() -> None:
    module = _load_module()

    assert module.tokenize_technical(
        "Error 0x80070005 on v1.2.3 / CVE-2026-1234"
    ) == ["error", "0x80070005", "on", "v1.2.3", "cve-2026-1234"]


def test_technical_tokenizer_normalizes_case() -> None:
    module = _load_module()

    assert module.tokenize_technical("IBM DataPower API-CONNECT") == [
        "ibm",
        "datapower",
        "api-connect",
    ]


def test_bm25_retriever_returns_document_ids() -> None:
    module = _load_module()
    retriever = module.TechQABM25Retriever(
        [
            TechQADocument("a", "permission denied error 0x80070005"),
            TechQADocument("b", "printer configuration"),
        ]
    )

    assert retriever.search("0x80070005", top_k=2)[0] == "a"


def test_bm25_retriever_is_deterministic_across_input_order() -> None:
    module = _load_module()
    documents = [
        TechQADocument("b", "shared lexical token"),
        TechQADocument("a", "shared lexical token"),
        TechQADocument("c", "unrelated content"),
    ]

    forward = module.TechQABM25Retriever(documents)
    reversed_input = module.TechQABM25Retriever(list(reversed(documents)))

    assert forward.search("shared lexical token", top_k=3) == reversed_input.search(
        "shared lexical token",
        top_k=3,
    )


def test_bm25_retriever_returns_empty_for_nonpositive_top_k() -> None:
    module = _load_module()
    retriever = module.TechQABM25Retriever(
        [TechQADocument("a", "permission denied")]
    )

    assert retriever.search("permission", top_k=0) == []
    assert retriever.search("permission", top_k=-1) == []


def test_bm25_retriever_returns_empty_for_empty_token_query() -> None:
    module = _load_module()
    retriever = module.TechQABM25Retriever(
        [TechQADocument("a", "permission denied")]
    )

    assert retriever.search("--- /// ...", top_k=100) == []
