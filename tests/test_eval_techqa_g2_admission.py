from __future__ import annotations

import importlib
import importlib.util

import pytest

from experiments.evals.rerankers.qwen3_reranker import (
    RerankResult,
    RerankedCandidate,
)


MODULE_NAME = "experiments.evals.eval_techqa_g2_admission"


def _g2_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, "eval_techqa_g2_admission is not implemented"
    return importlib.import_module(MODULE_NAME)


def _shared_rerank(*document_ids: str) -> RerankResult:
    return RerankResult(
        results=tuple(
            RerankedCandidate(
                chunk_id=f"chunk-{index}",
                document_id=document_id,
                content=f"content-{index}",
                original_index=index,
                relevance_score=1.0 - index / 100.0,
            )
            for index, document_id in enumerate(document_ids)
        ),
        request_id="shared-global-rerank",
        total_tokens=123,
    )


def test_rerank_informed_admission_uses_first_unique_docs_in_rerank_order():
    module = _g2_module()
    selector = getattr(module, "select_rerank_informed_document_ids", None)
    assert selector is not None, "select_rerank_informed_document_ids is not implemented"

    result = _shared_rerank("D3", "D3", "D1", "D5", "D2", "D4", "D6")

    assert selector(result, limit=5) == ("D3", "D1", "D5", "D2", "D4")


def test_rerank_informed_admission_returns_all_unique_docs_when_under_limit():
    module = _g2_module()
    selector = getattr(module, "select_rerank_informed_document_ids", None)
    assert selector is not None, "select_rerank_informed_document_ids is not implemented"

    result = _shared_rerank("D2", "D2", "D1")

    assert selector(result, limit=5) == ("D2", "D1")


@pytest.mark.parametrize("limit", [0, -1])
def test_rerank_informed_admission_requires_positive_limit(limit: int):
    module = _g2_module()
    selector = getattr(module, "select_rerank_informed_document_ids", None)
    assert selector is not None, "select_rerank_informed_document_ids is not implemented"

    with pytest.raises(ValueError, match="limit must be positive"):
        selector(_shared_rerank("D1"), limit=limit)
