from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rag_runtime.query_rag_chroma import get_llm_client

DEFAULT_RERANK_MODEL = "qwen3-rerank"
DEFAULT_RERANK_INSTRUCTION = (
    "Rank the candidate passages by relevance to resolving the technical support query."
)


class RerankProviderError(RuntimeError):
    """Raised when the rerank provider fails or returns an invalid ranking."""


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: str
    document_id: str
    content: str


@dataclass(frozen=True)
class RerankedCandidate:
    chunk_id: str
    document_id: str
    content: str
    original_index: int
    relevance_score: float


@dataclass(frozen=True)
class RerankResult:
    results: tuple[RerankedCandidate, ...]
    request_id: str | None
    total_tokens: int | None


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RerankProviderError(f"invalid rerank provider {label}")
    return value


def rerank_candidates(
    query: str,
    candidates: Sequence[RerankCandidate],
    *,
    client: Any | None = None,
    model: str = DEFAULT_RERANK_MODEL,
    instruct: str = DEFAULT_RERANK_INSTRUCTION,
) -> RerankResult:
    """Rerank an existing candidate pool without changing candidate identity."""
    provider = client or get_llm_client()
    body = {
        "model": model,
        "query": query.rstrip(),
        "documents": [candidate.content for candidate in candidates],
        "top_n": len(candidates),
        "instruct": instruct,
    }

    try:
        raw_response = provider.post("/reranks", body=body, cast_to=object)
    except Exception as error:
        raise RerankProviderError(f"rerank provider failed: {error}") from error

    response = _as_mapping(raw_response, label="response")
    raw_results = response.get("results")
    if not isinstance(raw_results, list):
        raise RerankProviderError("rerank provider did not return a complete permutation")

    parsed: list[tuple[int, float]] = []
    for item in raw_results:
        result_item = _as_mapping(item, label="result")
        try:
            original_index = int(result_item["index"])
            relevance_score = float(result_item["relevance_score"])
        except (KeyError, TypeError, ValueError) as error:
            raise RerankProviderError("invalid rerank provider result") from error
        parsed.append((original_index, relevance_score))

    expected_indices = list(range(len(candidates)))
    if sorted(index for index, _ in parsed) != expected_indices:
        raise RerankProviderError(
            "rerank provider must return a complete permutation of candidate indices"
        )

    reranked = tuple(
        RerankedCandidate(
            chunk_id=candidates[index].chunk_id,
            document_id=candidates[index].document_id,
            content=candidates[index].content,
            original_index=index,
            relevance_score=score,
        )
        for index, score in parsed
    )

    request_id = response.get("id")
    if request_id is not None:
        request_id = str(request_id)

    total_tokens: int | None = None
    usage = response.get("usage")
    if isinstance(usage, Mapping) and usage.get("total_tokens") is not None:
        total_tokens = int(usage["total_tokens"])

    return RerankResult(
        results=reranked,
        request_id=request_id,
        total_tokens=total_tokens,
    )
