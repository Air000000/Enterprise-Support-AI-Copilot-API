from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ranx import Qrels, Run, evaluate


PRIMARY_IR_METRICS = (
    "recall@5",
    "recall@20",
    "mrr@10",
)


def collapse_chunk_results_to_document_ranking(
    results: Iterable[Any],
) -> list[str]:
    """
    Convert chunk-level retrieval ranking into document-level ranking.

    Keep the first occurrence of each document because the input order already
    represents the retriever ranking.
    """
    document_ranking: list[str] = []
    seen: set[str] = set()

    for result in results:
        document_id = result.document_id

        if document_id in seen:
            continue

        seen.add(document_id)
        document_ranking.append(document_id)

    return document_ranking


def build_ranx_qrels(
    qrels_by_query: Mapping[str, Mapping[str, int]],
) -> Qrels:
    """Convert normalized document-level relevance labels into ranx Qrels."""
    qrels_dict = {
        str(query_id): {
            str(document_id): int(relevance)
            for document_id, relevance in document_relevance.items()
        }
        for query_id, document_relevance in qrels_by_query.items()
    }
    return Qrels(qrels_dict)


def build_ranx_run(
    query_id: str,
    ranked_document_ids: Sequence[str],
) -> Run:
    """Convert one ordered document ranking into a ranx Run object."""
    ranking_size = len(ranked_document_ids)
    scores = {
        str(document_id): float(ranking_size - rank)
        for rank, document_id in enumerate(ranked_document_ids)
    }
    return Run({str(query_id): scores})


def evaluate_ir_metrics(
    qrels: Qrels,
    run: Run,
    metrics: Sequence[str],
) -> dict[str, float]:
    """Evaluate a run with the requested retrieval metrics."""
    results = evaluate(qrels, run, list(metrics))

    return {
        metric: float(results[metric])
        for metric in metrics
    }


def evaluate_ir_run(qrels: Qrels, run: Run) -> dict[str, float]:
    """Evaluate a run with the frozen primary retrieval metrics."""
    return evaluate_ir_metrics(
        qrels,
        run,
        PRIMARY_IR_METRICS,
    )