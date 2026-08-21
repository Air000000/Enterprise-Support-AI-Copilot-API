from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ranx import Qrels


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


def build_ranx_qrels(rows: Iterable[dict[str, Any]]) -> Qrels:
    """Convert MTEB-style qrel rows into a ranx Qrels object."""
    qrels_dict: dict[str, dict[str, int]] = {}

    for row in rows:
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        relevance = int(row["score"])

        qrels_dict.setdefault(query_id, {})[document_id] = relevance

    return Qrels(qrels_dict)
