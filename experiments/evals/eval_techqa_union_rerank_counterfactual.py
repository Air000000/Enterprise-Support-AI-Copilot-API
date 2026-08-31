from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.evals.eval_techqa_hybrid_rerank import (
    HybridCandidate,
    HybridSnapshotRecord,
)
from experiments.evals.ir.ranx_adapter import (
    collapse_chunk_results_to_document_ranking,
)
from experiments.evals.rerankers.qwen3_reranker import (
    RerankCandidate,
    rerank_candidates,
)
from experiments.evals.retrievers.bm25_techqa_chunks import (
    TechQAChunk,
)


CANDIDATE_K = 100


@dataclass(frozen=True)
class UnionSnapshotRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    dense_chunk_ids: tuple[str, ...]
    bm25_chunk_ids: tuple[str, ...]
    union_candidates: tuple[HybridCandidate, ...]


@dataclass(frozen=True)
class UnionRerankResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    union_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    reranked_document_ids: tuple[str, ...]
    document_ranking: tuple[str, ...]
    rerank_latency_ms: float
    request_id: str | None
    total_tokens: int | None


def _require_exact_unique_source_ranking(
    values: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(
        str(value)
        for value in values
    )

    if (
        len(normalized) != CANDIDATE_K
        or len(set(normalized)) != CANDIDATE_K
    ):
        raise RuntimeError(
            f"{label} must contain exactly "
            f"{CANDIDATE_K} unique chunks"
        )

    return normalized


def build_union_candidate_ids(
    dense_chunk_ids: Sequence[str],
    bm25_chunk_ids: Sequence[str],
) -> tuple[str, ...]:
    """Preserve Dense order, then append BM25 chunks not already seen."""
    return tuple(
        dict.fromkeys(
            (*dense_chunk_ids, *bm25_chunk_ids)
        )
    )


def build_union_snapshot_record(
    record: HybridSnapshotRecord,
    *,
    chunks_by_id: Mapping[str, TechQAChunk],
) -> UnionSnapshotRecord:
    if not record.question_id.startswith("TRAIN_"):
        raise RuntimeError(
            "R2 requires TRAIN-only input"
        )

    dense_chunk_ids = _require_exact_unique_source_ranking(
        record.dense_chunk_ids,
        label="Dense candidate ranking",
    )

    bm25_chunk_ids = _require_exact_unique_source_ranking(
        record.bm25_chunk_ids,
        label="BM25 candidate ranking",
    )

    union_chunk_ids = build_union_candidate_ids(
        dense_chunk_ids,
        bm25_chunk_ids,
    )

    return UnionSnapshotRecord(
        question_id=record.question_id,
        question=record.question,
        relevant_document_ids=record.relevant_document_ids,
        dense_chunk_ids=dense_chunk_ids,
        bm25_chunk_ids=bm25_chunk_ids,
        union_candidates=tuple(
            HybridCandidate(
                chunk_id=chunks_by_id[chunk_id].chunk_id,
                document_id=chunks_by_id[chunk_id].document_id,
                content=chunks_by_id[chunk_id].content,
            )
            for chunk_id in union_chunk_ids
        ),
    )


def rerank_union_snapshot_record(
    record: UnionSnapshotRecord,
    *,
    reranker: Callable[..., Any] = rerank_candidates,
    clock: Callable[[], float] = time.perf_counter,
) -> UnionRerankResult:
    if not record.question_id.startswith("TRAIN_"):
        raise RuntimeError(
            "R2 requires TRAIN-only input"
        )

    union_chunk_ids = tuple(
        candidate.chunk_id
        for candidate in record.union_candidates
    )

    candidates = [
        RerankCandidate(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            content=candidate.content,
        )
        for candidate in record.union_candidates
    ]

    started = clock()

    rerank_result = reranker(
        record.question.rstrip(),
        candidates,
    )

    rerank_latency_ms = (
        clock() - started
    ) * 1000.0

    reranked_chunk_ids = tuple(
        str(candidate.chunk_id)
        for candidate in rerank_result.results
    )

    reranked_document_ids = tuple(
        str(candidate.document_id)
        for candidate in rerank_result.results
    )

    document_ranking = tuple(
        collapse_chunk_results_to_document_ranking(
            rerank_result.results
        )
    )

    return UnionRerankResult(
        question_id=record.question_id,
        relevant_document_ids=record.relevant_document_ids,
        union_chunk_ids=union_chunk_ids,
        reranked_chunk_ids=reranked_chunk_ids,
        reranked_document_ids=reranked_document_ids,
        document_ranking=document_ranking,
        rerank_latency_ms=rerank_latency_ms,
        request_id=rerank_result.request_id,
        total_tokens=rerank_result.total_tokens,
    )


FUSION_LOSS_IDS = (
    "TRAIN_Q005",
    "TRAIN_Q011",
    "TRAIN_Q098",
    "TRAIN_Q122",
    "TRAIN_Q130",
    "TRAIN_Q135",
    "TRAIN_Q137",
    "TRAIN_Q141",
    "TRAIN_Q207",
    "TRAIN_Q261",
    "TRAIN_Q318",
    "TRAIN_Q319",
    "TRAIN_Q328",
    "TRAIN_Q343",
    "TRAIN_Q372",
    "TRAIN_Q460",
    "TRAIN_Q497",
    "TRAIN_Q543",
    "TRAIN_Q565",
)

EVIDENCE_LOSS_IDS = (
    "TRAIN_Q045",
    "TRAIN_Q122",
    "TRAIN_Q152",
    "TRAIN_Q524",
    "TRAIN_Q579",
)


def build_frozen_r2_cohort_contract() -> dict[str, object]:
    overlap_ids = tuple(
        question_id
        for question_id in EVIDENCE_LOSS_IDS
        if question_id in FUSION_LOSS_IDS
    )

    query_ids = tuple(
        dict.fromkeys(
            (*FUSION_LOSS_IDS, *EVIDENCE_LOSS_IDS)
        )
    )

    return {
        "fusion_loss_ids": FUSION_LOSS_IDS,
        "evidence_loss_ids": EVIDENCE_LOSS_IDS,
        "overlap_ids": overlap_ids,
        "query_ids": query_ids,
        "max_provider_calls": len(query_ids),
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "selection_policy": (
            "frozen_two_cohort_no_posthoc_selection"
        ),
    }



def select_frozen_r2_paid_records(
    records: Sequence[UnionSnapshotRecord],
) -> tuple[UnionSnapshotRecord, ...]:
    contract = build_frozen_r2_cohort_contract()

    records_by_id = {
        record.question_id: record
        for record in records
    }

    frozen_ids = contract["query_ids"]

    missing_ids = tuple(
        question_id
        for question_id in frozen_ids
        if question_id not in records_by_id
    )

    if missing_ids:
        raise RuntimeError(
            "R2 frozen paid cohort is incomplete: "
            + ", ".join(missing_ids)
        )

    return tuple(
        records_by_id[question_id]
        for question_id in frozen_ids
    )


@dataclass(frozen=True)
class R2PaidEvalSummary:
    completed_query_count: int
    provider_calls: int
    provider_total_tokens: int
    stopped_reason: str | None
    results: tuple[UnionRerankResult, ...]


def _r2_result_from_payload(
    payload: Mapping[str, Any],
) -> UnionRerankResult:
    return UnionRerankResult(
        question_id=str(payload["question_id"]),
        relevant_document_ids=tuple(
            str(value)
            for value in payload["relevant_document_ids"]
        ),
        union_chunk_ids=tuple(
            str(value)
            for value in payload["union_chunk_ids"]
        ),
        reranked_chunk_ids=tuple(
            str(value)
            for value in payload["reranked_chunk_ids"]
        ),
        reranked_document_ids=tuple(
            str(value)
            for value in payload["reranked_document_ids"]
        ),
        document_ranking=tuple(
            str(value)
            for value in payload["document_ranking"]
        ),
        rerank_latency_ms=float(
            payload["rerank_latency_ms"]
        ),
        request_id=(
            None
            if payload.get("request_id") is None
            else str(payload["request_id"])
        ),
        total_tokens=(
            None
            if payload.get("total_tokens") is None
            else int(payload["total_tokens"])
        ),
    )


def _load_r2_paid_checkpoint(
    path: Path,
) -> list[UnionRerankResult]:
    if not path.exists():
        return []

    results = []

    for line in path.read_text(
        encoding="utf-8",
    ).splitlines():
        if not line.strip():
            continue

        results.append(
            _r2_result_from_payload(
                json.loads(line)
            )
        )

    question_ids = [
        result.question_id
        for result in results
    ]

    if len(set(question_ids)) != len(question_ids):
        raise RuntimeError(
            "R2 paid checkpoint contains duplicate question_id"
        )

    return results


def _write_r2_durable_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
            )
            + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def _append_r2_durable_result(
    path: Path,
    result: UnionRerankResult,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(
            json.dumps(
                asdict(result),
                ensure_ascii=False,
            )
            + "\n"
        )
        file.flush()
        os.fsync(file.fileno())


def run_resumable_r2_paid_eval(
    records: Sequence[UnionSnapshotRecord],
    *,
    evaluator: Callable[
        [UnionSnapshotRecord],
        UnionRerankResult,
    ],
    checkpoint_path: str | Path,
    inflight_path: str | Path,
    max_provider_calls: int,
    token_stop_threshold: int,
) -> R2PaidEvalSummary:
    contract = build_frozen_r2_cohort_contract()

    frozen_limit = int(
        contract["max_provider_calls"]
    )

    if (
        max_provider_calls <= 0
        or max_provider_calls > frozen_limit
    ):
        raise RuntimeError(
            "R2 max_provider_calls must be within "
            f"1..{frozen_limit}"
        )

    selected_records = select_frozen_r2_paid_records(
        records
    )

    checkpoint = Path(checkpoint_path)
    inflight = Path(inflight_path)

    existing = _load_r2_paid_checkpoint(
        checkpoint
    )

    frozen_ids = tuple(
        contract["query_ids"]
    )

    frozen_id_set = set(frozen_ids)

    for result in existing:
        if result.question_id not in frozen_id_set:
            raise RuntimeError(
                "R2 checkpoint contains non-frozen query: "
                f"{result.question_id}"
            )

    by_question_id = {
        result.question_id: result
        for result in existing
    }

    if len(by_question_id) > max_provider_calls:
        raise RuntimeError(
            "R2 checkpoint already exceeds provider call limit"
        )

    if inflight.exists():
        payload = json.loads(
            inflight.read_text(
                encoding="utf-8"
            )
        )

        inflight_question_id = str(
            payload["question_id"]
        )

        if inflight_question_id in by_question_id:
            inflight.unlink()
        else:
            raise RuntimeError(
                "R2 inflight/uncertain provider request "
                "requires manual review; automatic replay "
                "is forbidden"
            )

    provider_total_tokens = sum(
        result.total_tokens or 0
        for result in by_question_id.values()
    )

    stopped_reason = None

    for record in selected_records:
        question_id = record.question_id

        if question_id in by_question_id:
            continue

        if len(by_question_id) >= max_provider_calls:
            stopped_reason = "provider_call_limit"
            break

        if provider_total_tokens >= token_stop_threshold:
            stopped_reason = "token_stop_threshold"
            break

        _write_r2_durable_json(
            inflight,
            {
                "question_id": question_id,
            },
        )

        result = evaluator(record)

        if result.question_id != question_id:
            raise RuntimeError(
                "R2 provider result question_id mismatch"
            )

        if result.total_tokens is None:
            raise RuntimeError(
                "R2 provider result missing total_tokens"
            )

        _append_r2_durable_result(
            checkpoint,
            result,
        )

        by_question_id[question_id] = result

        provider_total_tokens += (
            result.total_tokens
        )

        inflight.unlink()

    ordered_results = tuple(
        by_question_id[question_id]
        for question_id in frozen_ids
        if question_id in by_question_id
    )

    return R2PaidEvalSummary(
        completed_query_count=len(
            ordered_results
        ),
        provider_calls=len(
            ordered_results
        ),
        provider_total_tokens=(
            provider_total_tokens
        ),
        stopped_reason=stopped_reason,
        results=ordered_results,
    )
