from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ranx import Run

from experiments.evals.adapters.techqa import build_techqa_documents
from experiments.evals.eval_techqa_chunk_bm25 import (
    EXPECTED_SPLITTER_BLOB_SHA,
    EXPECTED_TECHQA_CHUNK_COUNT,
)
from experiments.evals.eval_techqa_rerank import (
    load_frozen_e0_rerank_records,
)
from experiments.evals.ir.ranx_adapter import (
    build_ranx_qrels,
    collapse_chunk_results_to_document_ranking,
    evaluate_ir_run,
)
from experiments.evals.ir.rrf import fuse_rrf
from experiments.evals.rerankers.qwen3_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
    RerankCandidate,
    rerank_candidates,
)
from experiments.evals.retrievers.bm25_techqa_chunks import (
    TechQAChunk,
    TechQAChunkBM25Retriever,
    build_techqa_chunks,
)


EXPECTED_TRAIN_COUNT = 450
EXPECTED_DOCUMENT_COUNT = 28481
CANDIDATE_K = 100
RRF_K = 60
TOKEN_STOP_THRESHOLD = 13_500_000
MONETARY_SAFETY_ENVELOPE_USD = 1.50

DEFAULT_E0_TRAIN_RESULTS_PATH = Path(
    "experiments/evals/reports/e0_dense/train_results.jsonl"
)
DEFAULT_E1_TRAIN_MANIFEST_PATH = Path(
    "experiments/evals/reports/e1_rerank/train_manifest.json"
)
DEFAULT_R4_AUDIT_MANIFEST_PATH = Path(
    "experiments/evals/reports/r4_chunk_bm25_audit/train_manifest.json"
)
DEFAULT_TECHQA_MANIFEST_PATH = Path(
    "experiments/evals/datasets/techqa/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/evals/reports/r4_c1_hybrid_rerank"
)
DEFAULT_SNAPSHOT_PATH = (
    DEFAULT_OUTPUT_DIR / "train_fused_snapshot.jsonl"
)
DEFAULT_SNAPSHOT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR / "train_snapshot_manifest.json"
)
DEFAULT_PREFLIGHT_PATH = (
    DEFAULT_OUTPUT_DIR / "train_preflight.json"
)


@dataclass(frozen=True)
class HybridCandidate:
    chunk_id: str
    document_id: str
    content: str


@dataclass(frozen=True)
class HybridSnapshotRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    dense_chunk_ids: tuple[str, ...]
    bm25_chunk_ids: tuple[str, ...]
    fused_candidates: tuple[HybridCandidate, ...]


def _require_unique_exact(
    values: Sequence[str],
    *,
    expected: int,
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)

    if len(normalized) != expected:
        raise RuntimeError(
            f"{label} must contain exactly {expected} chunks: "
            f"actual={len(normalized)}"
        )

    if len(set(normalized)) != expected:
        raise RuntimeError(
            f"{label} contains duplicate chunk_id values"
        )

    return normalized


def build_hybrid_snapshot_record(
    record: Mapping[str, Any],
    *,
    chunks_by_id: Mapping[str, TechQAChunk],
    bm25_searcher: Callable[..., list[TechQAChunk]],
) -> HybridSnapshotRecord:
    question_id = str(record["question_id"])

    if not question_id.startswith("TRAIN_"):
        raise RuntimeError("C1 requires TRAIN-only input")

    dense_chunk_ids = _require_unique_exact(
        record["raw_chunk_ids"],
        expected=CANDIDATE_K,
        label="Dense candidate ranking",
    )

    bm25_chunks = bm25_searcher(
        str(record["question"]).rstrip(),
        top_k=CANDIDATE_K,
    )

    bm25_chunk_ids = _require_unique_exact(
        [chunk.chunk_id for chunk in bm25_chunks],
        expected=CANDIDATE_K,
        label="BM25 candidate ranking",
    )

    fused_chunk_ids = tuple(
        fuse_rrf(
            [dense_chunk_ids, bm25_chunk_ids],
            rrf_k=RRF_K,
            top_k=CANDIDATE_K,
        )
    )

    fused_chunk_ids = _require_unique_exact(
        fused_chunk_ids,
        expected=CANDIDATE_K,
        label="Hybrid fused ranking",
    )

    missing = [
        chunk_id
        for chunk_id in fused_chunk_ids
        if chunk_id not in chunks_by_id
    ]
    if missing:
        raise RuntimeError(
            "fused chunk(s) missing from frozen chunk universe: "
            + ", ".join(missing[:5])
        )

    return HybridSnapshotRecord(
        question_id=question_id,
        question=str(record["question"]),
        relevant_document_ids=tuple(
            str(value)
            for value in record["relevant_document_ids"]
        ),
        dense_chunk_ids=dense_chunk_ids,
        bm25_chunk_ids=bm25_chunk_ids,
        fused_candidates=tuple(
            HybridCandidate(
                chunk_id=chunks_by_id[chunk_id].chunk_id,
                document_id=chunks_by_id[chunk_id].document_id,
                content=chunks_by_id[chunk_id].content,
            )
            for chunk_id in fused_chunk_ids
        ),
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_project_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()

    if not value:
        raise RuntimeError("unable to resolve project SHA")

    return value


def _resolve_splitter_blob_sha() -> str:
    completed = subprocess.run(
        [
            "git",
            "hash-object",
            "rag_runtime/text_splitter.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    Path(path).write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_snapshot(
    records: Sequence[HybridSnapshotRecord],
    path: str | Path,
) -> None:
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with snapshot_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    asdict(record),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def prepare_frozen_snapshot(
    *,
    e0_results_path: str | Path = DEFAULT_E0_TRAIN_RESULTS_PATH,
    dataset_manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
    e1_manifest_path: str | Path = DEFAULT_E1_TRAIN_MANIFEST_PATH,
    audit_manifest_path: str | Path = DEFAULT_R4_AUDIT_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    e0_path = Path(e0_results_path)
    dataset_path = Path(dataset_manifest_path)
    e1_path = Path(e1_manifest_path)
    audit_path = Path(audit_manifest_path)
    output_path = Path(output_dir)

    required_paths = (
        e0_path,
        dataset_path,
        e1_path,
        audit_path,
    )
    missing_paths = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]
    if missing_paths:
        raise RuntimeError(
            "required C1 input artifact(s) missing: "
            + ", ".join(missing_paths)
        )

    print("[R4 C1] Loading frozen E0 TRAIN records...")

    records = load_frozen_e0_rerank_records(
        e0_path,
        expected_count=EXPECTED_TRAIN_COUNT,
    )

    question_ids = [
        str(record["question_id"])
        for record in records
    ]

    if any(
        not question_id.startswith("TRAIN_")
        for question_id in question_ids
    ):
        raise RuntimeError(
            "C1 snapshot preparation requires TRAIN-only input"
        )

    if len(set(question_ids)) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(
            "C1 requires exactly 450 unique TRAIN question IDs"
        )

    dataset_manifest = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )
    retrieval_dataset = dataset_manifest[
        "retrieval_dataset"
    ]

    audit_manifest = json.loads(
        audit_path.read_text(encoding="utf-8")
    )
    e1_manifest = json.loads(
        e1_path.read_text(encoding="utf-8")
    )

    actual_e0_sha = _sha256(e0_path)
    expected_current_e0_sha = str(
        audit_manifest[
            "input_e0_train_results_sha256"
        ]
    )
    historical_e0_sha = str(
        e1_manifest["identity"]["source_e0"][
            "results_sha256"
        ]
    )

    if actual_e0_sha != expected_current_e0_sha:
        raise RuntimeError(
            "current frozen E0 TRAIN artifact changed: "
            f"expected={expected_current_e0_sha}, "
            f"actual={actual_e0_sha}"
        )

    audit_historical_sha = str(
        audit_manifest[
            "historical_e0_train_results_sha256"
        ]
    )
    if historical_e0_sha != audit_historical_sha:
        raise RuntimeError(
            "historical E0 provenance disagreement between "
            "E1 and R4 audit manifests"
        )

    splitter_blob_sha = _resolve_splitter_blob_sha()
    if splitter_blob_sha != EXPECTED_SPLITTER_BLOB_SHA:
        raise RuntimeError(
            "splitter blob mismatch: "
            f"expected={EXPECTED_SPLITTER_BLOB_SHA}, "
            f"actual={splitter_blob_sha}"
        )

    bm25_version = version("bm25s")
    if bm25_version != "0.3.10":
        raise RuntimeError(
            "bm25s version mismatch: "
            f"expected=0.3.10, actual={bm25_version}"
        )

    print("[R4 C1] Loading frozen TechQA corpus...")

    from datasets import load_dataset

    corpus_rows = load_dataset(
        retrieval_dataset["repo"],
        "corpus",
        split="train",
        revision=retrieval_dataset["revision"],
    )

    documents = build_techqa_documents(corpus_rows)

    if len(documents) != EXPECTED_DOCUMENT_COUNT:
        raise RuntimeError(
            "TechQA document count mismatch: "
            f"expected={EXPECTED_DOCUMENT_COUNT}, "
            f"actual={len(documents)}"
        )

    print(
        "[R4 C1] Building deterministic chunk universe: "
        f"documents={len(documents)}"
    )

    chunks = build_techqa_chunks(documents)

    if len(chunks) != EXPECTED_TECHQA_CHUNK_COUNT:
        raise RuntimeError(
            "TechQA chunk count mismatch: "
            f"expected={EXPECTED_TECHQA_CHUNK_COUNT}, "
            f"actual={len(chunks)}"
        )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    if len(chunks_by_id) != EXPECTED_TECHQA_CHUNK_COUNT:
        raise RuntimeError(
            "TechQA chunk universe contains duplicate chunk IDs"
        )

    print(
        "[R4 C1] Building chunk BM25 index: "
        f"chunks={len(chunks)}"
    )

    bm25_retriever = TechQAChunkBM25Retriever(chunks)

    print(
        "[R4 C1] Building Hybrid fused candidates: "
        f"queries={len(records)}"
    )

    snapshot_records = [
        build_hybrid_snapshot_record(
            record,
            chunks_by_id=chunks_by_id,
            bm25_searcher=bm25_retriever.search,
        )
        for record in records
    ]

    if len(snapshot_records) != EXPECTED_TRAIN_COUNT:
        raise RuntimeError(
            "C1 snapshot query count mismatch"
        )

    snapshot_path = (
        output_path / DEFAULT_SNAPSHOT_PATH.name
    )
    snapshot_manifest_path = (
        output_path
        / DEFAULT_SNAPSHOT_MANIFEST_PATH.name
    )
    preflight_path = (
        output_path / DEFAULT_PREFLIGHT_PATH.name
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_snapshot(
        snapshot_records,
        snapshot_path,
    )

    snapshot_sha = _sha256(snapshot_path)

    manifest: dict[str, Any] = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r4_c1_hybrid_rerank",
        "split": "train",
        "query_count": EXPECTED_TRAIN_COUNT,
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "candidate_construction": {
            "dense_candidate_k": CANDIDATE_K,
            "bm25_candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "rrf_weights": "equal",
            "rrf_unit": "chunk_id",
            "fused_candidate_k": CANDIDATE_K,
            "cross_source_overlap_rule": (
                "retain both source rank contributions"
            ),
        },
        "snapshot": {
            "path": str(snapshot_path),
            "sha256": snapshot_sha,
            "format": "jsonl",
        },
        "reranker": {
            "model": DEFAULT_RERANK_MODEL,
            "instruction": DEFAULT_RERANK_INSTRUCTION,
            "candidate_chunk_k": CANDIDATE_K,
            "query_normalization": "rstrip",
            "provider_region": "Singapore",
        },
        "paid_safety": {
            "token_stop_threshold": (
                TOKEN_STOP_THRESHOLD
            ),
            "monetary_safety_envelope_usd": (
                MONETARY_SAFETY_ENVELOPE_USD
            ),
        },
        "chunking": {
            "strategy": "paragraph_aware_character",
            "chunk_size": 800,
            "chunk_overlap": 120,
            "min_chunk_size": 150,
            "splitter_blob_sha": splitter_blob_sha,
            "observed_chunk_count": len(chunks),
        },
        "bm25": {
            "library": "bm25s",
            "version": bm25_version,
            "method": "lucene",
            "k1": 1.5,
            "b": 0.75,
            "backend": "numpy",
            "indexed_unit": "chunk",
            "candidate_chunk_k": CANDIDATE_K,
            "query_normalization": "rstrip",
            "tokenizer_regex": (
                r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
            ),
        },
        "retrieval_dataset": retrieval_dataset,
        "input_e0_train_results_sha256": (
            actual_e0_sha
        ),
        "historical_e1_e0_train_results_sha256": (
            historical_e0_sha
        ),
        "e0_train_results_sha_matches_historical": (
            actual_e0_sha == historical_e0_sha
        ),
        "project_sha": _resolve_project_sha(),
    }

    preflight: dict[str, Any] = {
        "split": "train",
        "query_count": len(snapshot_records),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "dense_candidate_k": CANDIDATE_K,
        "bm25_candidate_k": CANDIDATE_K,
        "rrf_k": RRF_K,
        "fused_candidate_k": CANDIDATE_K,
        "all_dense_rankings_unique": all(
            len(set(record.dense_chunk_ids))
            == CANDIDATE_K
            for record in snapshot_records
        ),
        "all_bm25_rankings_unique": all(
            len(set(record.bm25_chunk_ids))
            == CANDIDATE_K
            for record in snapshot_records
        ),
        "all_fused_rankings_unique": all(
            len(
                {
                    candidate.chunk_id
                    for candidate
                    in record.fused_candidates
                }
            )
            == CANDIDATE_K
            for record in snapshot_records
        ),
        "snapshot_sha256": snapshot_sha,
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "reranker_model": DEFAULT_RERANK_MODEL,
        "token_stop_threshold": (
            TOKEN_STOP_THRESHOLD
        ),
        "monetary_safety_envelope_usd": (
            MONETARY_SAFETY_ENVELOPE_USD
        ),
    }

    if not (
        preflight["all_dense_rankings_unique"]
        and preflight["all_bm25_rankings_unique"]
        and preflight["all_fused_rankings_unique"]
    ):
        raise RuntimeError(
            "C1 candidate uniqueness preflight failed"
        )

    _write_json(
        snapshot_manifest_path,
        manifest,
    )
    _write_json(
        preflight_path,
        preflight,
    )

    print("[R4 C1] Frozen snapshot prepared.")
    print(
        json.dumps(
            preflight,
            ensure_ascii=False,
            indent=2,
        )
    )

    return preflight



@dataclass(frozen=True)
class HybridRerankResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    fused_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    reranked_document_ids: tuple[str, ...]
    document_ranking: tuple[str, ...]
    rerank_latency_ms: float
    request_id: str | None
    total_tokens: int | None


def rerank_snapshot_record(
    record: HybridSnapshotRecord,
    *,
    reranker: Callable[..., Any] = rerank_candidates,
    clock: Callable[[], float] = time.perf_counter,
) -> HybridRerankResult:
    if not record.question_id.startswith("TRAIN_"):
        raise RuntimeError("C1 requires TRAIN-only input")

    fused_chunk_ids = _require_unique_exact(
        [
            candidate.chunk_id
            for candidate in record.fused_candidates
        ],
        expected=CANDIDATE_K,
        label="Hybrid rerank input",
    )

    candidates = [
        RerankCandidate(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            content=candidate.content,
        )
        for candidate in record.fused_candidates
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

    return HybridRerankResult(
        question_id=record.question_id,
        relevant_document_ids=(
            record.relevant_document_ids
        ),
        fused_chunk_ids=fused_chunk_ids,
        reranked_chunk_ids=reranked_chunk_ids,
        reranked_document_ids=reranked_document_ids,
        document_ranking=document_ranking,
        rerank_latency_ms=rerank_latency_ms,
        request_id=rerank_result.request_id,
        total_tokens=rerank_result.total_tokens,
    )


E1_RECALL_AT_5 = 0.6911111111111111
E1_RECALL_AT_20 = 0.8155555555555556
E1_MRR_AT_10 = 0.5672063492063492

C1_MRR_AT_10_THRESHOLD = 0.5772063492063492
C1_RECALL_AT_20_THRESHOLD = 365 / 450

DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_OUTPUT_DIR / "train_checkpoint.jsonl"
)
DEFAULT_INFLIGHT_PATH = (
    DEFAULT_OUTPUT_DIR / "train_inflight.json"
)
DEFAULT_RESULTS_PATH = (
    DEFAULT_OUTPUT_DIR / "train_results.jsonl"
)
DEFAULT_PAID_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR / "train_manifest.json"
)
DEFAULT_METRICS_PATH = (
    DEFAULT_OUTPUT_DIR / "train_metrics.json"
)
DEFAULT_COMPARISON_PATH = (
    DEFAULT_OUTPUT_DIR / "comparison.md"
)


@dataclass(frozen=True)
class HybridPaidEvalSummary:
    completed_query_count: int
    metrics: dict[str, float]
    rerank_latency_p50_ms: float
    rerank_latency_p95_ms: float
    provider_total_tokens: int
    stopped_reason: str | None
    results: tuple[HybridRerankResult, ...]


def _snapshot_record_from_payload(
    payload: Mapping[str, Any],
) -> HybridSnapshotRecord:
    return HybridSnapshotRecord(
        question_id=str(payload["question_id"]),
        question=str(payload["question"]),
        relevant_document_ids=tuple(
            str(value)
            for value in payload["relevant_document_ids"]
        ),
        dense_chunk_ids=tuple(
            str(value)
            for value in payload["dense_chunk_ids"]
        ),
        bm25_chunk_ids=tuple(
            str(value)
            for value in payload["bm25_chunk_ids"]
        ),
        fused_candidates=tuple(
            HybridCandidate(
                chunk_id=str(candidate["chunk_id"]),
                document_id=str(candidate["document_id"]),
                content=str(candidate["content"]),
            )
            for candidate in payload["fused_candidates"]
        ),
    )


def load_validated_paid_snapshot(
    *,
    snapshot_path: str | Path,
    manifest_path: str | Path,
    expected_count: int = EXPECTED_TRAIN_COUNT,
) -> list[HybridSnapshotRecord]:
    snapshot = Path(snapshot_path)
    manifest_source = Path(manifest_path)

    if not snapshot.exists():
        raise RuntimeError(
            f"frozen snapshot missing: {snapshot}"
        )

    if not manifest_source.exists():
        raise RuntimeError(
            f"snapshot manifest missing: {manifest_source}"
        )

    manifest = json.loads(
        manifest_source.read_text(encoding="utf-8")
    )

    if manifest.get("run") != "r4_c1_hybrid_rerank":
        raise RuntimeError("invalid C1 snapshot manifest run")

    if manifest.get("split") != "train":
        raise RuntimeError(
            "C1 paid path requires TRAIN-only snapshot"
        )

    manifest_count = int(manifest.get("query_count", -1))
    if manifest_count != expected_count:
        raise RuntimeError(
            "snapshot manifest query count mismatch: "
            f"expected={expected_count}, "
            f"actual={manifest_count}"
        )

    expected_sha = str(
        manifest.get("snapshot", {}).get("sha256", "")
    )
    actual_sha = _sha256(snapshot)

    if actual_sha != expected_sha:
        raise RuntimeError(
            "snapshot SHA256 mismatch: "
            f"expected={expected_sha}, actual={actual_sha}"
        )

    records: list[HybridSnapshotRecord] = []

    for line in snapshot.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        record = _snapshot_record_from_payload(
            json.loads(line)
        )

        if not record.question_id.startswith("TRAIN_"):
            raise RuntimeError(
                "C1 paid path requires TRAIN-only records"
            )

        if len(record.fused_candidates) != CANDIDATE_K:
            raise RuntimeError(
                "C1 paid snapshot requires exactly "
                "100 fused candidates"
            )

        fused_chunk_ids = tuple(
            candidate.chunk_id
            for candidate in record.fused_candidates
        )

        _require_unique_exact(
            fused_chunk_ids,
            expected=CANDIDATE_K,
            label="Hybrid paid candidate ranking",
        )

        _require_unique_exact(
            record.dense_chunk_ids,
            expected=CANDIDATE_K,
            label="Dense frozen ranking",
        )

        _require_unique_exact(
            record.bm25_chunk_ids,
            expected=CANDIDATE_K,
            label="BM25 frozen ranking",
        )

        records.append(record)

    if len(records) != expected_count:
        raise RuntimeError(
            "snapshot record count mismatch: "
            f"expected={expected_count}, "
            f"actual={len(records)}"
        )

    question_ids = [
        record.question_id
        for record in records
    ]

    if len(set(question_ids)) != expected_count:
        raise RuntimeError(
            "snapshot contains duplicate question_id values"
        )

    return records


def _paid_result_from_payload(
    payload: Mapping[str, Any],
) -> HybridRerankResult:
    return HybridRerankResult(
        question_id=str(payload["question_id"]),
        relevant_document_ids=tuple(
            str(value)
            for value in payload["relevant_document_ids"]
        ),
        fused_chunk_ids=tuple(
            str(value)
            for value in payload["fused_chunk_ids"]
        ),
        reranked_chunk_ids=tuple(
            str(value)
            for value in payload["reranked_chunk_ids"]
        ),
        reranked_document_ids=tuple(
            str(value)
            for value in payload[
                "reranked_document_ids"
            ]
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


def _load_paid_checkpoint(
    path: str | Path,
) -> list[HybridRerankResult]:
    source = Path(path)

    if not source.exists():
        return []

    results: list[HybridRerankResult] = []

    for line in source.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        results.append(
            _paid_result_from_payload(
                json.loads(line)
            )
        )

    question_ids = [
        result.question_id
        for result in results
    ]

    if len(set(question_ids)) != len(question_ids):
        raise RuntimeError(
            "paid checkpoint contains duplicate question_id"
        )

    return results


def _write_durable_json(
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


def _append_durable_result(
    path: Path,
    result: HybridRerankResult,
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


def _paid_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )
    fraction = position - lower_index

    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _build_paid_summary(
    results: Sequence[HybridRerankResult],
    *,
    stopped_reason: str | None,
) -> HybridPaidEvalSummary:
    ordered_results = sorted(
        results,
        key=lambda result: result.question_id,
    )

    if ordered_results:
        qrels_by_query = {
            result.question_id: {
                document_id: 1
                for document_id
                in result.relevant_document_ids
            }
            for result in ordered_results
        }

        run_by_query: dict[str, dict[str, float]] = {}

        for result in ordered_results:
            ranking = result.document_ranking[:20]
            ranking_size = len(ranking)

            run_by_query[result.question_id] = {
                document_id: float(
                    ranking_size - rank
                )
                for rank, document_id
                in enumerate(ranking)
            }

        metrics = evaluate_ir_run(
            build_ranx_qrels(qrels_by_query),
            Run(run_by_query),
        )
    else:
        metrics = {
            "recall@5": 0.0,
            "recall@20": 0.0,
            "mrr@10": 0.0,
        }

    latencies = [
        result.rerank_latency_ms
        for result in ordered_results
    ]

    return HybridPaidEvalSummary(
        completed_query_count=len(ordered_results),
        metrics=metrics,
        rerank_latency_p50_ms=_paid_percentile(
            latencies,
            0.50,
        ),
        rerank_latency_p95_ms=_paid_percentile(
            latencies,
            0.95,
        ),
        provider_total_tokens=sum(
            result.total_tokens or 0
            for result in ordered_results
        ),
        stopped_reason=stopped_reason,
        results=tuple(ordered_results),
    )


def run_resumable_paid_eval(
    records: Sequence[HybridSnapshotRecord],
    *,
    evaluator: Callable[
        [HybridSnapshotRecord],
        HybridRerankResult,
    ],
    checkpoint_path: str | Path,
    inflight_path: str | Path,
    token_stop_threshold: int = TOKEN_STOP_THRESHOLD,
) -> HybridPaidEvalSummary:
    checkpoint = Path(checkpoint_path)
    inflight = Path(inflight_path)

    existing = _load_paid_checkpoint(checkpoint)

    by_question_id = {
        result.question_id: result
        for result in existing
    }

    record_ids = [
        record.question_id
        for record in records
    ]

    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError(
            "paid input contains duplicate question_id"
        )

    for record in records:
        if not record.question_id.startswith("TRAIN_"):
            raise RuntimeError(
                "C1 paid path requires TRAIN-only records"
            )

        if len(record.fused_candidates) != CANDIDATE_K:
            raise RuntimeError(
                "C1 paid path requires exactly "
                "100 fused candidates"
            )

    if inflight.exists():
        inflight_payload = json.loads(
            inflight.read_text(encoding="utf-8")
        )
        inflight_question_id = str(
            inflight_payload["question_id"]
        )

        if inflight_question_id in by_question_id:
            inflight.unlink()
        else:
            raise RuntimeError(
                "inflight/uncertain provider request "
                "requires manual review; automatic "
                "replay is forbidden"
            )

    provider_total_tokens = sum(
        result.total_tokens or 0
        for result in by_question_id.values()
    )

    stopped_reason: str | None = None

    for record in records:
        question_id = record.question_id

        if question_id in by_question_id:
            continue

        if provider_total_tokens >= token_stop_threshold:
            stopped_reason = "token_stop_threshold"
            break

        _write_durable_json(
            inflight,
            {
                "question_id": question_id,
            },
        )

        # If evaluator/provider raises, the durable inflight
        # marker intentionally remains for manual review.
        result = evaluator(record)

        if result.question_id != question_id:
            raise RuntimeError(
                "provider result question_id mismatch"
            )

        _append_durable_result(
            checkpoint,
            result,
        )

        by_question_id[question_id] = result
        provider_total_tokens += (
            result.total_tokens or 0
        )

        inflight.unlink()

    return _build_paid_summary(
        list(by_question_id.values()),
        stopped_reason=stopped_reason,
    )


def evaluate_c1_gate(
    summary: HybridPaidEvalSummary,
) -> dict[str, Any]:
    complete = (
        summary.completed_query_count
        == EXPECTED_TRAIN_COUNT
    )

    passed = (
        complete
        and summary.metrics["mrr@10"]
        >= C1_MRR_AT_10_THRESHOLD
        and summary.metrics["recall@20"]
        >= C1_RECALL_AT_20_THRESHOLD
    )

    return {
        "complete_train_run": complete,
        "mrr@10_threshold": C1_MRR_AT_10_THRESHOLD,
        "recall@20_threshold": (
            C1_RECALL_AT_20_THRESHOLD
        ),
        "mrr@10_pass": (
            summary.metrics["mrr@10"]
            >= C1_MRR_AT_10_THRESHOLD
        ),
        "recall@20_pass": (
            summary.metrics["recall@20"]
            >= C1_RECALL_AT_20_THRESHOLD
        ),
        "c1_pass": passed,
    }


def write_paid_reports(
    summary: HybridPaidEvalSummary,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
    snapshot_manifest_path: str | Path = (
        DEFAULT_SNAPSHOT_MANIFEST_PATH
    ),
) -> None:
    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = output / "train_results.jsonl"

    with results_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for result in summary.results:
            file.write(
                json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                )
                + "\n"
            )

    gate = evaluate_c1_gate(summary)

    metrics_payload = {
        "query_count": summary.completed_query_count,
        "metrics": summary.metrics,
        "rerank_latency_p50_ms": (
            summary.rerank_latency_p50_ms
        ),
        "rerank_latency_p95_ms": (
            summary.rerank_latency_p95_ms
        ),
        "provider_total_tokens": (
            summary.provider_total_tokens
        ),
        "stopped_reason": summary.stopped_reason,
        "gate": gate,
    }

    _write_json(
        output / "train_metrics.json",
        metrics_payload,
    )

    snapshot_manifest = json.loads(
        Path(snapshot_manifest_path).read_text(
            encoding="utf-8"
        )
    )

    paid_manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r4_c1_hybrid_rerank",
        "split": "train",
        "query_count": (
            summary.completed_query_count
        ),
        "snapshot_sha256": _sha256(snapshot_path),
        "snapshot_manifest_sha256": _sha256(
            snapshot_manifest_path
        ),
        "candidate_contract": {
            "dense_candidate_k": CANDIDATE_K,
            "bm25_candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "fused_candidate_k": CANDIDATE_K,
        },
        "reranker": {
            "model": DEFAULT_RERANK_MODEL,
            "instruction": DEFAULT_RERANK_INSTRUCTION,
            "candidate_chunk_k": CANDIDATE_K,
            "query_normalization": "rstrip",
            "provider_region": (
                snapshot_manifest["reranker"][
                    "provider_region"
                ]
            ),
            "sdk_max_retries": 0,
        },
        "paid_safety": {
            "token_stop_threshold": (
                TOKEN_STOP_THRESHOLD
            ),
            "monetary_safety_envelope_usd": (
                MONETARY_SAFETY_ENVELOPE_USD
            ),
            "stopped_reason": (
                summary.stopped_reason
            ),
        },
        "provider_total_tokens": (
            summary.provider_total_tokens
        ),
        "gate": gate,
        "project_sha": _resolve_project_sha(),
    }

    _write_json(
        output / "train_manifest.json",
        paid_manifest,
    )

    metrics = summary.metrics

    comparison = f"""# R4 C1 Hybrid + Rerank vs E1

## Frozen E1 TRAIN baseline

- Recall@5: {E1_RECALL_AT_5:.12f}
- Recall@20: {E1_RECALL_AT_20:.12f}
- MRR@10: {E1_MRR_AT_10:.12f}

## R4 C1 TRAIN

- completed queries: {summary.completed_query_count}
- Recall@5: {metrics["recall@5"]:.12f}
- Recall@20: {metrics["recall@20"]:.12f}
- MRR@10: {metrics["mrr@10"]:.12f}
- provider tokens: {summary.provider_total_tokens}
- stopped reason: {summary.stopped_reason}

## Pre-registered executable gate

- MRR@10 >= {C1_MRR_AT_10_THRESHOLD:.16f}
- Recall@20 >= {C1_RECALL_AT_20_THRESHOLD:.16f}
- C1 PASS: {gate["c1_pass"]}

Complete historical E1 per-query evidence is not preserved,
so C1 does not rerun E1 solely to recreate paired diagnostics.
"""

    (
        output / "comparison.md"
    ).write_text(
        comparison,
        encoding="utf-8",
        newline="\n",
    )


def execute_paid_run() -> HybridPaidEvalSummary:
    records = load_validated_paid_snapshot(
        snapshot_path=DEFAULT_SNAPSHOT_PATH,
        manifest_path=DEFAULT_SNAPSHOT_MANIFEST_PATH,
        expected_count=EXPECTED_TRAIN_COUNT,
    )

    summary = run_resumable_paid_eval(
        records,
        evaluator=rerank_snapshot_record,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        inflight_path=DEFAULT_INFLIGHT_PATH,
        token_stop_threshold=TOKEN_STOP_THRESHOLD,
    )

    write_paid_reports(summary)

    return summary


def main(
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or execute the frozen TechQA "
            "R4 C1 Hybrid rerank experiment."
        )
    )
    parser.add_argument(
        "command",
        choices=("prepare", "paid"),
    )

    args = parser.parse_args(
        sys.argv[1:]
        if argv is None
        else argv
    )

    if args.command == "prepare":
        prepare_frozen_snapshot()
        return

    summary = execute_paid_run()

    print(
        json.dumps(
            {
                "completed_query_count": (
                    summary.completed_query_count
                ),
                "metrics": summary.metrics,
                "provider_total_tokens": (
                    summary.provider_total_tokens
                ),
                "stopped_reason": (
                    summary.stopped_reason
                ),
                "gate": evaluate_c1_gate(summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
