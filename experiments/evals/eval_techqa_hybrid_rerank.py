from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from experiments.evals.adapters.techqa import build_techqa_documents
from experiments.evals.eval_techqa_chunk_bm25 import (
    EXPECTED_SPLITTER_BLOB_SHA,
    EXPECTED_TECHQA_CHUNK_COUNT,
)
from experiments.evals.eval_techqa_rerank import (
    load_frozen_e0_rerank_records,
)
from experiments.evals.ir.rrf import fuse_rrf
from experiments.evals.rerankers.qwen3_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
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


def main(
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen TechQA R4 C1 Hybrid "
            "candidate snapshot."
        )
    )
    parser.add_argument(
        "command",
        choices=("prepare",),
    )

    args = parser.parse_args(
        sys.argv[1:] if argv is None else argv
    )

    if args.command == "prepare":
        prepare_frozen_snapshot()


if __name__ == "__main__":
    main()
