from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ranx import Run

from experiments.evals.build_techqa_index import (
    DEFAULT_TECHQA_CHROMA_DIR,
    DEFAULT_TECHQA_COLLECTION_NAME,
)
from experiments.evals.ir.ranx_adapter import (
    build_ranx_qrels,
    collapse_chunk_results_to_document_ranking,
    evaluate_ir_run,
)
from experiments.evals.rerankers.qwen3_reranker import (
    DEFAULT_RERANK_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
    RerankCandidate,
    rerank_candidates,
)
from rag_runtime.build_chroma_index import get_chroma_client

DEFAULT_E0_RERANK_RESULTS_PATH = Path(
    "experiments/evals/reports/e0_dense/train_results.jsonl"
)
DEFAULT_E0_RERANK_MANIFEST_PATH = Path(
    "experiments/evals/reports/e0_dense/train_manifest.json"
)
DEFAULT_E0_DEV_RERANK_RESULTS_PATH = Path(
    "experiments/evals/reports/e0_dense/dev_results.jsonl"
)
DEFAULT_E0_DEV_RERANK_MANIFEST_PATH = Path(
    "experiments/evals/reports/e0_dense/dev_manifest.json"
)
DEFAULT_RERANK_REPORT_DIR = Path("experiments/evals/reports/e1_rerank")
DEFAULT_RERANK_CHECKPOINT_PATH = DEFAULT_RERANK_REPORT_DIR / "train_checkpoint.jsonl"
DEFAULT_RERANK_RUN_MANIFEST_PATH = DEFAULT_RERANK_REPORT_DIR / "train_manifest.json"
DEFAULT_RERANK_DEV_CHECKPOINT_PATH = DEFAULT_RERANK_REPORT_DIR / "dev_checkpoint.jsonl"
DEFAULT_RERANK_DEV_RUN_MANIFEST_PATH = DEFAULT_RERANK_REPORT_DIR / "dev_manifest.json"
DEFAULT_RERANK_TRAIN_COUNT = 450
DEFAULT_RERANK_DEV_COUNT = 160


class FrozenCandidatePoolError(RuntimeError):
    """Raised when the frozen E0 candidate pool cannot be reconstructed exactly."""


@dataclass(frozen=True)
class TechQARerankResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    dense_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    reranked_document_ids: tuple[str, ...]
    document_ranking: tuple[str, ...]
    rerank_latency_ms: float
    request_id: str | None
    total_tokens: int | None


@dataclass(frozen=True)
class TechQARerankEvalSummary:
    query_count: int
    metrics: dict[str, float]
    rerank_latency_p50_ms: float
    rerank_latency_p95_ms: float
    provider_total_tokens: int
    results: tuple[TechQARerankResult, ...]


Reranker = Callable[[str, list[RerankCandidate]], Any]
Clock = Callable[[], float]
RerankEvaluator = Callable[[Mapping[str, Any]], TechQARerankResult]


def load_frozen_e0_rerank_records(
    path: str | Path = DEFAULT_E0_RERANK_RESULTS_PATH,
    *,
    expected_count: int = DEFAULT_RERANK_TRAIN_COUNT,
) -> list[dict[str, Any]]:
    """Load a frozen E0 retrieval result set used as the E1 candidate pool."""
    source = Path(path)
    records = [
        dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != expected_count:
        raise RuntimeError(
            "Frozen E0 record count mismatch: "
            f"expected={expected_count}, actual={len(records)}"
        )
    return records


def open_frozen_techqa_collection() -> Any:
    """Open the existing isolated TechQA E0 Chroma collection without rebuilding it."""
    client = get_chroma_client(DEFAULT_TECHQA_CHROMA_DIR)
    return client.get_collection(
        name=DEFAULT_TECHQA_COLLECTION_NAME,
        embedding_function=None,
    )


def _rehydrate_candidates(
    record: Mapping[str, Any],
    *,
    collection: Any,
) -> list[RerankCandidate]:
    dense_chunk_ids = [str(chunk_id) for chunk_id in record["raw_chunk_ids"]]
    response = collection.get(
        ids=dense_chunk_ids,
        include=["documents", "metadatas"],
    )

    by_chunk_id: dict[str, RerankCandidate] = {}
    for chunk_id, content, metadata in zip(
        response["ids"],
        response["documents"],
        response["metadatas"],
    ):
        chunk_key = str(chunk_id)
        by_chunk_id[chunk_key] = RerankCandidate(
            chunk_id=chunk_key,
            document_id=str(metadata["document_id"]),
            content=str(content),
        )

    missing = [chunk_id for chunk_id in dense_chunk_ids if chunk_id not in by_chunk_id]
    if missing:
        raise FrozenCandidatePoolError(
            "missing frozen Chroma candidate(s): " + ", ".join(missing)
        )

    return [by_chunk_id[chunk_id] for chunk_id in dense_chunk_ids]


def rerank_frozen_e0_record(
    record: Mapping[str, Any],
    *,
    collection: Any,
    reranker: Reranker = rerank_candidates,
    clock: Clock = time.perf_counter,
) -> TechQARerankResult:
    """Rerank one frozen E0 candidate pool without rerunning dense retrieval."""
    candidates = _rehydrate_candidates(record, collection=collection)

    started = clock()
    rerank_result = reranker(str(record["question"]).rstrip(), candidates)
    rerank_latency_ms = (clock() - started) * 1000.0

    reranked_chunk_ids = tuple(
        str(candidate.chunk_id) for candidate in rerank_result.results
    )
    reranked_document_ids = tuple(
        str(candidate.document_id) for candidate in rerank_result.results
    )
    document_ranking = tuple(
        collapse_chunk_results_to_document_ranking(rerank_result.results)
    )

    return TechQARerankResult(
        question_id=str(record["question_id"]),
        relevant_document_ids=tuple(
            str(document_id) for document_id in record["relevant_document_ids"]
        ),
        dense_chunk_ids=tuple(candidate.chunk_id for candidate in candidates),
        reranked_chunk_ids=reranked_chunk_ids,
        reranked_document_ids=reranked_document_ids,
        document_ranking=document_ranking,
        rerank_latency_ms=rerank_latency_ms,
        request_id=rerank_result.request_id,
        total_tokens=rerank_result.total_tokens,
    )


def _result_from_payload(payload: Mapping[str, Any]) -> TechQARerankResult:
    return TechQARerankResult(
        question_id=str(payload["question_id"]),
        relevant_document_ids=tuple(
            str(value) for value in payload["relevant_document_ids"]
        ),
        dense_chunk_ids=tuple(str(value) for value in payload["dense_chunk_ids"]),
        reranked_chunk_ids=tuple(
            str(value) for value in payload["reranked_chunk_ids"]
        ),
        reranked_document_ids=tuple(
            str(value) for value in payload["reranked_document_ids"]
        ),
        document_ranking=tuple(str(value) for value in payload["document_ranking"]),
        rerank_latency_ms=float(payload["rerank_latency_ms"]),
        request_id=(
            None if payload.get("request_id") is None else str(payload["request_id"])
        ),
        total_tokens=(
            None if payload.get("total_tokens") is None else int(payload["total_tokens"])
        ),
    )


def load_rerank_checkpoint(
    checkpoint_path: str | Path,
) -> list[TechQARerankResult]:
    path = Path(checkpoint_path)
    if not path.exists():
        return []

    results: list[TechQARerankResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        results.append(_result_from_payload(json.loads(line)))
    return results


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _build_summary(results: list[TechQARerankResult]) -> TechQARerankEvalSummary:
    qrels_by_query = {
        result.question_id: {
            document_id: 1 for document_id in result.relevant_document_ids
        }
        for result in results
    }
    run_by_query: dict[str, dict[str, float]] = {}
    for result in results:
        ranking = result.document_ranking[:20]
        ranking_size = len(ranking)
        run_by_query[result.question_id] = {
            document_id: float(ranking_size - rank)
            for rank, document_id in enumerate(ranking)
        }

    metrics = evaluate_ir_run(
        build_ranx_qrels(qrels_by_query),
        Run(run_by_query),
    )
    latencies = [result.rerank_latency_ms for result in results]
    total_tokens = sum(result.total_tokens or 0 for result in results)

    return TechQARerankEvalSummary(
        query_count=len(results),
        metrics=metrics,
        rerank_latency_p50_ms=_percentile(latencies, 0.50),
        rerank_latency_p95_ms=_percentile(latencies, 0.95),
        provider_total_tokens=total_tokens,
        results=tuple(results),
    )


def run_resumable_rerank_eval(
    records: Iterable[Mapping[str, Any]],
    *,
    evaluator: RerankEvaluator,
    checkpoint_path: str | Path,
) -> TechQARerankEvalSummary:
    checkpoint = Path(checkpoint_path)
    existing = load_rerank_checkpoint(checkpoint)
    by_question_id = {result.question_id: result for result in existing}

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        question_id = str(record["question_id"])
        if question_id in by_question_id:
            continue

        result = evaluator(record)
        with checkpoint.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        by_question_id[result.question_id] = result

    results = sorted(by_question_id.values(), key=lambda result: result.question_id)
    return _build_summary(results)


def write_rerank_reports(
    summary: TechQARerankEvalSummary,
    *,
    report_dir: str | Path,
    split: str = "train",
) -> None:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = split

    with (output_dir / f"{artifact_prefix}_results.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for result in summary.results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    metrics = {
        "query_count": summary.query_count,
        "metrics": summary.metrics,
        "rerank_latency_p50_ms": summary.rerank_latency_p50_ms,
        "rerank_latency_p95_ms": summary.rerank_latency_p95_ms,
        "provider_total_tokens": summary.provider_total_tokens,
    }
    (output_dir / f"{artifact_prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_rerank_run_manifest(
    *,
    e0_results_path: str | Path,
    e0_manifest_path: str | Path,
    project_sha: str,
    created_at: str,
    split: str = "train",
    query_count: int | None = None,
) -> dict[str, Any]:
    e0_manifest = json.loads(Path(e0_manifest_path).read_text(encoding="utf-8"))
    candidate_chunk_k = int(e0_manifest["candidate_chunk_k"])

    identity: dict[str, Any] = {
        "run": "e1_rerank",
        "split": split,
        "source_e0": {
            "results_sha256": _sha256(e0_results_path),
            "manifest_sha256": _sha256(e0_manifest_path),
        },
        "reranker": {
            "model": DEFAULT_RERANK_MODEL,
            "instruction": DEFAULT_RERANK_INSTRUCTION,
            "candidate_chunk_k": candidate_chunk_k,
            "query_normalization": "rstrip",
        },
        "document_ranking": {
            "top_k": 20,
            "rule": "first occurrence of document_id after chunk rerank",
        },
    }
    if query_count is not None:
        identity["query_count"] = query_count

    return {
        "identity": identity,
        "provenance": {
            "project_sha": project_sha,
            "created_at": created_at,
        },
    }


def ensure_rerank_run_manifest(
    expected: Mapping[str, Any],
    *,
    run_manifest_path: str | Path,
    checkpoint_path: str | Path,
) -> None:
    manifest_path = Path(run_manifest_path)
    if manifest_path.exists():
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        if persisted.get("identity") != expected.get("identity"):
            raise RuntimeError("rerank run manifest identity mismatch")
        return

    checkpoint = Path(checkpoint_path)
    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        raise RuntimeError(
            "existing rerank checkpoint has no run manifest; explicit adoption required"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(dict(expected), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_project_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    project_sha = result.stdout.strip()
    if not project_sha:
        raise RuntimeError("Unable to resolve project SHA")
    return project_sha


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen TechQA reranking.")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    args = parser.parse_args([] if argv is None else argv)
    split = str(args.split)

    if split == "dev":
        e0_results_path = DEFAULT_E0_DEV_RERANK_RESULTS_PATH
        e0_manifest_path = DEFAULT_E0_DEV_RERANK_MANIFEST_PATH
        checkpoint_path = DEFAULT_RERANK_DEV_CHECKPOINT_PATH
        run_manifest_path = DEFAULT_RERANK_DEV_RUN_MANIFEST_PATH
        expected_count = DEFAULT_RERANK_DEV_COUNT
    else:
        e0_results_path = DEFAULT_E0_RERANK_RESULTS_PATH
        e0_manifest_path = DEFAULT_E0_RERANK_MANIFEST_PATH
        checkpoint_path = DEFAULT_RERANK_CHECKPOINT_PATH
        run_manifest_path = DEFAULT_RERANK_RUN_MANIFEST_PATH
        expected_count = DEFAULT_RERANK_TRAIN_COUNT

    print(f"Loading frozen E0 {split.upper()} candidate pools...")
    records = load_frozen_e0_rerank_records(
        e0_results_path,
        expected_count=expected_count,
    )

    print("Validating E1 rerank run identity...")
    if split == "dev":
        run_manifest = build_rerank_run_manifest(
            e0_results_path=e0_results_path,
            e0_manifest_path=e0_manifest_path,
            project_sha=_resolve_project_sha(),
            created_at=datetime.now(timezone.utc).isoformat(),
            split="dev",
            query_count=expected_count,
        )
    else:
        run_manifest = build_rerank_run_manifest(
            e0_results_path=e0_results_path,
            e0_manifest_path=e0_manifest_path,
            project_sha=_resolve_project_sha(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    ensure_rerank_run_manifest(
        run_manifest,
        run_manifest_path=run_manifest_path,
        checkpoint_path=checkpoint_path,
    )

    print("Opening frozen TechQA E0 Chroma collection...")
    collection = open_frozen_techqa_collection()

    def evaluator(record: Mapping[str, Any]) -> TechQARerankResult:
        return rerank_frozen_e0_record(record, collection=collection)

    print(f"Running resumable E1 qwen3-rerank evaluation on {split.upper()}...")
    summary = run_resumable_rerank_eval(
        records,
        evaluator=evaluator,
        checkpoint_path=checkpoint_path,
    )
    if split == "dev":
        write_rerank_reports(
            summary,
            report_dir=DEFAULT_RERANK_REPORT_DIR,
            split="dev",
        )
    else:
        write_rerank_reports(summary, report_dir=DEFAULT_RERANK_REPORT_DIR)

    print("TechQA E1 rerank evaluation completed.")
    print(f"Query count:              {summary.query_count}")
    print(f"Recall@5:                 {summary.metrics['recall@5']:.6f}")
    print(f"Recall@20:                {summary.metrics['recall@20']:.6f}")
    print(f"MRR@10:                   {summary.metrics['mrr@10']:.6f}")
    print(f"Rerank latency p50 (ms):  {summary.rerank_latency_p50_ms:.3f}")
    print(f"Rerank latency p95 (ms):  {summary.rerank_latency_p95_ms:.3f}")
    print(f"Provider total tokens:    {summary.provider_total_tokens}")


if __name__ == "__main__":
    main(sys.argv[1:])
