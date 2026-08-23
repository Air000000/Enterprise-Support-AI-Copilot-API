from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ranx import Run

from experiments.evals.adapters.techqa import (
    TechQARetrievalCase,
    build_qrels_by_query,
    build_techqa_retrieval_cases,
)
from experiments.evals.build_techqa_index import search_techqa_index
from experiments.evals.ir.ranx_adapter import (
    build_ranx_qrels,
    collapse_chunk_results_to_document_ranking,
    evaluate_ir_run,
)

DEFAULT_TECHQA_MANIFEST_PATH = Path(
    "experiments/evals/datasets/techqa/manifest.json"
)
DEFAULT_REPORT_DIR = Path("experiments/evals/reports/e0_dense")
DEFAULT_DEV_CHECKPOINT_PATH = DEFAULT_REPORT_DIR / "dev_checkpoint.jsonl"
DEFAULT_DEV_RUN_MANIFEST_PATH = DEFAULT_REPORT_DIR / "dev_manifest.json"
DEFAULT_CANDIDATE_CHUNK_K = 100
DEFAULT_DOCUMENT_TOP_K = 20

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]
Searcher = Callable[..., list[Any]]
Clock = Callable[[], float]
EvalSplit = Literal["train", "dev"]
RetrievalEvaluator = Callable[[TechQARetrievalCase], "TechQARetrievalResult"]


@dataclass(frozen=True)
class TechQARetrievalResult:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    raw_chunk_ids: tuple[str, ...]
    raw_document_ids: tuple[str, ...]
    document_ranking: tuple[str, ...]
    latency_ms: float


@dataclass(frozen=True)
class TechQARetrievalEvalSummary:
    split: EvalSplit
    query_count: int
    candidate_chunk_k: int
    document_top_k: int
    metrics: dict[str, float]
    latency_p50_ms: float
    latency_p95_ms: float
    results: tuple[TechQARetrievalResult, ...]


def _load_manifest(
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def load_frozen_techqa_retrieval_cases(
    *,
    dataset_loader: DatasetLoader | None = None,
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
    expected_query_count: int | None = 610,
) -> list[TechQARetrievalCase]:
    """Load TechQA retrieval queries and qrels from the frozen dataset revision."""
    manifest = _load_manifest(manifest_path)
    retrieval = manifest["retrieval_dataset"]

    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    query_rows = dataset_loader(
        retrieval["repo"],
        "queries",
        split="train",
        revision=retrieval["revision"],
    )
    qrel_rows = dataset_loader(
        retrieval["repo"],
        "default",
        split="train",
        revision=retrieval["revision"],
    )

    qrels_by_query = build_qrels_by_query(qrel_rows)
    cases = build_techqa_retrieval_cases(query_rows, qrels_by_query)

    if expected_query_count is not None and len(cases) != expected_query_count:
        raise RuntimeError(
            "Frozen TechQA retrieval query count mismatch: "
            f"expected={expected_query_count}, actual={len(cases)}"
        )

    return cases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")

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


def evaluate_techqa_retrieval_case(
    case: TechQARetrievalCase,
    *,
    searcher: Searcher = search_techqa_index,
    candidate_chunk_k: int = DEFAULT_CANDIDATE_CHUNK_K,
    document_top_k: int = DEFAULT_DOCUMENT_TOP_K,
    clock: Clock = time.perf_counter,
) -> TechQARetrievalResult:
    """Evaluate one frozen TechQA retrieval case."""
    if candidate_chunk_k <= 0:
        raise ValueError("candidate_chunk_k must be greater than 0")
    if document_top_k <= 0:
        raise ValueError("document_top_k must be greater than 0")

    started = clock()
    raw_results = searcher(case.question.rstrip(), top_k=candidate_chunk_k)
    latency_ms = (clock() - started) * 1000.0

    raw_chunk_ids = tuple(str(result.chunk_id) for result in raw_results)
    raw_document_ids = tuple(str(result.document_id) for result in raw_results)
    document_ranking = tuple(
        collapse_chunk_results_to_document_ranking(raw_results)[:document_top_k]
    )

    return TechQARetrievalResult(
        question_id=case.question_id,
        question=case.question,
        relevant_document_ids=case.relevant_document_ids,
        raw_chunk_ids=raw_chunk_ids,
        raw_document_ids=raw_document_ids,
        document_ranking=document_ranking,
        latency_ms=latency_ms,
    )


def _build_retrieval_summary(
    results: list[TechQARetrievalResult],
    *,
    split: EvalSplit,
    candidate_chunk_k: int = DEFAULT_CANDIDATE_CHUNK_K,
    document_top_k: int = DEFAULT_DOCUMENT_TOP_K,
) -> TechQARetrievalEvalSummary:
    if not results:
        raise ValueError(f"No TechQA retrieval results found for split={split}")

    qrels_by_query = {
        result.question_id: {
            document_id: 1 for document_id in result.relevant_document_ids
        }
        for result in results
    }
    run_by_query: dict[str, dict[str, float]] = {}
    for result in results:
        ranking = result.document_ranking[:document_top_k]
        ranking_size = len(ranking)
        run_by_query[result.question_id] = {
            document_id: float(ranking_size - rank)
            for rank, document_id in enumerate(ranking)
        }

    metrics = evaluate_ir_run(
        build_ranx_qrels(qrels_by_query),
        Run(run_by_query),
    )
    latencies_ms = [result.latency_ms for result in results]

    return TechQARetrievalEvalSummary(
        split=split,
        query_count=len(results),
        candidate_chunk_k=candidate_chunk_k,
        document_top_k=document_top_k,
        metrics=metrics,
        latency_p50_ms=_percentile(latencies_ms, 0.50),
        latency_p95_ms=_percentile(latencies_ms, 0.95),
        results=tuple(results),
    )


def evaluate_techqa_retrieval_cases(
    cases: Iterable[TechQARetrievalCase],
    *,
    searcher: Searcher = search_techqa_index,
    split: EvalSplit = "train",
    candidate_chunk_k: int = DEFAULT_CANDIDATE_CHUNK_K,
    document_top_k: int = DEFAULT_DOCUMENT_TOP_K,
    clock: Clock = time.perf_counter,
) -> TechQARetrievalEvalSummary:
    """Evaluate dense TechQA retrieval at document level for one split."""
    selected_cases = sorted(
        (case for case in cases if case.split == split),
        key=lambda case: case.question_id,
    )
    if not selected_cases:
        raise ValueError(f"No TechQA retrieval cases found for split={split}")

    results = [
        evaluate_techqa_retrieval_case(
            case,
            searcher=searcher,
            candidate_chunk_k=candidate_chunk_k,
            document_top_k=document_top_k,
            clock=clock,
        )
        for case in selected_cases
    ]
    return _build_retrieval_summary(
        results,
        split=split,
        candidate_chunk_k=candidate_chunk_k,
        document_top_k=document_top_k,
    )


def _result_from_payload(payload: Mapping[str, Any]) -> TechQARetrievalResult:
    return TechQARetrievalResult(
        question_id=str(payload["question_id"]),
        question=str(payload["question"]),
        relevant_document_ids=tuple(
            str(value) for value in payload["relevant_document_ids"]
        ),
        raw_chunk_ids=tuple(str(value) for value in payload["raw_chunk_ids"]),
        raw_document_ids=tuple(str(value) for value in payload["raw_document_ids"]),
        document_ranking=tuple(str(value) for value in payload["document_ranking"]),
        latency_ms=float(payload["latency_ms"]),
    )


def load_retrieval_checkpoint(
    checkpoint_path: str | Path,
) -> list[TechQARetrievalResult]:
    path = Path(checkpoint_path)
    if not path.exists():
        return []

    results: list[TechQARetrievalResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        results.append(_result_from_payload(json.loads(line)))
    return results


def run_resumable_retrieval_eval(
    cases: Iterable[TechQARetrievalCase],
    *,
    evaluator: RetrievalEvaluator,
    checkpoint_path: str | Path,
    split: EvalSplit,
) -> TechQARetrievalEvalSummary:
    """Run one retrieval split sequentially, checkpointing each successful query."""
    selected_cases = sorted(
        (case for case in cases if case.split == split),
        key=lambda case: case.question_id,
    )
    if not selected_cases:
        raise ValueError(f"No TechQA retrieval cases found for split={split}")

    selected_ids = {case.question_id for case in selected_cases}
    existing = load_retrieval_checkpoint(checkpoint_path)
    by_question_id = {
        result.question_id: result
        for result in existing
        if result.question_id in selected_ids
    }

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for case in selected_cases:
        if case.question_id in by_question_id:
            continue

        result = evaluator(case)
        with checkpoint.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        by_question_id[result.question_id] = result

    results = sorted(by_question_id.values(), key=lambda result: result.question_id)
    return _build_retrieval_summary(results, split=split)


def _dev_retrieval_contract(
    dataset_manifest: Mapping[str, Any],
    *,
    query_count: int,
) -> dict[str, Any]:
    retrieval = dataset_manifest["retrieval_dataset"]
    baseline = dataset_manifest["baseline_rag"]
    retrieval_dataset = {
        "repo": retrieval["repo"],
        "revision": retrieval["revision"],
        "corpus_sha256": retrieval["corpus_sha256"],
        "queries_sha256": retrieval["queries_sha256"],
        "qrels_sha256": retrieval["qrels_sha256"],
    }
    document_ranking_rule = (
        "collapse chunk results by document_id, retaining the first occurrence"
    )

    return {
        "benchmark": "TechQA-RAG-Eval",
        "run": "e0_dense",
        "split": "dev",
        "query_count": query_count,
        "candidate_chunk_k": DEFAULT_CANDIDATE_CHUNK_K,
        "document_top_k": DEFAULT_DOCUMENT_TOP_K,
        "query_normalization": "rstrip",
        "document_ranking_rule": document_ranking_rule,
        "retrieval_dataset": retrieval_dataset,
        "retriever": baseline["retriever"],
        "chunk_strategy": baseline["chunk_strategy"],
        "chunk_size_chars": baseline["chunk_size_chars"],
        "chunk_overlap_chars": baseline["chunk_overlap_chars"],
        "min_chunk_size_chars": baseline["min_chunk_size_chars"],
    }


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


def build_dev_retrieval_run_manifest(
    *,
    dataset_manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
    query_count: int,
    project_sha: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the immutable identity for the frozen E0 DEV retrieval run."""
    dataset_manifest = _load_manifest(dataset_manifest_path)
    contract = _dev_retrieval_contract(dataset_manifest, query_count=query_count)
    identity = {
        "run": contract["run"],
        "split": contract["split"],
        "query_count": contract["query_count"],
        "candidate_chunk_k": contract["candidate_chunk_k"],
        "document_top_k": contract["document_top_k"],
        "query_normalization": contract["query_normalization"],
        "document_ranking_rule": contract["document_ranking_rule"],
        "retrieval_dataset": contract["retrieval_dataset"],
    }

    return {
        **contract,
        "identity": identity,
        "provenance": {
            "project_sha": project_sha or _resolve_project_sha(),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        },
    }


def ensure_dev_retrieval_run_manifest(
    expected: Mapping[str, Any],
    *,
    run_manifest_path: str | Path = DEFAULT_DEV_RUN_MANIFEST_PATH,
    checkpoint_path: str | Path = DEFAULT_DEV_CHECKPOINT_PATH,
) -> None:
    """Persist or validate the frozen DEV retrieval identity before provider calls."""
    manifest_path = Path(run_manifest_path)
    if manifest_path.exists():
        persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
        if persisted.get("identity") != expected.get("identity"):
            raise RuntimeError("DEV retrieval run manifest identity mismatch")
        return

    checkpoint = Path(checkpoint_path)
    if checkpoint.exists() and checkpoint.stat().st_size > 0:
        raise RuntimeError(
            "existing DEV retrieval checkpoint has no run manifest; "
            "explicit adoption required"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(dict(expected), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_e0_reports(
    summary: TechQARetrievalEvalSummary,
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
) -> None:
    """Persist E0 raw rankings and aggregate metrics for one split."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    retrieval = manifest["retrieval_dataset"]
    baseline = manifest["baseline_rag"]
    artifact_prefix = summary.split

    run_manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "e0_dense",
        "split": summary.split,
        "query_count": summary.query_count,
        "candidate_chunk_k": summary.candidate_chunk_k,
        "document_top_k": summary.document_top_k,
        "document_ranking_rule": (
            "collapse chunk results by document_id, retaining the first occurrence"
        ),
        "retrieval_dataset": {
            "repo": retrieval["repo"],
            "revision": retrieval["revision"],
            "corpus_sha256": retrieval["corpus_sha256"],
            "queries_sha256": retrieval["queries_sha256"],
            "qrels_sha256": retrieval["qrels_sha256"],
        },
        "retriever": baseline["retriever"],
        "chunk_strategy": baseline["chunk_strategy"],
        "chunk_size_chars": baseline["chunk_size_chars"],
        "chunk_overlap_chars": baseline["chunk_overlap_chars"],
        "min_chunk_size_chars": baseline["min_chunk_size_chars"],
    }
    manifest_output = output_dir / f"{artifact_prefix}_manifest.json"
    preserve_locked_dev_manifest = False
    if artifact_prefix == "dev" and manifest_output.exists():
        persisted = json.loads(manifest_output.read_text(encoding="utf-8"))
        preserve_locked_dev_manifest = "identity" in persisted

    if not preserve_locked_dev_manifest:
        manifest_output.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with (output_dir / f"{artifact_prefix}_results.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for result in summary.results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    metrics_payload = {
        "query_count": summary.query_count,
        "metrics": summary.metrics,
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
    }
    (output_dir / f"{artifact_prefix}_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen TechQA dense retrieval.")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    args = parser.parse_args(argv)
    split: EvalSplit = args.split

    print("Loading frozen TechQA retrieval cases...")
    cases = load_frozen_techqa_retrieval_cases()
    train_count = sum(case.split == "train" for case in cases)
    dev_count = sum(case.split == "dev" for case in cases)
    print(f"Loaded cases: {len(cases)} (TRAIN={train_count}, DEV={dev_count})")

    print(f"Running E0 dense retrieval on {split.upper()} only...")
    if split == "dev":
        run_manifest = build_dev_retrieval_run_manifest(query_count=dev_count)
        ensure_dev_retrieval_run_manifest(run_manifest)
        summary = run_resumable_retrieval_eval(
            cases,
            evaluator=evaluate_techqa_retrieval_case,
            checkpoint_path=DEFAULT_DEV_CHECKPOINT_PATH,
            split="dev",
        )
    else:
        summary = evaluate_techqa_retrieval_cases(cases, split="train")

    write_e0_reports(summary)

    print(f"TechQA E0 {split.upper()} retrieval completed.")
    print(f"Queries:     {summary.query_count}")
    print(f"Recall@5:    {summary.metrics['recall@5']:.6f}")
    print(f"Recall@20:   {summary.metrics['recall@20']:.6f}")
    print(f"MRR@10:      {summary.metrics['mrr@10']:.6f}")
    print(f"p50 latency: {summary.latency_p50_ms:.3f} ms")
    print(f"p95 latency: {summary.latency_p95_ms:.3f} ms")
    print(f"Reports:     {DEFAULT_REPORT_DIR}")


if __name__ == "__main__":
    main()
