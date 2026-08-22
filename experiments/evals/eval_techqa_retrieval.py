from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
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
DEFAULT_CANDIDATE_CHUNK_K = 100
DEFAULT_DOCUMENT_TOP_K = 20

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]
Searcher = Callable[..., list[Any]]
Clock = Callable[[], float]
EvalSplit = Literal["train", "dev"]


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
    if candidate_chunk_k <= 0:
        raise ValueError("candidate_chunk_k must be greater than 0")
    if document_top_k <= 0:
        raise ValueError("document_top_k must be greater than 0")

    selected_cases = sorted(
        (case for case in cases if case.split == split),
        key=lambda case: case.question_id,
    )
    if not selected_cases:
        raise ValueError(f"No TechQA retrieval cases found for split={split}")

    results: list[TechQARetrievalResult] = []
    qrels_by_query: dict[str, dict[str, int]] = {}
    run_by_query: dict[str, dict[str, float]] = {}
    latencies_ms: list[float] = []

    for case in selected_cases:
        started = clock()
        raw_results = searcher(case.question, top_k=candidate_chunk_k)
        latency_ms = (clock() - started) * 1000.0

        raw_chunk_ids = tuple(str(result.chunk_id) for result in raw_results)
        raw_document_ids = tuple(str(result.document_id) for result in raw_results)
        document_ranking = tuple(
            collapse_chunk_results_to_document_ranking(raw_results)[:document_top_k]
        )

        results.append(
            TechQARetrievalResult(
                question_id=case.question_id,
                question=case.question,
                relevant_document_ids=case.relevant_document_ids,
                raw_chunk_ids=raw_chunk_ids,
                raw_document_ids=raw_document_ids,
                document_ranking=document_ranking,
                latency_ms=latency_ms,
            )
        )
        qrels_by_query[case.question_id] = {
            document_id: 1 for document_id in case.relevant_document_ids
        }
        ranking_size = len(document_ranking)
        run_by_query[case.question_id] = {
            document_id: float(ranking_size - rank)
            for rank, document_id in enumerate(document_ranking)
        }
        latencies_ms.append(latency_ms)

    qrels = build_ranx_qrels(qrels_by_query)
    run = Run(run_by_query)
    metrics = evaluate_ir_run(qrels, run)

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


def write_e0_reports(
    summary: TechQARetrievalEvalSummary,
    *,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    manifest_path: str | Path = DEFAULT_TECHQA_MANIFEST_PATH,
) -> None:
    """Persist the E0 run contract, raw rankings, and aggregate metrics."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    retrieval = manifest["retrieval_dataset"]
    baseline = manifest["baseline_rag"]

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
    (output_dir / "train_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "train_results.jsonl").open("w", encoding="utf-8") as file:
        for result in summary.results:
            file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    metrics_payload = {
        "query_count": summary.query_count,
        "metrics": summary.metrics,
        "latency_p50_ms": summary.latency_p50_ms,
        "latency_p95_ms": summary.latency_p95_ms,
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    print("Loading frozen TechQA retrieval cases...")
    cases = load_frozen_techqa_retrieval_cases()
    train_count = sum(case.split == "train" for case in cases)
    dev_count = sum(case.split == "dev" for case in cases)
    print(f"Loaded cases: {len(cases)} (TRAIN={train_count}, DEV={dev_count})")

    print("Running E0 dense retrieval on TRAIN only...")
    summary = evaluate_techqa_retrieval_cases(cases, split="train")
    write_e0_reports(summary)

    print("TechQA E0 TRAIN retrieval completed.")
    print(f"Queries:    {summary.query_count}")
    print(f"Recall@5:   {summary.metrics['recall@5']:.6f}")
    print(f"Recall@20:  {summary.metrics['recall@20']:.6f}")
    print(f"MRR@10:     {summary.metrics['mrr@10']:.6f}")
    print(f"p50 latency: {summary.latency_p50_ms:.3f} ms")
    print(f"p95 latency: {summary.latency_p95_ms:.3f} ms")
    print(f"Reports:    {DEFAULT_REPORT_DIR}")


if __name__ == "__main__":
    main()
