from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import json
from pathlib import Path

import hashlib
import subprocess
from importlib.metadata import version
from experiments.evals.adapters.techqa import (
    TechQARetrievalCase,
    build_techqa_documents,
)
from experiments.evals.retrievers.bm25_techqa_chunks import (
    TechQAChunkBM25Retriever,
    build_techqa_chunks,
)

EXPECTED_SPLITTER_BLOB_SHA = (
    "64026b4434f1eea46b95bfce9f667680a37a2103"
)
EXPECTED_TECHQA_CHUNK_COUNT = 172614
EXPECTED_TECHQA_TRAIN_QUERY_COUNT = 450

DEFAULT_TECHQA_MANIFEST_PATH = Path(
    "experiments/evals/datasets/techqa/manifest.json"
)
DEFAULT_E0_TRAIN_RESULTS_PATH = Path(
    "experiments/evals/reports/e0_dense/train_results.jsonl"
)
DEFAULT_R3_MANIFEST_PATH = Path(
    "experiments/evals/reports/r3_hybrid/train_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/evals/reports/r4_chunk_bm25_audit"
)

@dataclass(frozen=True)
class ChunkCutoffObservation:
    cutoff: int
    returned_chunk_count: int
    unique_document_count: int
    duplicate_slot_count: int
    duplicate_ratio: float
    gold_document_hit_within_chunk_k: bool
    crowding_rescue: bool


def collapse_document_ids(
    document_ids: Sequence[str],
) -> list[str]:
    collapsed: list[str] = []
    seen: set[str] = set()

    for document_id in document_ids:
        value = str(document_id)

        if value in seen:
            continue

        seen.add(value)
        collapsed.append(value)

    return collapsed


def first_relevant_rank(
    document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
) -> int | None:
    relevant = {
        str(document_id)
        for document_id in relevant_document_ids
    }

    for rank, document_id in enumerate(
        document_ids,
        start=1,
    ):
        if str(document_id) in relevant:
            return rank

    return None


@dataclass(frozen=True)
class TechQAChunkRef:
    chunk_id: str
    document_id: str
    chunk_index: int


@dataclass(frozen=True)
class AdaptiveSearchResult:
    candidates: tuple[object, ...]
    final_search_depth: int
    latency_ms: float

RAW_CUTOFFS = (20, 50, 100)


@dataclass(frozen=True)
class TechQAChunkBM25AuditResult:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    audit_search_depth: int
    latency_ms: float
    raw_top100_chunk_ids: tuple[str, ...]
    raw_top100_document_ids: tuple[str, ...]
    document_top100: tuple[str, ...]
    first_gold_chunk_rank: int | None
    first_gold_document_rank: int | None
    crowding_gap: int | None
    cutoff_observations: tuple[ChunkCutoffObservation, ...]

def retrieve_with_unique_document_depth(
    query: str,
    *,
    searcher: Callable[..., list[object]],
    clock: Callable[[], float] = time.perf_counter,
    initial_depth: int = 500,
    required_unique_documents: int = 100,
    max_depth: int = 172614,
) -> AdaptiveSearchResult:
    if (
        initial_depth <= 0
        or required_unique_documents <= 0
        or max_depth <= 0
    ):
        raise ValueError(
            "audit depth values must be greater than 0"
        )

    depth = min(initial_depth, max_depth)
    total_latency_ms = 0.0
    latest: list[object] = []

    while True:
        started = clock()

        latest = searcher(
            query,
            top_k=depth,
        )

        total_latency_ms += (
            clock() - started
        ) * 1000.0

        unique_document_count = len(
            collapse_document_ids(
                [
                    str(candidate.document_id)
                    for candidate in latest
                ]
            )
        )

        if unique_document_count >= required_unique_documents:
            break

        if depth >= max_depth or len(latest) < depth:
            break

        depth = min(
            depth * 2,
            max_depth,
        )

    return AdaptiveSearchResult(
        candidates=tuple(latest),
        final_search_depth=depth,
        latency_ms=total_latency_ms,
    )

def evaluate_audit_case(
    case: TechQARetrievalCase,
    *,
    searcher: Callable[..., list[object]],
    clock: Callable[[], float] = time.perf_counter,
    max_depth: int = 172614,
) -> TechQAChunkBM25AuditResult:
    search_result = retrieve_with_unique_document_depth(
        case.question.rstrip(),
        searcher=searcher,
        clock=clock,
        max_depth=max_depth,
    )

    raw_chunk_ids = tuple(
        str(candidate.chunk_id)
        for candidate in search_result.candidates
    )
    raw_document_ids = tuple(
        str(candidate.document_id)
        for candidate in search_result.candidates
    )

    collapsed_document_ids = tuple(
        collapse_document_ids(raw_document_ids)
    )

    first_gold_chunk_rank = first_relevant_rank(
        raw_document_ids,
        case.relevant_document_ids,
    )
    first_gold_document_rank = first_relevant_rank(
        collapsed_document_ids,
        case.relevant_document_ids,
    )

    crowding_gap = (
        first_gold_chunk_rank - first_gold_document_rank
        if (
            first_gold_chunk_rank is not None
            and first_gold_document_rank is not None
        )
        else None
    )

    cutoff_observations = tuple(
        build_cutoff_observation(
            raw_document_ids=raw_document_ids,
            relevant_document_ids=case.relevant_document_ids,
            cutoff=cutoff,
        )
        for cutoff in RAW_CUTOFFS
    )

    return TechQAChunkBM25AuditResult(
        question_id=case.question_id,
        question=case.question,
        relevant_document_ids=case.relevant_document_ids,
        audit_search_depth=search_result.final_search_depth,
        latency_ms=search_result.latency_ms,
        raw_top100_chunk_ids=raw_chunk_ids[:100],
        raw_top100_document_ids=raw_document_ids[:100],
        document_top100=collapsed_document_ids[:100],
        first_gold_chunk_rank=first_gold_chunk_rank,
        first_gold_document_rank=first_gold_document_rank,
        crowding_gap=crowding_gap,
        cutoff_observations=cutoff_observations,
    )

def build_cutoff_observation(
    raw_document_ids: Sequence[str],
    relevant_document_ids: Sequence[str],
    cutoff: int,
) -> ChunkCutoffObservation:
    raw_prefix = [
        str(document_id)
        for document_id in raw_document_ids[:cutoff]
    ]

    returned_chunk_count = len(raw_prefix)
    unique_document_count = len(set(raw_prefix))
    duplicate_slot_count = (
        returned_chunk_count - unique_document_count
    )
    duplicate_ratio = (
        duplicate_slot_count / returned_chunk_count
        if returned_chunk_count
        else 0.0
    )

    first_gold_chunk_rank = first_relevant_rank(
        raw_document_ids,
        relevant_document_ids,
    )

    collapsed_document_ids = collapse_document_ids(
        raw_document_ids
    )
    first_gold_document_rank = first_relevant_rank(
        collapsed_document_ids,
        relevant_document_ids,
    )

    gold_document_hit_within_chunk_k = (
        first_gold_chunk_rank is not None
        and first_gold_chunk_rank <= cutoff
    )

    crowding_rescue = (
        first_gold_chunk_rank is not None
        and first_gold_document_rank is not None
        and first_gold_chunk_rank > cutoff
        and first_gold_document_rank <= cutoff
    )

    return ChunkCutoffObservation(
        cutoff=cutoff,
        returned_chunk_count=returned_chunk_count,
        unique_document_count=unique_document_count,
        duplicate_slot_count=duplicate_slot_count,
        duplicate_ratio=duplicate_ratio,
        gold_document_hit_within_chunk_k=(
            gold_document_hit_within_chunk_k
        ),
        crowding_rescue=crowding_rescue,
    )

def percentile(
    values: Sequence[float],
    p: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )
    fraction = position - lower_index

    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def build_audit_summary(
    results: Sequence[TechQAChunkBM25AuditResult],
) -> dict[str, object]:
    query_count = len(results)

    cutoffs: dict[str, dict[str, float | int]] = {}

    for cutoff in RAW_CUTOFFS:
        observations = [
            next(
                observation
                for observation in result.cutoff_observations
                if observation.cutoff == cutoff
            )
            for result in results
        ]

        unique_document_counts = [
            observation.unique_document_count
            for observation in observations
        ]
        duplicate_ratios = [
            observation.duplicate_ratio
            for observation in observations
        ]

        gold_hit_count = sum(
            observation.gold_document_hit_within_chunk_k
            for observation in observations
        )
        crowding_rescue_count = sum(
            observation.crowding_rescue
            for observation in observations
        )

        cutoffs[str(cutoff)] = {
            "unique_document_count_p05": percentile(
                unique_document_counts,
                0.05,
            ),
            "unique_document_count_p50": percentile(
                unique_document_counts,
                0.50,
            ),
            "duplicate_ratio_p50": percentile(
                duplicate_ratios,
                0.50,
            ),
            "duplicate_ratio_p95": percentile(
                duplicate_ratios,
                0.95,
            ),
            "gold_document_hit_count": gold_hit_count,
            "gold_document_hit_rate": (
                gold_hit_count / query_count
                if query_count
                else 0.0
            ),
            "crowding_rescue_count": crowding_rescue_count,
            "crowding_rescue_rate": (
                crowding_rescue_count / query_count
                if query_count
                else 0.0
            ),
        }

    collapsed_document_recall: dict[str, float] = {}

    for cutoff in RAW_CUTOFFS:
        hit_count = sum(
            first_relevant_rank(
                result.document_top100[:cutoff],
                result.relevant_document_ids,
            )
            is not None
            for result in results
        )

        collapsed_document_recall[f"recall@{cutoff}"] = (
            hit_count / query_count
            if query_count
            else 0.0
        )

    crowding_gaps = [
        result.crowding_gap
        for result in results
        if result.crowding_gap is not None
    ]

    latencies_ms = [
        result.latency_ms
        for result in results
    ]

    return {
        "query_count": query_count,
        "cutoffs": cutoffs,
        "collapsed_document_recall": (
            collapsed_document_recall
        ),
        "crowding_gap": {
            "observed_count": len(crowding_gaps),
            "p50": percentile(crowding_gaps, 0.50),
            "p95": percentile(crowding_gaps, 0.95),
        },
        "latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
        },
    }

def build_diagnostic_cases(
    results: Sequence[TechQAChunkBM25AuditResult],
    *,
    limit_per_group: int = 10,
) -> dict[str, list[dict[str, object]]]:
    def build_record(
        result: TechQAChunkBM25AuditResult,
        observation: ChunkCutoffObservation,
    ) -> dict[str, object]:
        return {
            "question_id": result.question_id,
            "question": result.question,
            "cutoff": observation.cutoff,
            "returned_chunk_count": (
                observation.returned_chunk_count
            ),
            "unique_document_count": (
                observation.unique_document_count
            ),
            "duplicate_ratio": observation.duplicate_ratio,
            "relevant_document_ids": (
                result.relevant_document_ids
            ),
            "first_gold_chunk_rank": (
                result.first_gold_chunk_rank
            ),
            "first_gold_document_rank": (
                result.first_gold_document_rank
            ),
            "crowding_gap": result.crowding_gap,
            "crowding_rescue": observation.crowding_rescue,
            "raw_top100_chunk_ids": (
                result.raw_top100_chunk_ids
            ),
            "raw_top100_document_ids": (
                result.raw_top100_document_ids
            ),
        }

    high_duplication_cases: list[dict[str, object]] = []
    crowding_rescue_cases: list[dict[str, object]] = []

    for result in results:
        top100_observation = next(
            observation
            for observation in result.cutoff_observations
            if observation.cutoff == 100
        )

        high_duplication_cases.append(
            build_record(
                result,
                top100_observation,
            )
        )

        for observation in result.cutoff_observations:
            if observation.crowding_rescue:
                crowding_rescue_cases.append(
                    build_record(
                        result,
                        observation,
                    )
                )

    high_duplication_cases.sort(
        key=lambda row: (
            -float(row["duplicate_ratio"]),
            str(row["question_id"]),
        )
    )

    def crowding_sort_key(
        row: dict[str, object],
    ) -> tuple[int, int, str]:
        crowding_gap = row["crowding_gap"]
        gap = (
            int(crowding_gap)
            if crowding_gap is not None
            else -1
        )

        return (
            int(row["cutoff"]),
            -gap,
            str(row["question_id"]),
        )

    crowding_rescue_cases.sort(
        key=crowding_sort_key
    )

    return {
        "high_duplication_cases": (
            high_duplication_cases[:limit_per_group]
        ),
        "crowding_rescue_cases": (
            crowding_rescue_cases[:limit_per_group]
        ),
    }

def load_frozen_train_cases_from_e0(
    path: str | Path,
    *,
    expected_count: int | None = 450,
) -> list[TechQARetrievalCase]:
    cases: list[TechQARetrievalCase] = []
    seen_question_ids: set[str] = set()

    for line in Path(path).read_text(
        encoding="utf-8",
    ).splitlines():
        if not line.strip():
            continue

        payload = json.loads(line)
        question_id = str(payload["question_id"])

        if not question_id.startswith("TRAIN_"):
            raise RuntimeError(
                "R4 chunk BM25 audit requires TRAIN-only input"
            )

        if question_id in seen_question_ids:
            raise RuntimeError(
                "duplicate TRAIN question_id: "
                f"{question_id}"
            )

        seen_question_ids.add(question_id)

        relevant_document_ids = tuple(
            str(document_id)
            for document_id in payload[
                "relevant_document_ids"
            ]
        )

        if len(relevant_document_ids) != 1:
            raise RuntimeError(
                "TRAIN case requires exactly one relevant document: "
                f"question_id={question_id}, "
                f"count={len(relevant_document_ids)}"
            )

        cases.append(
            TechQARetrievalCase(
                question_id=question_id,
                question=str(payload["question"]),
                relevant_document_ids=relevant_document_ids,
                split="train",
            )
        )

    if (
        expected_count is not None
        and len(cases) != expected_count
    ):
        raise RuntimeError(
            "TRAIN query count mismatch: "
            f"expected={expected_count}, actual={len(cases)}"
        )

    return cases

def run_preflight(
    *,
    train_results_path: str | Path,
    r3_manifest_path: str | Path,
    dataset_manifest_path: str | Path,
    train_query_count: int,
    observed_chunk_count: int,
    splitter_blob_loader: Callable[[], str],
    version_loader: Callable[[str], str],
) -> dict[str, object]:
    actual_train_results_sha = hashlib.sha256(
        Path(train_results_path).read_bytes()
    ).hexdigest()

    r3_manifest = json.loads(
        Path(r3_manifest_path).read_text(
            encoding="utf-8",
        )
    )
    historical_train_results_sha = str(
        r3_manifest["dense_source"]["results_sha256"]
    )

    train_results_sha_matches_historical = (
        actual_train_results_sha
        == historical_train_results_sha
    )

    bm25_version = version_loader("bm25s")

    if bm25_version != "0.3.10":
        raise RuntimeError(
            "bm25s version mismatch: "
            f"expected=0.3.10, actual={bm25_version}"
        )

    splitter_blob_sha = splitter_blob_loader()

    if splitter_blob_sha != EXPECTED_SPLITTER_BLOB_SHA:
        raise RuntimeError(
            "splitter blob mismatch: "
            f"expected={EXPECTED_SPLITTER_BLOB_SHA}, "
            f"actual={splitter_blob_sha}"
        )

    if observed_chunk_count != EXPECTED_TECHQA_CHUNK_COUNT:
        raise RuntimeError(
            "TechQA chunk count mismatch: "
            f"expected={EXPECTED_TECHQA_CHUNK_COUNT}, "
            f"actual={observed_chunk_count}"
        )

    if train_query_count != EXPECTED_TECHQA_TRAIN_QUERY_COUNT:
        raise RuntimeError(
            "TRAIN query count mismatch: "
            f"expected={EXPECTED_TECHQA_TRAIN_QUERY_COUNT}, "
            f"actual={train_query_count}"
        )

    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(
            encoding="utf-8",
        )
    )

    return {
        "split": "train",
        "query_count": train_query_count,
        "chunk_count": observed_chunk_count,
        "bm25_version": bm25_version,
        "splitter_blob_sha": splitter_blob_sha,
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "historical_e0_train_results_sha256": (
            historical_train_results_sha
        ),
        "input_e0_train_results_sha256": (
            actual_train_results_sha
        ),
        "e0_train_results_sha_matches_historical": (
            train_results_sha_matches_historical
        ),
        "retrieval_dataset": (
            dataset_manifest["retrieval_dataset"]
        ),
    }

def build_run_manifest(
    preflight_report: dict[str, object],
) -> dict[str, object]:
    return {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r4_chunk_bm25_candidate_audit",
        "split": preflight_report["split"],
        "query_count": preflight_report["query_count"],
        "provider_calls": preflight_report["provider_calls"],
        "dev_artifact_opened": preflight_report[
            "dev_artifact_opened"
        ],
        "chunking": {
            "strategy": "paragraph_aware_character",
            "chunk_size": 800,
            "chunk_overlap": 120,
            "min_chunk_size": 150,
            "splitter_blob_sha": preflight_report[
                "splitter_blob_sha"
            ],
            "observed_chunk_count": preflight_report[
                "chunk_count"
            ],
        },
        "bm25": {
            "library": "bm25s",
            "version": preflight_report["bm25_version"],
            "method": "lucene",
            "k1": 1.5,
            "b": 0.75,
            "backend": "numpy",
            "indexed_unit": "chunk",
            "tokenizer_regex": (
                r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
            ),
            "query_normalization": "rstrip",
        },
        "audit": {
            "raw_chunk_cutoffs": [20, 50, 100],
            "initial_search_depth": 500,
            "required_unique_documents": 100,
            "depth_growth_factor": 2,
            "max_search_depth": EXPECTED_TECHQA_CHUNK_COUNT,
        },
        "retrieval_dataset": preflight_report[
            "retrieval_dataset"
        ],
        "input_e0_train_results_sha256": preflight_report[
            "input_e0_train_results_sha256"
        ],
        "historical_e0_train_results_sha256": (
            preflight_report.get(
                "historical_e0_train_results_sha256"
            )
        ),
        "e0_train_results_sha_matches_historical": (
            preflight_report.get(
                "e0_train_results_sha_matches_historical"
            )
        ),
    }

def run_audit_cases(
    *,
    cases: Sequence[TechQARetrievalCase],
    chunks: Sequence[TechQAChunkRef],
    retriever_factory: Callable[..., object],
) -> tuple[list[TechQAChunkBM25AuditResult], float]:
    started = time.perf_counter()

    retriever = retriever_factory(chunks)

    index_build_seconds = (
        time.perf_counter() - started
    )

    results: list[TechQAChunkBM25AuditResult] = []

    for case in cases:
        result = evaluate_audit_case(
            case,
            searcher=retriever.search,
        )
        results.append(result)

    return results, index_build_seconds

def write_audit_artifacts(
    *,
    output_dir: str | Path,
    manifest: dict[str, object],
    metrics: dict[str, object],
    results: Sequence[TechQAChunkBM25AuditResult],
    diagnostics: dict[str, object],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_path / "train_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    (output_path / "train_metrics.json").write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    results_text = "".join(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
        )
        + "\n"
        for result in results
    )
    (output_path / "train_results.jsonl").write_text(
        results_text,
        encoding="utf-8",
    )

    (output_path / "diagnostic_cases.json").write_text(
        json.dumps(
            diagnostics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

def execute_audit(
    *,
    cases: Sequence[TechQARetrievalCase],
    chunks: Sequence[TechQAChunkRef],
    preflight_report: dict[str, object],
    output_dir: str | Path,
    retriever_factory: Callable[..., object],
) -> dict[str, object]:
    results, index_build_seconds = run_audit_cases(
        cases=cases,
        chunks=chunks,
        retriever_factory=retriever_factory,
    )

    metrics = build_audit_summary(results)
    metrics["index_build_seconds"] = index_build_seconds

    diagnostics = build_diagnostic_cases(results)
    manifest = build_run_manifest(preflight_report)

    write_audit_artifacts(
        output_dir=output_dir,
        manifest=manifest,
        metrics=metrics,
        results=results,
        diagnostics=diagnostics,
    )

    return metrics

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


def main() -> None:
    print("[R4] Loading frozen E0 TRAIN cases...")

    cases = load_frozen_train_cases_from_e0(
        DEFAULT_E0_TRAIN_RESULTS_PATH,
        expected_count=EXPECTED_TECHQA_TRAIN_QUERY_COUNT,
    )

    dataset_manifest = json.loads(
        DEFAULT_TECHQA_MANIFEST_PATH.read_text(
            encoding="utf-8",
        )
    )
    retrieval_dataset = dataset_manifest[
        "retrieval_dataset"
    ]

    print("[R4] Loading frozen TechQA corpus...")

    from datasets import load_dataset

    corpus_rows = load_dataset(
        retrieval_dataset["repo"],
        "corpus",
        split="train",
        revision=retrieval_dataset["revision"],
    )

    documents = build_techqa_documents(corpus_rows)

    print(
        "[R4] Building deterministic chunks: "
        f"documents={len(documents)}"
    )

    chunks = build_techqa_chunks(documents)

    print(
        "[R4] Chunk universe built: "
        f"chunks={len(chunks)}"
    )

    preflight_report = run_preflight(
        train_results_path=DEFAULT_E0_TRAIN_RESULTS_PATH,
        r3_manifest_path=DEFAULT_R3_MANIFEST_PATH,
        dataset_manifest_path=DEFAULT_TECHQA_MANIFEST_PATH,
        train_query_count=len(cases),
        observed_chunk_count=len(chunks),
        splitter_blob_loader=_resolve_splitter_blob_sha,
        version_loader=version,
    )

    print(
        "[R4] Preflight passed: "
        "TRAIN-only, provider_calls=0"
    )
    print("[R4] Building chunk BM25 index and running audit...")

    metrics = execute_audit(
        cases=cases,
        chunks=chunks,
        preflight_report=preflight_report,
        output_dir=DEFAULT_OUTPUT_DIR,
        retriever_factory=TechQAChunkBM25Retriever,
    )

    print("[R4] Audit complete.")
    print(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()