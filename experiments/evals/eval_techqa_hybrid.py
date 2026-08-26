from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ranx import Qrels, Run

from experiments.evals.ir.ranx_adapter import evaluate_ir_metrics
import hashlib
from pathlib import Path
import json

R3_IR_METRICS = (
    "recall@20",
    "recall@100",
    "mrr@10",
)


@dataclass(frozen=True)
class TechQAHybridResult:
    question_id: str
    relevant_document_ids: tuple[str, ...]
    dense_document_ids: tuple[str, ...]
    bm25_document_ids: tuple[str, ...]
    hybrid_document_ids: tuple[str, ...]
    bm25_latency_ms: float
    fusion_latency_ms: float


@dataclass(frozen=True)
class TechQAHybridSummary:
    query_count: int
    dense_metrics: dict[str, float]
    bm25_metrics: dict[str, float]
    hybrid_metrics: dict[str, float]
    dense_hit100: int
    bm25_hit100: int
    hybrid_hit100: int
    dense_only_hits: int
    bm25_only_hits: int
    both_hits: int
    neither_hits: int
    recovered_dense_misses: int
    lost_dense_hits: int
    bm25_latency_p50_ms: float
    bm25_latency_p95_ms: float
    fusion_latency_p50_ms: float
    fusion_latency_p95_ms: float


def collapse_document_ids(document_ids: Sequence[str]) -> list[str]:
    collapsed: list[str] = []
    seen: set[str] = set()

    for document_id in document_ids:
        document_id = str(document_id)

        if document_id in seen:
            continue

        seen.add(document_id)
        collapsed.append(document_id)

    return collapsed


def _has_relevant_document(
    ranking: Sequence[str],
    relevant_document_ids: Sequence[str],
    *,
    top_k: int = 100,
) -> bool:
    relevant = {str(document_id) for document_id in relevant_document_ids}

    return any(
        str(document_id) in relevant
        for document_id in ranking[:top_k]
    )


def _build_qrels(results: Sequence[TechQAHybridResult]) -> Qrels:
    return Qrels(
        {
            result.question_id: {
                document_id: 1
                for document_id in result.relevant_document_ids
            }
            for result in results
        }
    )


def _build_run(
    rankings: dict[str, Sequence[str]],
) -> Run:
    run: dict[str, dict[str, float]] = {}

    for question_id, ranking in rankings.items():
        ranking_size = len(ranking)
        run[question_id] = {
            str(document_id): float(ranking_size - rank)
            for rank, document_id in enumerate(ranking)
        }

    return Run(run)


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(float(value) for value in values)

    if not ordered:
        raise ValueError("percentile requires at least one value")

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


def build_hybrid_summary(
    results: Sequence[TechQAHybridResult],
) -> TechQAHybridSummary:
    if not results:
        raise ValueError("hybrid summary requires at least one result")

    qrels = _build_qrels(results)

    dense_run = _build_run(
        {
            result.question_id: result.dense_document_ids
            for result in results
        }
    )
    bm25_run = _build_run(
        {
            result.question_id: result.bm25_document_ids
            for result in results
        }
    )
    hybrid_run = _build_run(
        {
            result.question_id: result.hybrid_document_ids
            for result in results
        }
    )

    dense_metrics = evaluate_ir_metrics(
        qrels,
        dense_run,
        R3_IR_METRICS,
    )
    bm25_metrics = evaluate_ir_metrics(
        qrels,
        bm25_run,
        R3_IR_METRICS,
    )
    hybrid_metrics = evaluate_ir_metrics(
        qrels,
        hybrid_run,
        R3_IR_METRICS,
    )

    dense_hit100 = 0
    bm25_hit100 = 0
    hybrid_hit100 = 0
    dense_only_hits = 0
    bm25_only_hits = 0
    both_hits = 0
    neither_hits = 0
    recovered_dense_misses = 0
    lost_dense_hits = 0

    for result in results:
        dense_hit = _has_relevant_document(
            result.dense_document_ids,
            result.relevant_document_ids,
        )
        bm25_hit = _has_relevant_document(
            result.bm25_document_ids,
            result.relevant_document_ids,
        )
        hybrid_hit = _has_relevant_document(
            result.hybrid_document_ids,
            result.relevant_document_ids,
        )

        dense_hit100 += int(dense_hit)
        bm25_hit100 += int(bm25_hit)
        hybrid_hit100 += int(hybrid_hit)

        dense_only_hits += int(dense_hit and not bm25_hit)
        bm25_only_hits += int(bm25_hit and not dense_hit)
        both_hits += int(dense_hit and bm25_hit)
        neither_hits += int(not dense_hit and not bm25_hit)

        recovered_dense_misses += int(
            not dense_hit and hybrid_hit
        )
        lost_dense_hits += int(
            dense_hit and not hybrid_hit
        )

    bm25_latencies = [
        result.bm25_latency_ms
        for result in results
    ]
    fusion_latencies = [
        result.fusion_latency_ms
        for result in results
    ]

    return TechQAHybridSummary(
        query_count=len(results),
        dense_metrics=dense_metrics,
        bm25_metrics=bm25_metrics,
        hybrid_metrics=hybrid_metrics,
        dense_hit100=dense_hit100,
        bm25_hit100=bm25_hit100,
        hybrid_hit100=hybrid_hit100,
        dense_only_hits=dense_only_hits,
        bm25_only_hits=bm25_only_hits,
        both_hits=both_hits,
        neither_hits=neither_hits,
        recovered_dense_misses=recovered_dense_misses,
        lost_dense_hits=lost_dense_hits,
        bm25_latency_p50_ms=_percentile(
            bm25_latencies,
            0.50,
        ),
        bm25_latency_p95_ms=_percentile(
            bm25_latencies,
            0.95,
        ),
        fusion_latency_p50_ms=_percentile(
            fusion_latencies,
            0.50,
        ),
        fusion_latency_p95_ms=_percentile(
            fusion_latencies,
            0.95,
        ),
    )

def build_r3_manifest(
    *,
    dense_results_sha256: str,
    gate_sha256: str,
    corpus_sha256: str,
    queries_sha256: str,
    qrels_sha256: str,
) -> dict[str, object]:
    return {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r3_hybrid_pilot",
        "split": "train",
        "query_count": 450,
        "dense_source": {
            "candidate_chunk_k": 100,
            "document_rule": (
                "first occurrence over frozen E0 raw_document_ids"
            ),
            "results_sha256": dense_results_sha256,
        },
        "bm25": {
            "library": "bm25s",
            "version": "0.3.10",
            "method": "lucene",
            "k1": 1.5,
            "b": 0.75,
            "backend": "numpy",
            "candidate_document_k": 100,
            "query_normalization": "rstrip",
            "tokenizer_regex": (
                r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
            ),
        },
        "rrf": {
            "rrf_k": 60,
            "top_k": 100,
            "weights": "equal",
        },
        "retrieval_dataset": {
            "corpus_sha256": corpus_sha256,
            "queries_sha256": queries_sha256,
            "qrels_sha256": qrels_sha256,
        },
        "gate_sha256": gate_sha256,
        "provider_calls": 0,
    }


def evaluate_r3_gate(
    *,
    dense_hit100: int,
    hybrid_hit100: int,
    recovered_dense_misses: int,
    gate: dict[str, int],
) -> dict[str, object]:
    net_gain_cases = hybrid_hit100 - dense_hit100

    admitted = (
        recovered_dense_misses
        >= gate["required_recovered_dense_misses"]
        and net_gain_cases
        >= gate["required_net_gain_cases"]
    )

    return {
        "recovered_dense_misses": recovered_dense_misses,
        "net_gain_cases": net_gain_cases,
        "admitted": admitted,
        "status": (
            "ADMIT_PAID_R4"
            if admitted
            else "SKIP_PAID_R4"
        ),
    }

def evaluate_hybrid_row(
    row,
    *,
    bm25_searcher,
    clock,
) -> TechQAHybridResult:
    from experiments.evals.ir.rrf import fuse_rrf

    dense_document_ids = tuple(
        collapse_document_ids(row["raw_document_ids"])
    )

    bm25_started = clock()
    bm25_document_ids = tuple(
        str(document_id)
        for document_id in bm25_searcher(
            str(row["question"]).rstrip(),
            top_k=100,
        )
    )
    bm25_latency_ms = (clock() - bm25_started) * 1000.0

    fusion_started = clock()
    hybrid_document_ids = tuple(
        fuse_rrf(
            [
                dense_document_ids,
                bm25_document_ids,
            ],
            rrf_k=60,
            top_k=100,
        )
    )
    fusion_latency_ms = (clock() - fusion_started) * 1000.0

    return TechQAHybridResult(
        question_id=str(row["question_id"]),
        relevant_document_ids=tuple(
            str(document_id)
            for document_id in row["relevant_document_ids"]
        ),
        dense_document_ids=dense_document_ids,
        bm25_document_ids=bm25_document_ids,
        hybrid_document_ids=hybrid_document_ids,
        bm25_latency_ms=bm25_latency_ms,
        fusion_latency_ms=fusion_latency_ms,
    )

def render_admission_decision(
    *,
    summary: TechQAHybridSummary,
    gate: dict[str, int],
) -> str:
    decision = evaluate_r3_gate(
        dense_hit100=summary.dense_hit100,
        hybrid_hit100=summary.hybrid_hit100,
        recovered_dense_misses=summary.recovered_dense_misses,
        gate=gate,
    )

    lines = [
        "# R3 Hybrid Admission Decision",
        "",
        "## Committed thresholds",
        (
            "required_recovered_dense_misses = "
            f"{gate['required_recovered_dense_misses']}"
        ),
        (
            "required_net_gain_cases = "
            f"{gate['required_net_gain_cases']}"
        ),
        "",
        "## Observed results",
        (
            "recovered_dense_misses = "
            f"{summary.recovered_dense_misses}"
        ),
        f"dense_hit100 = {summary.dense_hit100}",
        f"bm25_hit100 = {summary.bm25_hit100}",
        f"hybrid_hit100 = {summary.hybrid_hit100}",
        f"net_gain_cases = {decision['net_gain_cases']}",
        "",
        "### Dense metrics",
        f"recall@20 = {summary.dense_metrics['recall@20']}",
        f"recall@100 = {summary.dense_metrics['recall@100']}",
        f"mrr@10 = {summary.dense_metrics['mrr@10']}",
        "",
        "### BM25 metrics",
        f"recall@20 = {summary.bm25_metrics['recall@20']}",
        f"recall@100 = {summary.bm25_metrics['recall@100']}",
        f"mrr@10 = {summary.bm25_metrics['mrr@10']}",
        "",
        "### Hybrid metrics",
        f"recall@20 = {summary.hybrid_metrics['recall@20']}",
        f"recall@100 = {summary.hybrid_metrics['recall@100']}",
        f"mrr@10 = {summary.hybrid_metrics['mrr@10']}",
        "",
        "## Status",
        str(decision["status"]),
    ]

    return "\n".join(lines) + "\n"


def build_preflight_report(
    *,
    train_query_count: int,
    corpus_count: int,
    dense_results_sha256: str,
    gate_sha256: str,
    bm25_version: str,
) -> dict[str, object]:
    return {
        "split": "train",
        "train_query_count": train_query_count,
        "corpus_count": corpus_count,
        "dense_results_sha256": dense_results_sha256,
        "gate_sha256": gate_sha256,
        "bm25_version": bm25_version,
        "provider_calls": 0,
        "dev_artifact_opened": False,
    }

def run_preflight(
    *,
    dense_results_path,
    gate_path,
    document_loader,
    version_loader,
) -> dict[str, object]:
    dense_path = Path(dense_results_path)
    gate_file = Path(gate_path)

    train_query_count = sum(
        1
        for line in dense_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    )

    if train_query_count != 450:
        raise RuntimeError(
            "TRAIN query count mismatch: "
            f"expected=450, actual={train_query_count}"
        )

    corpus_count = len(document_loader())

    if corpus_count != 28481:
        raise RuntimeError(
            "TechQA corpus count mismatch: "
            f"expected=28481, actual={corpus_count}"
        )

    gate = json.loads(
        gate_file.read_text(encoding="utf-8")
    )
    gate_query_count = int(gate["query_count"])

    if gate_query_count != 450:
        raise RuntimeError(
            "R3 gate query count mismatch: "
            f"expected=450, actual={gate_query_count}"
        )

    bm25_version = version_loader("bm25s")

    if bm25_version != "0.3.10":
        raise RuntimeError(
            "bm25s version mismatch: "
            f"expected=0.3.10, actual={bm25_version}"
        )

    return build_preflight_report(
        train_query_count=train_query_count,
        corpus_count=corpus_count,
        dense_results_sha256=hashlib.sha256(
            dense_path.read_bytes()
        ).hexdigest(),
        gate_sha256=hashlib.sha256(
            gate_file.read_bytes()
        ).hexdigest(),
        bm25_version=bm25_version,
    )

def evaluate_hybrid_rows(
    rows,
    *,
    bm25_searcher,
    clock,
) -> tuple[
    tuple[TechQAHybridResult, ...],
    TechQAHybridSummary,
]:
    results = tuple(
        evaluate_hybrid_row(
            row,
            bm25_searcher=bm25_searcher,
            clock=clock,
        )
        for row in rows
    )

    return results, build_hybrid_summary(results)

def write_train_results_jsonl(
    results: Sequence[TechQAHybridResult],
    output_path: str | Path,
) -> str:
    path = Path(output_path)

    with path.open("w", encoding="utf-8") as file:
        for result in results:
            payload = {
                "question_id": result.question_id,
                "relevant_document_ids": list(
                    result.relevant_document_ids
                ),
                "dense_document_ids": list(
                    result.dense_document_ids
                ),
                "bm25_document_ids": list(
                    result.bm25_document_ids
                ),
                "hybrid_document_ids": list(
                    result.hybrid_document_ids
                ),
                "bm25_latency_ms": result.bm25_latency_ms,
                "fusion_latency_ms": result.fusion_latency_ms,
            }
            file.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def write_train_metrics_json(
    summary: TechQAHybridSummary,
    output_path: str | Path,
) -> None:
    payload = {
        "query_count": summary.query_count,
        "dense_metrics": summary.dense_metrics,
        "bm25_metrics": summary.bm25_metrics,
        "hybrid_metrics": summary.hybrid_metrics,
        "dense_hit100": summary.dense_hit100,
        "bm25_hit100": summary.bm25_hit100,
        "hybrid_hit100": summary.hybrid_hit100,
        "dense_only_hits": summary.dense_only_hits,
        "bm25_only_hits": summary.bm25_only_hits,
        "both_hits": summary.both_hits,
        "neither_hits": summary.neither_hits,
        "recovered_dense_misses": (
            summary.recovered_dense_misses
        ),
        "lost_dense_hits": summary.lost_dense_hits,
        "bm25_latency_p50_ms": (
            summary.bm25_latency_p50_ms
        ),
        "bm25_latency_p95_ms": (
            summary.bm25_latency_p95_ms
        ),
        "fusion_latency_p50_ms": (
            summary.fusion_latency_p50_ms
        ),
        "fusion_latency_p95_ms": (
            summary.fusion_latency_p95_ms
        ),
    }

    Path(output_path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

def materialize_r3_artifacts(
    *,
    results: Sequence[TechQAHybridResult],
    summary: TechQAHybridSummary,
    gate: dict[str, int],
    manifest: dict[str, object],
    output_dir: str | Path,
) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_sha256 = write_train_results_jsonl(
        results,
        output_root / "train_results.jsonl",
    )

    write_train_metrics_json(
        summary,
        output_root / "train_metrics.json",
    )

    persisted_manifest = {
        **manifest,
        "train_results_sha256": results_sha256,
    }
    (output_root / "train_manifest.json").write_text(
        json.dumps(
            persisted_manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    decision = render_admission_decision(
        summary=summary,
        gate=gate,
    )
    (output_root / "admission_decision.md").write_text(
        decision,
        encoding="utf-8",
    )

def load_e0_train_rows(
    input_path: str | Path,
    *,
    expected_count: int,
) -> list[dict[str, object]]:
    path = Path(input_path)

    payloads = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    if len(payloads) != expected_count:
        raise RuntimeError(
            "E0 TRAIN row count mismatch: "
            f"expected={expected_count}, actual={len(payloads)}"
        )

    return [
        {
            "question_id": str(payload["question_id"]),
            "question": str(payload["question"]),
            "relevant_document_ids": [
                str(document_id)
                for document_id in payload[
                    "relevant_document_ids"
                ]
            ],
            "raw_document_ids": [
                str(document_id)
                for document_id in payload[
                    "raw_document_ids"
                ]
            ],
        }
        for payload in payloads
    ]

def build_r3_manifest_from_frozen_inputs(
    *,
    dense_results_path: str | Path,
    gate_path: str | Path,
    dataset_manifest_path: str | Path,
) -> dict[str, object]:
    dense_path = Path(dense_results_path)
    gate_file = Path(gate_path)
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(
            encoding="utf-8",
        )
    )

    retrieval_dataset = dataset_manifest[
        "retrieval_dataset"
    ]

    return build_r3_manifest(
        dense_results_sha256=hashlib.sha256(
            dense_path.read_bytes()
        ).hexdigest(),
        gate_sha256=hashlib.sha256(
            gate_file.read_bytes()
        ).hexdigest(),
        corpus_sha256=str(
            retrieval_dataset["corpus_sha256"]
        ),
        queries_sha256=str(
            retrieval_dataset["queries_sha256"]
        ),
        qrels_sha256=str(
            retrieval_dataset["qrels_sha256"]
        ),
    )

def run_r3_hybrid_pilot(
    *,
    dense_results_path: str | Path,
    gate_path: str | Path,
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    expected_query_count: int,
    document_loader,
    bm25_factory,
    clock,
) -> TechQAHybridSummary:
    rows = load_e0_train_rows(
        dense_results_path,
        expected_count=expected_query_count,
    )

    documents = document_loader()
    bm25_retriever = bm25_factory(documents)

    results, summary = evaluate_hybrid_rows(
        rows,
        bm25_searcher=bm25_retriever.search,
        clock=clock,
    )

    gate = json.loads(
        Path(gate_path).read_text(
            encoding="utf-8",
        )
    )

    manifest = build_r3_manifest_from_frozen_inputs(
        dense_results_path=dense_results_path,
        gate_path=gate_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    materialize_r3_artifacts(
        results=results,
        summary=summary,
        gate=gate,
        manifest=manifest,
        output_dir=output_dir,
    )

    return summary

def verify_r3_artifacts(
    *,
    output_dir: str | Path,
    dense_results_path: str | Path,
    gate_path: str | Path,
    dataset_manifest_path: str | Path,
) -> str:
    output_root = Path(output_dir)

    persisted_manifest = json.loads(
        (
            output_root / "train_manifest.json"
        ).read_text(encoding="utf-8")
    )
    persisted_metrics = json.loads(
        (
            output_root / "train_metrics.json"
        ).read_text(encoding="utf-8")
    )
    persisted_decision = (
        output_root / "admission_decision.md"
    ).read_text(encoding="utf-8")

    results_path = output_root / "train_results.jsonl"

    actual_results_sha256 = hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()

    if (
        persisted_manifest["train_results_sha256"]
        != actual_results_sha256
    ):
        raise RuntimeError(
            "train_results SHA256 verification failed"
        )

    result_rows = [
        json.loads(line)
        for line in results_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    results = tuple(
        TechQAHybridResult(
            question_id=str(row["question_id"]),
            relevant_document_ids=tuple(
                str(document_id)
                for document_id in row[
                    "relevant_document_ids"
                ]
            ),
            dense_document_ids=tuple(
                str(document_id)
                for document_id in row[
                    "dense_document_ids"
                ]
            ),
            bm25_document_ids=tuple(
                str(document_id)
                for document_id in row[
                    "bm25_document_ids"
                ]
            ),
            hybrid_document_ids=tuple(
                str(document_id)
                for document_id in row[
                    "hybrid_document_ids"
                ]
            ),
            bm25_latency_ms=float(
                row["bm25_latency_ms"]
            ),
            fusion_latency_ms=float(
                row["fusion_latency_ms"]
            ),
        )
        for row in result_rows
    )

    summary = build_hybrid_summary(results)

    expected_metrics = {
        "query_count": summary.query_count,
        "dense_metrics": summary.dense_metrics,
        "bm25_metrics": summary.bm25_metrics,
        "hybrid_metrics": summary.hybrid_metrics,
        "dense_hit100": summary.dense_hit100,
        "bm25_hit100": summary.bm25_hit100,
        "hybrid_hit100": summary.hybrid_hit100,
        "dense_only_hits": summary.dense_only_hits,
        "bm25_only_hits": summary.bm25_only_hits,
        "both_hits": summary.both_hits,
        "neither_hits": summary.neither_hits,
        "recovered_dense_misses": (
            summary.recovered_dense_misses
        ),
        "lost_dense_hits": summary.lost_dense_hits,
        "bm25_latency_p50_ms": (
            summary.bm25_latency_p50_ms
        ),
        "bm25_latency_p95_ms": (
            summary.bm25_latency_p95_ms
        ),
        "fusion_latency_p50_ms": (
            summary.fusion_latency_p50_ms
        ),
        "fusion_latency_p95_ms": (
            summary.fusion_latency_p95_ms
        ),
    }

    if persisted_metrics != expected_metrics:
        raise RuntimeError(
            "R3 metrics verification failed"
        )

    if persisted_manifest["query_count"] != summary.query_count:
        raise RuntimeError(
            "R3 manifest query count verification failed"
        )

    dense_sha256 = hashlib.sha256(
        Path(dense_results_path).read_bytes()
    ).hexdigest()
    if (
        persisted_manifest["dense_source"][
            "results_sha256"
        ]
        != dense_sha256
    ):
        raise RuntimeError(
            "Frozen E0 results SHA256 verification failed"
        )

    gate_file = Path(gate_path)
    gate_sha256 = hashlib.sha256(
        gate_file.read_bytes()
    ).hexdigest()
    if persisted_manifest["gate_sha256"] != gate_sha256:
        raise RuntimeError(
            "R3 gate SHA256 verification failed"
        )

    gate = json.loads(
        gate_file.read_text(encoding="utf-8")
    )

    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(
            encoding="utf-8",
        )
    )
    retrieval_dataset = dataset_manifest[
        "retrieval_dataset"
    ]

    expected_dataset_hashes = {
        "corpus_sha256": str(
            retrieval_dataset["corpus_sha256"]
        ),
        "queries_sha256": str(
            retrieval_dataset["queries_sha256"]
        ),
        "qrels_sha256": str(
            retrieval_dataset["qrels_sha256"]
        ),
    }

    if (
        persisted_manifest["retrieval_dataset"]
        != expected_dataset_hashes
    ):
        raise RuntimeError(
            "TechQA dataset hash verification failed"
        )

    expected_decision = render_admission_decision(
        summary=summary,
        gate=gate,
    )

    if persisted_decision != expected_decision:
        raise RuntimeError(
            "R3 admission decision verification failed"
        )

    if persisted_manifest["provider_calls"] != 0:
        raise RuntimeError(
            "R3 provider_calls verification failed"
        )

    return "R3 HYBRID ARTIFACT VERIFICATION = OK"

def main(
    argv: Sequence[str] | None = None,
    *,
    preflight_runner=run_preflight,
    r3_runner=run_r3_hybrid_pilot,
    verify_runner=verify_r3_artifacts,
    document_loader=None,
    bm25_factory=None,
    clock=None,
    version_loader=None,
) -> None:
    import argparse
    from importlib.metadata import version

    parser = argparse.ArgumentParser(
        description="Evaluate frozen TechQA hybrid retrieval."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
    )
    parser.add_argument(
        "--verify",
        action="store_true"
    )
    args = parser.parse_args(argv)

    if args.verify:
        status = verify_runner(
            output_dir=Path(
                "experiments/evals/reports/r3_hybrid"
            ),
            dense_results_path=Path(
                "experiments/evals/reports/e0_dense/"
                "train_results.jsonl"
            ),
            gate_path=Path(
                "experiments/evals/reports/e1_rerank/"
                "r3_gate.json"
            ),
            dataset_manifest_path=Path(
                "experiments/evals/datasets/techqa/"
                "manifest.json"
            ),
        )
        print(status)
        return

    if not args.preflight:
        if document_loader is None:
            from experiments.evals.build_techqa_index import (
                load_frozen_techqa_documents,
            )

            document_loader = load_frozen_techqa_documents

        if bm25_factory is None:
            from experiments.evals.retrievers.bm25_techqa import (
                TechQABM25Retriever,
            )

            bm25_factory = TechQABM25Retriever

        if clock is None:
            import time

            clock = time.perf_counter

        r3_runner(
            dense_results_path=Path(
                "experiments/evals/reports/e0_dense/"
                "train_results.jsonl"
            ),
            gate_path=Path(
                "experiments/evals/reports/e1_rerank/"
                "r3_gate.json"
            ),
            dataset_manifest_path=Path(
                "experiments/evals/datasets/techqa/"
                "manifest.json"
            ),
            output_dir=Path(
                "experiments/evals/reports/r3_hybrid"
            ),
            expected_query_count=450,
            document_loader=document_loader,
            bm25_factory=bm25_factory,
            clock=clock,
        )
        return

    if document_loader is None:
        from experiments.evals.build_techqa_index import (
            load_frozen_techqa_documents,
        )

        document_loader = load_frozen_techqa_documents

    if version_loader is None:
        version_loader = version

    report = preflight_runner(
        dense_results_path=Path(
            "experiments/evals/reports/e0_dense/"
            "train_results.jsonl"
        ),
        gate_path=Path(
            "experiments/evals/reports/e1_rerank/"
            "r3_gate.json"
        ),
        document_loader=document_loader,
        version_loader=version_loader,
    )

    print(
        f"TRAIN count = "
        f"{report['train_query_count']}"
    )
    print(
        f"TechQA corpus count = "
        f"{report['corpus_count']}"
    )
    print(
        "frozen E0 input sha256 = "
        f"{report['dense_results_sha256']}"
    )
    print(
        "frozen R3 gate sha256 = "
        f"{report['gate_sha256']}"
    )
    print(
        f"bm25s = "
        f"{report['bm25_version']}"
    )
    print(
        f"provider_calls = "
        f"{report['provider_calls']}"
    )
    print(
        f"dev_artifact_opened = "
        f"{report['dev_artifact_opened']}"
    )

if __name__ == "__main__":
    main()