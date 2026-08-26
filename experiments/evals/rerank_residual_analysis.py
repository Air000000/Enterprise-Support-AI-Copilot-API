from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from experiments.evals.adapters.techqa import TechQADocument

ResidualBucket = Literal[
    "resolved_top20",
    "rerank_residual_top20",
    "dense_candidate_miss_top100",
]
ReviewLabel = Literal[
    "lexical_candidate",
    "semantic_or_indirect_miss",
    "qrel_or_query_ambiguity",
]
REVIEW_LABELS: tuple[ReviewLabel, ...] = (
    "lexical_candidate",
    "semantic_or_indirect_miss",
    "qrel_or_query_ambiguity",
)
DocumentLoader = Callable[[], Sequence[TechQADocument]]

DEFAULT_E0_TRAIN_RESULTS_PATH = Path(
    "experiments/evals/reports/e0_dense/train_results.jsonl"
)
DEFAULT_E1_TRAIN_RESULTS_PATH = Path(
    "experiments/evals/reports/e1_rerank/train_results.jsonl"
)
DEFAULT_RERANK_REPORT_DIR = Path("experiments/evals/reports/e1_rerank")
DEFAULT_EXPECTED_TRAIN_COUNT = 450
DEFAULT_REVIEW_SAMPLE_SIZE = 30


@dataclass(frozen=True)
class RerankResidualRecord:
    question_id: str
    question: str
    relevant_document_ids: tuple[str, ...]
    dense_candidate_document_ids: tuple[str, ...]
    e1_document_ranking: tuple[str, ...]
    residual_bucket: ResidualBucket


def _index_unique(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in indexed:
            raise RuntimeError(f"duplicate {label} question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def _collapse_document_ids(document_ids: Sequence[Any]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in document_ids:
        document_id = str(value)
        if document_id in seen:
            continue
        seen.add(document_id)
        ordered.append(document_id)
    return tuple(ordered)


def _contains_relevant(
    ranking: Sequence[Any],
    relevant_document_ids: Sequence[str],
) -> bool:
    relevant = set(relevant_document_ids)
    return any(str(document_id) in relevant for document_id in ranking)


def build_residual_records(
    e0_rows: Sequence[Mapping[str, Any]],
    e1_rows: Sequence[Mapping[str, Any]],
) -> list[RerankResidualRecord]:
    e0_by_id = _index_unique(e0_rows, label="E0")
    e1_by_id = _index_unique(e1_rows, label="E1")

    if set(e0_by_id) != set(e1_by_id):
        raise RuntimeError("E0/E1 question_id mismatch")

    records: list[RerankResidualRecord] = []
    for question_id in sorted(e0_by_id):
        e0 = e0_by_id[question_id]
        e1 = e1_by_id[question_id]

        dense_chunk_ids = tuple(str(value) for value in e0["raw_chunk_ids"])
        rerank_dense_chunk_ids = tuple(str(value) for value in e1["dense_chunk_ids"])
        if rerank_dense_chunk_ids != dense_chunk_ids:
            raise RuntimeError(
                f"dense candidate chunk identity mismatch for {question_id}"
            )

        relevant_document_ids = tuple(
            str(value) for value in e0["relevant_document_ids"]
        )
        dense_candidate_document_ids = _collapse_document_ids(
            e0["raw_document_ids"]
        )
        e1_document_ranking = tuple(
            str(value) for value in e1["document_ranking"]
        )

        if _contains_relevant(e1_document_ranking[:20], relevant_document_ids):
            residual_bucket: ResidualBucket = "resolved_top20"
        elif _contains_relevant(
            dense_candidate_document_ids,
            relevant_document_ids,
        ):
            residual_bucket = "rerank_residual_top20"
        else:
            residual_bucket = "dense_candidate_miss_top100"

        records.append(
            RerankResidualRecord(
                question_id=question_id,
                question=str(e0["question"]),
                relevant_document_ids=relevant_document_ids,
                dense_candidate_document_ids=dense_candidate_document_ids,
                e1_document_ranking=e1_document_ranking,
                residual_bucket=residual_bucket,
            )
        )

    return records


def select_candidate_miss_review_sample(
    records: Sequence[RerankResidualRecord],
    *,
    sample_size: int = 30,
) -> list[RerankResidualRecord]:
    if sample_size <= 0:
        return []

    misses = [
        record
        for record in records
        if record.residual_bucket == "dense_candidate_miss_top100"
    ]
    return sorted(
        misses,
        key=lambda record: hashlib.sha256(
            record.question_id.encode("utf-8")
        ).hexdigest(),
    )[: min(sample_size, len(misses))]


def summarize_residual_records(
    records: Sequence[RerankResidualRecord],
) -> dict[str, Any]:
    ordered_records = list(records)
    query_count = len(ordered_records)
    bucket_counts = {
        bucket: sum(record.residual_bucket == bucket for record in ordered_records)
        for bucket in (
            "dense_candidate_miss_top100",
            "rerank_residual_top20",
            "resolved_top20",
        )
    }
    bucket_rates = {
        bucket: count / query_count if query_count else 0.0
        for bucket, count in bucket_counts.items()
    }
    return {
        "query_count": query_count,
        "bucket_counts": bucket_counts,
        "bucket_rates": bucket_rates,
        "dense_candidate_miss_count": bucket_counts[
            "dense_candidate_miss_top100"
        ],
    }


def summarize_review_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    label_counts = {label: 0 for label in REVIEW_LABELS}
    for row in rows:
        question_id = str(row.get("question_id", "<unknown>"))
        label = str(row.get("manual_label", "")).strip()
        if not label:
            raise RuntimeError(f"review label missing for {question_id}")
        if label not in REVIEW_LABELS:
            raise RuntimeError(f"unknown review label for {question_id}: {label}")
        label_counts[label] += 1

    reviewed_count = len(rows)
    label_rates = {
        label: count / reviewed_count if reviewed_count else 0.0
        for label, count in label_counts.items()
    }
    return {
        "reviewed_count": reviewed_count,
        "label_counts": label_counts,
        "label_rates": label_rates,
        "population_rate_claim_allowed": False,
    }


def build_r3_gate(dense_candidate_miss_count: int) -> dict[str, Any]:
    required_recovered_dense_misses = max(
        5,
        math.ceil(0.15 * dense_candidate_miss_count),
    )
    required_net_gain_cases = max(
        5,
        math.ceil(0.10 * dense_candidate_miss_count),
    )
    return {
        "split": "train",
        "query_count": 450,
        "dense_candidate_chunk_k": 100,
        "bm25_candidate_document_k": 100,
        "hybrid_candidate_document_k": 100,
        "rrf_k": 60,
        "dense_candidate_miss_count": dense_candidate_miss_count,
        "required_recovered_dense_misses": required_recovered_dense_misses,
        "required_net_gain_cases": required_net_gain_cases,
        "required_net_gain_pp": required_net_gain_cases / 450 * 100.0,
        "admission_logic": (
            "recovered_dense_misses >= required_recovered_dense_misses AND "
            "hybrid_hit100 - dense_hit100 >= required_net_gain_cases"
        ),
    }


def _document_excerpt(
    document_id: str,
    *,
    documents_by_id: Mapping[str, str],
    excerpt_chars: int,
) -> dict[str, str]:
    return {
        "document_id": document_id,
        "text_excerpt": documents_by_id[document_id][:excerpt_chars],
    }


def build_candidate_miss_review_rows(
    records: Sequence[RerankResidualRecord],
    *,
    documents_by_id: Mapping[str, str],
    excerpt_chars: int = 600,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "question_id": record.question_id,
                "question": record.question,
                "gold_documents": [
                    _document_excerpt(
                        document_id,
                        documents_by_id=documents_by_id,
                        excerpt_chars=excerpt_chars,
                    )
                    for document_id in record.relevant_document_ids
                ],
                "dense_top5": [
                    _document_excerpt(
                        document_id,
                        documents_by_id=documents_by_id,
                        excerpt_chars=excerpt_chars,
                    )
                    for document_id in record.dense_candidate_document_ids[:5]
                ],
                "e1_top5": [
                    _document_excerpt(
                        document_id,
                        documents_by_id=documents_by_id,
                        excerpt_chars=excerpt_chars,
                    )
                    for document_id in record.e1_document_ranking[:5]
                ],
                "manual_label": "",
                "notes": "",
            }
        )
    return rows


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_default_documents() -> Sequence[TechQADocument]:
    from experiments.evals.build_techqa_index import load_frozen_techqa_documents

    return load_frozen_techqa_documents()


def materialize_rerank_residual_analysis(
    *,
    e0_results_path: str | Path,
    e1_results_path: str | Path,
    report_dir: str | Path,
    expected_count: int = DEFAULT_EXPECTED_TRAIN_COUNT,
    review_sample_size: int = DEFAULT_REVIEW_SAMPLE_SIZE,
    document_loader: DocumentLoader | None = None,
) -> dict[str, Any]:
    e0_rows = _load_jsonl(e0_results_path)
    e1_rows = _load_jsonl(e1_results_path)
    if len(e0_rows) != expected_count or len(e1_rows) != expected_count:
        raise RuntimeError(
            "TRAIN row count mismatch: "
            f"expected={expected_count}, e0={len(e0_rows)}, e1={len(e1_rows)}"
        )

    records = build_residual_records(e0_rows, e1_rows)
    summary = summarize_residual_records(records)
    sample = select_candidate_miss_review_sample(
        records,
        sample_size=review_sample_size,
    )

    loader = document_loader or _load_default_documents
    documents = loader()
    documents_by_id = {
        document.document_id: document.text
        for document in documents
    }
    review_rows = build_candidate_miss_review_rows(
        sample,
        documents_by_id=documents_by_id,
    )

    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "train_residual_summary.json", summary)
    with (output_dir / "train_residual_review.jsonl").open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in review_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    return summary


def main(
    argv: Sequence[str] | None = None,
    *,
    document_loader: DocumentLoader | None = None,
) -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen E1 TRAIN residuals.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--e0-results", type=Path, default=DEFAULT_E0_TRAIN_RESULTS_PATH)
    prepare.add_argument("--e1-results", type=Path, default=DEFAULT_E1_TRAIN_RESULTS_PATH)
    prepare.add_argument("--report-dir", type=Path, default=DEFAULT_RERANK_REPORT_DIR)
    prepare.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_TRAIN_COUNT,
    )
    prepare.add_argument(
        "--review-sample-size",
        type=int,
        default=DEFAULT_REVIEW_SAMPLE_SIZE,
    )

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--report-dir", type=Path, default=DEFAULT_RERANK_REPORT_DIR)

    freeze_gate = subparsers.add_parser("freeze-gate")
    freeze_gate.add_argument("--report-dir", type=Path, default=DEFAULT_RERANK_REPORT_DIR)

    args = parser.parse_args(argv)

    if args.command == "prepare":
        summary = materialize_rerank_residual_analysis(
            e0_results_path=args.e0_results,
            e1_results_path=args.e1_results,
            report_dir=args.report_dir,
            expected_count=args.expected_count,
            review_sample_size=args.review_sample_size,
            document_loader=document_loader,
        )
        print("R2 TRAIN residual evidence materialized.")
        print(f"query_count = {summary['query_count']}")
        for bucket, count in summary["bucket_counts"].items():
            print(f"{bucket} = {count}")
        print("provider_calls = 0")
        return

    if args.command == "summarize":
        review_rows = _load_jsonl(args.report_dir / "train_residual_review.jsonl")
        summary = summarize_review_rows(review_rows)
        _write_json(
            args.report_dir / "train_residual_review_summary.json",
            summary,
        )
        print("R2 TRAIN residual review summarized.")
        print(f"reviewed_count = {summary['reviewed_count']}")
        for label, count in summary["label_counts"].items():
            print(f"{label} = {count}")
        print("provider_calls = 0")
        return

    if args.command == "freeze-gate":
        residual_summary = _load_json(args.report_dir / "train_residual_summary.json")
        dense_candidate_miss_count = int(
            residual_summary["dense_candidate_miss_count"]
        )
        gate = build_r3_gate(dense_candidate_miss_count)
        _write_json(args.report_dir / "r3_gate.json", gate)
        print("R3 admission gate frozen.")
        print(f"dense_candidate_miss_count = {gate['dense_candidate_miss_count']}")
        print(
            "required_recovered_dense_misses = "
            f"{gate['required_recovered_dense_misses']}"
        )
        print(f"required_net_gain_cases = {gate['required_net_gain_cases']}")
        print(f"required_net_gain_pp = {gate['required_net_gain_pp']}")
        print("provider_calls = 0")
        return

    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
