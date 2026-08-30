from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PostmortemRecord:
    question_id: str
    dense_gold_hit: bool
    bm25_gold_hit: bool
    fused_gold_hit: bool
    hybrid_rescued_dense_miss: bool
    hybrid_lost_dense_hit: bool
    final_gold_rank: int | None
    final_hit5: bool
    final_hit20: bool
    residual_bucket: str
    fused_unique_document_count: int
    fused_duplicate_ratio: float
    max_chunks_per_document: int
    dense_only_chunk_count: int
    bm25_only_chunk_count: int
    shared_chunk_count: int


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}

    for row in rows:
        question_id = str(row["question_id"])

        if question_id in indexed:
            raise RuntimeError(
                f"duplicate {label} question_id: {question_id}"
            )

        indexed[question_id] = row

    return indexed


def _gold_hit_from_chunks(
    chunk_ids: Sequence[str],
    *,
    gold_document_ids: set[str],
    chunk_document_ids: Mapping[str, str],
) -> bool:
    return any(
        str(chunk_document_ids[str(chunk_id)])
        in gold_document_ids
        for chunk_id in chunk_ids
    )


def _first_gold_rank(
    document_ranking: Sequence[str],
    *,
    gold_document_ids: set[str],
) -> int | None:
    for rank, document_id in enumerate(
        document_ranking,
        start=1,
    ):
        if str(document_id) in gold_document_ids:
            return rank

    return None


def build_postmortem_records(
    snapshot_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    chunk_document_ids: Mapping[str, str],
) -> tuple[PostmortemRecord, ...]:
    results_by_id = _index_rows(
        result_rows,
        label="result",
    )

    records: list[PostmortemRecord] = []

    for snapshot in snapshot_rows:
        question_id = str(snapshot["question_id"])

        if question_id not in results_by_id:
            raise RuntimeError(
                f"missing result for question_id: {question_id}"
            )

        result = results_by_id[question_id]

        gold_document_ids = {
            str(document_id)
            for document_id
            in snapshot["relevant_document_ids"]
        }

        result_gold_document_ids = {
            str(document_id)
            for document_id
            in result["relevant_document_ids"]
        }

        if result_gold_document_ids != gold_document_ids:
            raise RuntimeError(
                "snapshot/result gold mismatch for "
                f"question_id: {question_id}"
            )

        dense_chunk_ids = tuple(
            str(chunk_id)
            for chunk_id in snapshot["dense_chunk_ids"]
        )
        bm25_chunk_ids = tuple(
            str(chunk_id)
            for chunk_id in snapshot["bm25_chunk_ids"]
        )

        fused_candidates = tuple(
            snapshot["fused_candidates"]
        )
        fused_chunk_ids = tuple(
            str(candidate["chunk_id"])
            for candidate in fused_candidates
        )
        fused_document_ids = tuple(
            str(candidate["document_id"])
            for candidate in fused_candidates
        )

        dense_gold_hit = _gold_hit_from_chunks(
            dense_chunk_ids,
            gold_document_ids=gold_document_ids,
            chunk_document_ids=chunk_document_ids,
        )
        bm25_gold_hit = _gold_hit_from_chunks(
            bm25_chunk_ids,
            gold_document_ids=gold_document_ids,
            chunk_document_ids=chunk_document_ids,
        )
        fused_gold_hit = any(
            document_id in gold_document_ids
            for document_id in fused_document_ids
        )

        final_gold_rank = _first_gold_rank(
            result["document_ranking"],
            gold_document_ids=gold_document_ids,
        )

        final_hit5 = (
            final_gold_rank is not None
            and final_gold_rank <= 5
        )
        final_hit20 = (
            final_gold_rank is not None
            and final_gold_rank <= 20
        )

        if not fused_gold_hit:
            residual_bucket = "candidate_miss_top100"
        elif not final_hit20:
            residual_bucket = "ranking_miss_top20"
        else:
            residual_bucket = "resolved_top20"

        document_counts = Counter(
            fused_document_ids
        )
        fused_count = len(fused_document_ids)
        unique_document_count = len(document_counts)

        duplicate_ratio = (
            0.0
            if fused_count == 0
            else 1.0
            - (unique_document_count / fused_count)
        )

        max_chunks_per_document = (
            max(document_counts.values())
            if document_counts
            else 0
        )

        dense_set = set(dense_chunk_ids)
        bm25_set = set(bm25_chunk_ids)

        dense_only_chunk_count = sum(
            chunk_id in dense_set
            and chunk_id not in bm25_set
            for chunk_id in fused_chunk_ids
        )
        bm25_only_chunk_count = sum(
            chunk_id in bm25_set
            and chunk_id not in dense_set
            for chunk_id in fused_chunk_ids
        )
        shared_chunk_count = sum(
            chunk_id in dense_set
            and chunk_id in bm25_set
            for chunk_id in fused_chunk_ids
        )

        records.append(
            PostmortemRecord(
                question_id=question_id,
                dense_gold_hit=dense_gold_hit,
                bm25_gold_hit=bm25_gold_hit,
                fused_gold_hit=fused_gold_hit,
                hybrid_rescued_dense_miss=(
                    not dense_gold_hit
                    and fused_gold_hit
                ),
                hybrid_lost_dense_hit=(
                    dense_gold_hit
                    and not fused_gold_hit
                ),
                final_gold_rank=final_gold_rank,
                final_hit5=final_hit5,
                final_hit20=final_hit20,
                residual_bucket=residual_bucket,
                fused_unique_document_count=(
                    unique_document_count
                ),
                fused_duplicate_ratio=duplicate_ratio,
                max_chunks_per_document=(
                    max_chunks_per_document
                ),
                dense_only_chunk_count=(
                    dense_only_chunk_count
                ),
                bm25_only_chunk_count=(
                    bm25_only_chunk_count
                ),
                shared_chunk_count=shared_chunk_count,
            )
        )

    if set(results_by_id) != {
        record.question_id
        for record in records
    }:
        raise RuntimeError(
            "snapshot/result question_id sets differ"
        )

    return tuple(records)


def summarize_postmortem(
    records: Sequence[PostmortemRecord],
) -> dict[str, int]:
    rescued = sum(
        record.hybrid_rescued_dense_miss
        for record in records
    )
    lost = sum(
        record.hybrid_lost_dense_hit
        for record in records
    )

    return {
        "query_count": len(records),
        "dense_gold_hit_count": sum(
            record.dense_gold_hit
            for record in records
        ),
        "bm25_gold_hit_count": sum(
            record.bm25_gold_hit
            for record in records
        ),
        "fused_gold_hit_count": sum(
            record.fused_gold_hit
            for record in records
        ),
        "hybrid_rescued_dense_misses": rescued,
        "hybrid_lost_dense_hits": lost,
        "net_candidate_gain": rescued - lost,
        "resolved_top20_count": sum(
            record.residual_bucket == "resolved_top20"
            for record in records
        ),
        "ranking_miss_top20_count": sum(
            record.residual_bucket
            == "ranking_miss_top20"
            for record in records
        ),
        "candidate_miss_top100_count": sum(
            record.residual_bucket
            == "candidate_miss_top100"
            for record in records
        ),
    }

def document_id_from_chunk_id(chunk_id: str) -> str:
    marker = "_chunk_"

    if marker not in chunk_id:
        raise RuntimeError(
            f"invalid TechQA chunk_id: {chunk_id}"
        )

    document_id, chunk_index = chunk_id.rsplit(
        marker,
        maxsplit=1,
    )

    if not document_id or not chunk_index.isdigit():
        raise RuntimeError(
            f"invalid TechQA chunk_id: {chunk_id}"
        )

    return document_id


def _read_jsonl(path):
    import json
    from pathlib import Path

    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc

    return rows


def _percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def _build_chunk_document_mapping(
    snapshot_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for row in snapshot_rows:
        for key in (
            "dense_chunk_ids",
            "bm25_chunk_ids",
        ):
            for chunk_id_value in row[key]:
                chunk_id = str(chunk_id_value)
                mapping[chunk_id] = (
                    document_id_from_chunk_id(
                        chunk_id
                    )
                )

        for candidate in row["fused_candidates"]:
            chunk_id = str(candidate["chunk_id"])
            document_id = str(
                candidate["document_id"]
            )

            existing = mapping.get(chunk_id)

            if (
                existing is not None
                and existing != document_id
            ):
                raise RuntimeError(
                    "chunk/document mapping mismatch: "
                    f"{chunk_id}"
                )

            mapping[chunk_id] = document_id

    return mapping


def _write_postmortem_report(
    path,
    summary: Mapping[str, Any],
) -> None:
    from pathlib import Path

    complementarity = summary[
        "candidate_complementarity"
    ]
    residual = summary[
        "residual_attribution"
    ]
    crowding = summary["crowding"]
    composition = summary["source_composition"]

    report = f"""# R4 C1 Zero-Cost Postmortem

## Scope

- TRAIN queries: {summary["query_count"]}
- Provider calls: {summary["provider_calls"]}
- DEV artifact opened: {summary["dev_artifact_opened"]}

## Candidate complementarity

- Dense-only gold hits: {complementarity["dense_only_gold_hit_count"]}
- BM25-only gold hits: {complementarity["bm25_only_gold_hit_count"]}
- Both-source gold hits: {complementarity["both_gold_hit_count"]}
- Neither-source gold hits: {complementarity["neither_gold_hit_count"]}
- Hybrid rescued Dense misses: {complementarity["hybrid_rescued_dense_misses"]}
- Hybrid lost Dense hits: {complementarity["hybrid_lost_dense_hits"]}
- Net candidate gain: {complementarity["net_candidate_gain"]}

## Candidate miss vs ranking miss

- Resolved in final Top-20: {residual["resolved_top20_count"]}
- Gold in fused candidates but final rank > 20: {residual["ranking_miss_top20_count"]}
- Gold absent from fused Top-100: {residual["candidate_miss_top100_count"]}

## Chunk crowding

- Fused unique documents p50: {crowding["fused_unique_document_count_p50"]:.3f}
- Fused unique documents p95: {crowding["fused_unique_document_count_p95"]:.3f}
- Duplicate ratio p50: {crowding["fused_duplicate_ratio_p50"]:.6f}
- Duplicate ratio p95: {crowding["fused_duplicate_ratio_p95"]:.6f}
- Duplicate ratio max: {crowding["fused_duplicate_ratio_max"]:.6f}
- Max chunks per document p95: {crowding["max_chunks_per_document_p95"]:.3f}

## Source composition

- Dense-only fused chunks: {composition["dense_only_chunk_count_total"]}
- BM25-only fused chunks: {composition["bm25_only_chunk_count_total"]}
- Shared fused chunks: {composition["shared_chunk_count_total"]}

## Interpretation boundary

This report is a zero-provider diagnostic over the already frozen
R4 C1 TRAIN candidate snapshot and completed C1 rerank results.

It can distinguish candidate coverage failures from post-candidate
ranking failures and quantify chunk/document crowding.

It does not by itself prove that BM25, RRF, crowding, or the reranker
is the causal root cause of any observed failure. No E1 rerun,
parameter search, DEV inspection, or provider call is performed.
"""

    Path(path).write_text(
        report,
        encoding="utf-8",
    )


def run_postmortem(
    *,
    snapshot_path,
    results_path,
    output_dir,
    chunk_document_ids=None,
    expected_count: int = 450,
) -> dict[str, Any]:
    import json
    from dataclasses import asdict
    from pathlib import Path

    snapshot_rows = _read_jsonl(
        snapshot_path
    )
    result_rows = _read_jsonl(
        results_path
    )

    if len(snapshot_rows) != expected_count:
        raise RuntimeError(
            "snapshot query count mismatch: "
            f"expected={expected_count}, "
            f"actual={len(snapshot_rows)}"
        )

    if len(result_rows) != expected_count:
        raise RuntimeError(
            "result query count mismatch: "
            f"expected={expected_count}, "
            f"actual={len(result_rows)}"
        )

    question_ids = [
        str(row["question_id"])
        for row in snapshot_rows
    ]

    if any(
        not question_id.startswith("TRAIN_")
        for question_id in question_ids
    ):
        raise RuntimeError(
            "postmortem requires TRAIN-only input"
        )

    if len(set(question_ids)) != expected_count:
        raise RuntimeError(
            "postmortem requires unique TRAIN question IDs"
        )

    if chunk_document_ids is None:
        resolved_chunk_document_ids = (
            _build_chunk_document_mapping(
                snapshot_rows
            )
        )
    else:
        resolved_chunk_document_ids = {
            str(chunk_id): str(document_id)
            for chunk_id, document_id
            in chunk_document_ids.items()
        }

    records = build_postmortem_records(
        snapshot_rows,
        result_rows,
        chunk_document_ids=(
            resolved_chunk_document_ids
        ),
    )

    base = summarize_postmortem(records)

    dense_only_gold_hit_count = sum(
        record.dense_gold_hit
        and not record.bm25_gold_hit
        for record in records
    )
    bm25_only_gold_hit_count = sum(
        record.bm25_gold_hit
        and not record.dense_gold_hit
        for record in records
    )
    both_gold_hit_count = sum(
        record.dense_gold_hit
        and record.bm25_gold_hit
        for record in records
    )
    neither_gold_hit_count = sum(
        not record.dense_gold_hit
        and not record.bm25_gold_hit
        for record in records
    )

    duplicate_ratios = [
        record.fused_duplicate_ratio
        for record in records
    ]
    unique_document_counts = [
        float(
            record.fused_unique_document_count
        )
        for record in records
    ]
    max_chunks_per_document = [
        float(
            record.max_chunks_per_document
        )
        for record in records
    ]

    summary: dict[str, Any] = {
        "query_count": len(records),
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "candidate_complementarity": {
            "dense_only_gold_hit_count": (
                dense_only_gold_hit_count
            ),
            "bm25_only_gold_hit_count": (
                bm25_only_gold_hit_count
            ),
            "both_gold_hit_count": (
                both_gold_hit_count
            ),
            "neither_gold_hit_count": (
                neither_gold_hit_count
            ),
            "hybrid_rescued_dense_misses": (
                base[
                    "hybrid_rescued_dense_misses"
                ]
            ),
            "hybrid_lost_dense_hits": (
                base[
                    "hybrid_lost_dense_hits"
                ]
            ),
            "net_candidate_gain": (
                base["net_candidate_gain"]
            ),
        },
        "residual_attribution": {
            "resolved_top20_count": (
                base["resolved_top20_count"]
            ),
            "ranking_miss_top20_count": (
                base[
                    "ranking_miss_top20_count"
                ]
            ),
            "candidate_miss_top100_count": (
                base[
                    "candidate_miss_top100_count"
                ]
            ),
        },
        "crowding": {
            "fused_unique_document_count_p50": (
                _percentile(
                    unique_document_counts,
                    0.50,
                )
            ),
            "fused_unique_document_count_p95": (
                _percentile(
                    unique_document_counts,
                    0.95,
                )
            ),
            "fused_duplicate_ratio_p50": (
                _percentile(
                    duplicate_ratios,
                    0.50,
                )
            ),
            "fused_duplicate_ratio_p95": (
                _percentile(
                    duplicate_ratios,
                    0.95,
                )
            ),
            "fused_duplicate_ratio_max": (
                max(
                    duplicate_ratios,
                    default=0.0,
                )
            ),
            "max_chunks_per_document_p95": (
                _percentile(
                    max_chunks_per_document,
                    0.95,
                )
            ),
        },
        "source_composition": {
            "dense_only_chunk_count_total": sum(
                record.dense_only_chunk_count
                for record in records
            ),
            "bm25_only_chunk_count_total": sum(
                record.bm25_only_chunk_count
                for record in records
            ),
            "shared_chunk_count_total": sum(
                record.shared_chunk_count
                for record in records
            ),
        },
    }

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_path
        / "postmortem_summary.json"
    )
    cases_path = (
        output_path
        / "postmortem_cases.jsonl"
    )
    report_path = (
        output_path
        / "postmortem.md"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with cases_path.open(
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

    _write_postmortem_report(
        report_path,
        summary,
    )

    return summary
