from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.evals.rerank_evidence import (
    build_paired_evidence,
    materialize_rerank_evidence,
    sha256_file,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _e0_row(question_id: str, gold: str, ranking: list[str], latency_ms: float) -> dict:
    return {
        "question_id": question_id,
        "question": f"question for {question_id}",
        "relevant_document_ids": [gold],
        "raw_chunk_ids": [f"{question_id}-chunk-1"],
        "raw_document_ids": [ranking[0]],
        "document_ranking": ranking,
        "latency_ms": latency_ms,
    }


def _e1_row(
    question_id: str,
    gold: str,
    ranking: list[str],
    rerank_latency_ms: float,
    *,
    total_tokens: int = 10,
) -> dict:
    return {
        "question_id": question_id,
        "relevant_document_ids": [gold],
        "dense_chunk_ids": [f"{question_id}-chunk-1"],
        "reranked_chunk_ids": [f"{question_id}-chunk-1"],
        "reranked_document_ids": [ranking[0]],
        "document_ranking": ranking,
        "rerank_latency_ms": rerank_latency_ms,
        "request_id": f"request-{question_id}",
        "total_tokens": total_tokens,
    }


def _write_split_artifacts(
    *,
    e0_dir: Path,
    e1_dir: Path,
    split: str,
    e0_rows: list[dict],
    e1_rows: list[dict],
) -> None:
    _write_jsonl(e0_dir / f"{split}_results.jsonl", e0_rows)
    _write_json(
        e0_dir / f"{split}_metrics.json",
        {
            "query_count": len(e0_rows),
            "metrics": {"recall@5": 0.5, "recall@20": 0.5, "mrr@10": 0.5},
            "latency_p50_ms": 100.0,
            "latency_p95_ms": 200.0,
        },
    )
    _write_json(
        e0_dir / f"{split}_manifest.json",
        {"run": "e0_dense", "split": split, "query_count": len(e0_rows)},
    )

    _write_jsonl(e1_dir / f"{split}_checkpoint.jsonl", e1_rows)
    _write_jsonl(e1_dir / f"{split}_results.jsonl", e1_rows)
    _write_json(
        e1_dir / f"{split}_metrics.json",
        {
            "query_count": len(e1_rows),
            "metrics": {"recall@5": 1.0, "recall@20": 1.0, "mrr@10": 1.0},
            "rerank_latency_p50_ms": 300.0,
            "rerank_latency_p95_ms": 400.0,
            "provider_total_tokens": sum(row["total_tokens"] for row in e1_rows),
        },
    )
    _write_json(
        e1_dir / f"{split}_manifest.json",
        {
            "identity": {
                "run": "e1_rerank",
                "split": split,
                "reranker": {
                    "model": "qwen3-rerank",
                    "candidate_chunk_k": 100,
                    "query_normalization": "rstrip",
                },
            },
            "provenance": {"project_sha": "synthetic-sha"},
        },
    )


def test_paired_evidence_uses_per_query_latency_sums() -> None:
    e0 = [
        _e0_row("Q1", "g1", ["x", "g1"], 100.0),
        _e0_row("Q2", "g2", ["x", "y"], 900.0),
    ]
    e1 = [
        _e1_row("Q1", "g1", ["g1"], 900.0),
        _e1_row("Q2", "g2", ["g2"], 100.0),
    ]

    evidence = build_paired_evidence(e0, e1)

    assert evidence["hit5_fixed"] == 1
    assert evidence["hit5_regressed"] == 0
    assert evidence["hit20_fixed"] == 1
    assert evidence["hit20_regressed"] == 0
    assert evidence["full_retrieval_latency_ms"] == [1000.0, 1000.0]
    assert evidence["full_retrieval_p95_ms"] == 1000.0


def test_materialize_rerank_evidence_emits_hashes_and_aggregate_only_dev_report(
    tmp_path: Path,
) -> None:
    e0_dir = tmp_path / "e0_dense"
    e1_dir = tmp_path / "e1_rerank"

    train_e0 = [
        _e0_row("TRAIN_Q1", "g1", ["x", "g1"], 100.0),
        _e0_row("TRAIN_Q2", "g2", ["x", "y"], 200.0),
    ]
    train_e1 = [
        _e1_row("TRAIN_Q1", "g1", ["g1"], 300.0),
        _e1_row("TRAIN_Q2", "g2", ["g2"], 400.0),
    ]
    dev_e0 = [
        _e0_row("DEV_SECRET_1", "g3", ["x", "g3"], 150.0),
        _e0_row("DEV_SECRET_2", "g4", ["x", "y"], 250.0),
    ]
    dev_e1 = [
        _e1_row("DEV_SECRET_1", "g3", ["g3"], 350.0),
        _e1_row("DEV_SECRET_2", "g4", ["g4"], 450.0),
    ]

    _write_split_artifacts(
        e0_dir=e0_dir,
        e1_dir=e1_dir,
        split="train",
        e0_rows=train_e0,
        e1_rows=train_e1,
    )
    _write_split_artifacts(
        e0_dir=e0_dir,
        e1_dir=e1_dir,
        split="dev",
        e0_rows=dev_e0,
        e1_rows=dev_e1,
    )

    result = materialize_rerank_evidence(
        e0_dir=e0_dir,
        e1_dir=e1_dir,
        expected_counts={"train": 2, "dev": 2},
        provider_region="ap-southeast-1",
    )

    hashes = json.loads(
        (e1_dir / "artifact_hashes.json").read_text(encoding="utf-8")
    )
    assert hashes["train"]["checkpoint_sha256"] == sha256_file(
        e1_dir / "train_checkpoint.jsonl"
    )
    assert hashes["train"]["results_sha256"] == sha256_file(
        e1_dir / "train_results.jsonl"
    )
    assert hashes["dev"]["checkpoint_sha256"] == sha256_file(
        e1_dir / "dev_checkpoint.jsonl"
    )
    assert hashes["dev"]["results_sha256"] == sha256_file(
        e1_dir / "dev_results.jsonl"
    )

    comparison = (e1_dir / "comparison.md").read_text(encoding="utf-8")
    assert "qwen3-rerank" in comparison
    assert "ap-southeast-1" in comparison
    assert "DEV_SECRET_1" not in comparison
    assert "DEV_SECRET_2" not in comparison
    assert result["dev"]["hit5_fixed"] == 1
    assert result["dev"]["hit5_regressed"] == 0


def test_materialize_rerank_evidence_rejects_checkpoint_result_drift(
    tmp_path: Path,
) -> None:
    e0_dir = tmp_path / "e0_dense"
    e1_dir = tmp_path / "e1_rerank"
    e0_rows = [_e0_row("TRAIN_Q1", "g1", ["x", "g1"], 100.0)]
    e1_rows = [_e1_row("TRAIN_Q1", "g1", ["g1"], 300.0)]
    dev_e0_rows = [_e0_row("DEV_Q1", "g2", ["x", "g2"], 100.0)]
    dev_e1_rows = [_e1_row("DEV_Q1", "g2", ["g2"], 300.0)]

    _write_split_artifacts(
        e0_dir=e0_dir,
        e1_dir=e1_dir,
        split="train",
        e0_rows=e0_rows,
        e1_rows=e1_rows,
    )
    _write_split_artifacts(
        e0_dir=e0_dir,
        e1_dir=e1_dir,
        split="dev",
        e0_rows=dev_e0_rows,
        e1_rows=dev_e1_rows,
    )
    drifted = [dict(e1_rows[0], document_ranking=["x"])]
    _write_jsonl(e1_dir / "train_checkpoint.jsonl", drifted)

    with pytest.raises(RuntimeError, match="checkpoint/results mismatch"):
        materialize_rerank_evidence(
            e0_dir=e0_dir,
            e1_dir=e1_dir,
            expected_counts={"train": 1, "dev": 1},
            provider_region="ap-southeast-1",
        )
