from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_E0_DIR = Path("experiments/evals/reports/e0_dense")
DEFAULT_E1_DIR = Path("experiments/evals/reports/e1_rerank")
DEFAULT_EXPECTED_COUNTS = {"train": 450, "dev": 160}
DEFAULT_PROVIDER_REGION = "ap-southeast-1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _index_unique(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row["question_id"])
        if question_id in indexed:
            raise RuntimeError(f"duplicate {label} question_id: {question_id}")
        indexed[question_id] = row
    return indexed


def first_relevant_rank(row: Mapping[str, Any]) -> int | None:
    relevant = {str(value) for value in row["relevant_document_ids"]}
    for rank, document_id in enumerate(row["document_ranking"], start=1):
        if str(document_id) in relevant:
            return rank
    return None


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_paired_evidence(
    e0_rows: list[dict[str, Any]],
    e1_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    e0_by_id = _index_unique(e0_rows, label="E0")
    e1_by_id = _index_unique(e1_rows, label="E1")
    if set(e0_by_id) != set(e1_by_id):
        raise RuntimeError("E0/E1 question_id mismatch")

    hit5_fixed = 0
    hit5_regressed = 0
    hit20_fixed = 0
    hit20_regressed = 0
    full_retrieval_latency_ms: list[float] = []

    for question_id in sorted(e0_by_id):
        e0 = e0_by_id[question_id]
        e1 = e1_by_id[question_id]
        before_rank = first_relevant_rank(e0)
        after_rank = first_relevant_rank(e1)

        before_hit5 = before_rank is not None and before_rank <= 5
        after_hit5 = after_rank is not None and after_rank <= 5
        before_hit20 = before_rank is not None and before_rank <= 20
        after_hit20 = after_rank is not None and after_rank <= 20

        hit5_fixed += int((not before_hit5) and after_hit5)
        hit5_regressed += int(before_hit5 and (not after_hit5))
        hit20_fixed += int((not before_hit20) and after_hit20)
        hit20_regressed += int(before_hit20 and (not after_hit20))

        full_retrieval_latency_ms.append(
            float(e0["latency_ms"]) + float(e1["rerank_latency_ms"])
        )

    return {
        "hit5_fixed": hit5_fixed,
        "hit5_regressed": hit5_regressed,
        "hit20_fixed": hit20_fixed,
        "hit20_regressed": hit20_regressed,
        "full_retrieval_latency_ms": full_retrieval_latency_ms,
        "full_retrieval_p50_ms": percentile(full_retrieval_latency_ms, 0.50),
        "full_retrieval_p95_ms": percentile(full_retrieval_latency_ms, 0.95),
    }


def _validate_split(
    *,
    split: str,
    e0_dir: Path,
    e1_dir: Path,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    e0_results_path = e0_dir / f"{split}_results.jsonl"
    e1_checkpoint_path = e1_dir / f"{split}_checkpoint.jsonl"
    e1_results_path = e1_dir / f"{split}_results.jsonl"
    e0_metrics_path = e0_dir / f"{split}_metrics.json"
    e1_metrics_path = e1_dir / f"{split}_metrics.json"

    e0_rows = _load_jsonl(e0_results_path)
    e1_checkpoint_rows = _load_jsonl(e1_checkpoint_path)
    e1_rows = _load_jsonl(e1_results_path)

    if len(e0_rows) != expected_count:
        raise RuntimeError(
            f"{split} E0 row count mismatch: expected={expected_count}, actual={len(e0_rows)}"
        )
    if len(e1_checkpoint_rows) != expected_count:
        raise RuntimeError(
            f"{split} E1 checkpoint count mismatch: "
            f"expected={expected_count}, actual={len(e1_checkpoint_rows)}"
        )
    if len(e1_rows) != expected_count:
        raise RuntimeError(
            f"{split} E1 result count mismatch: expected={expected_count}, actual={len(e1_rows)}"
        )

    checkpoint_by_id = _index_unique(e1_checkpoint_rows, label=f"{split} checkpoint")
    result_by_id = _index_unique(e1_rows, label=f"{split} result")
    if checkpoint_by_id != result_by_id:
        raise RuntimeError(f"{split} checkpoint/results mismatch")

    e0_by_id = _index_unique(e0_rows, label=f"{split} E0")
    if set(e0_by_id) != set(result_by_id):
        raise RuntimeError(f"{split} E0/E1 question_id mismatch")

    e0_metrics = _load_json(e0_metrics_path)
    e1_metrics = _load_json(e1_metrics_path)
    if int(e0_metrics["query_count"]) != expected_count:
        raise RuntimeError(f"{split} E0 metrics query_count mismatch")
    if int(e1_metrics["query_count"]) != expected_count:
        raise RuntimeError(f"{split} E1 metrics query_count mismatch")

    return e0_rows, e1_rows, e0_metrics, e1_metrics


def _artifact_hashes(e0_dir: Path, e1_dir: Path, split: str) -> dict[str, str]:
    return {
        "e0_manifest_sha256": sha256_file(e0_dir / f"{split}_manifest.json"),
        "e0_metrics_sha256": sha256_file(e0_dir / f"{split}_metrics.json"),
        "e0_results_sha256": sha256_file(e0_dir / f"{split}_results.jsonl"),
        "manifest_sha256": sha256_file(e1_dir / f"{split}_manifest.json"),
        "metrics_sha256": sha256_file(e1_dir / f"{split}_metrics.json"),
        "checkpoint_sha256": sha256_file(e1_dir / f"{split}_checkpoint.jsonl"),
        "results_sha256": sha256_file(e1_dir / f"{split}_results.jsonl"),
    }


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    return float(metrics["metrics"][name])


def _render_comparison(
    *,
    split_payloads: Mapping[str, Mapping[str, Any]],
    provider_region: str,
    reranker_model: str,
) -> str:
    lines = [
        "# E0 Dense vs E1 Rerank Evidence",
        "",
        f"- Reranker: `{reranker_model}`",
        f"- Provider region: `{provider_region}`",
        "- DEV evidence is aggregate-only; individual DEV failure IDs are intentionally omitted.",
        "",
    ]

    for split in ("train", "dev"):
        payload = split_payloads[split]
        e0_metrics = payload["e0_metrics"]
        e1_metrics = payload["e1_metrics"]
        paired = payload["paired"]
        lines.extend(
            [
                f"## {split.upper()}",
                "",
                "| Metric | E0 Dense | E1 Rerank |",
                "| --- | ---: | ---: |",
                f"| Recall@5 | {_metric(e0_metrics, 'recall@5'):.6f} | {_metric(e1_metrics, 'recall@5'):.6f} |",
                f"| Recall@20 | {_metric(e0_metrics, 'recall@20'):.6f} | {_metric(e1_metrics, 'recall@20'):.6f} |",
                f"| MRR@10 | {_metric(e0_metrics, 'mrr@10'):.6f} | {_metric(e1_metrics, 'mrr@10'):.6f} |",
                "",
                f"- Top-5 fixed/regressed: {paired['hit5_fixed']}/{paired['hit5_regressed']}",
                f"- Top-20 fixed/regressed: {paired['hit20_fixed']}/{paired['hit20_regressed']}",
                f"- E1 full retrieval p50: {paired['full_retrieval_p50_ms']:.3f} ms",
                f"- E1 full retrieval p95: {paired['full_retrieval_p95_ms']:.3f} ms",
                f"- Provider total tokens: {int(e1_metrics.get('provider_total_tokens', 0))}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def materialize_rerank_evidence(
    *,
    e0_dir: str | Path = DEFAULT_E0_DIR,
    e1_dir: str | Path = DEFAULT_E1_DIR,
    expected_counts: Mapping[str, int] = DEFAULT_EXPECTED_COUNTS,
    provider_region: str = DEFAULT_PROVIDER_REGION,
) -> dict[str, Any]:
    e0_root = Path(e0_dir)
    e1_root = Path(e1_dir)
    e1_root.mkdir(parents=True, exist_ok=True)

    split_payloads: dict[str, dict[str, Any]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    reranker_model: str | None = None

    for split in ("train", "dev"):
        if split not in expected_counts:
            raise RuntimeError(f"missing expected count for split: {split}")
        expected_count = int(expected_counts[split])
        e0_rows, e1_rows, e0_metrics, e1_metrics = _validate_split(
            split=split,
            e0_dir=e0_root,
            e1_dir=e1_root,
            expected_count=expected_count,
        )
        manifest = _load_json(e1_root / f"{split}_manifest.json")
        model = str(manifest["identity"]["reranker"]["model"])
        if reranker_model is None:
            reranker_model = model
        elif model != reranker_model:
            raise RuntimeError("TRAIN/DEV reranker model mismatch")

        paired = build_paired_evidence(e0_rows, e1_rows)
        split_payloads[split] = {
            "e0_metrics": e0_metrics,
            "e1_metrics": e1_metrics,
            "paired": paired,
        }
        artifact_hashes[split] = _artifact_hashes(e0_root, e1_root, split)

    if reranker_model is None:
        raise RuntimeError("reranker model could not be resolved")

    (e1_root / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (e1_root / "comparison.md").write_text(
        _render_comparison(
            split_payloads=split_payloads,
            provider_region=provider_region,
            reranker_model=reranker_model,
        ),
        encoding="utf-8",
    )

    return {
        split: {
            **payload["paired"],
            "e0_metrics": payload["e0_metrics"],
            "e1_metrics": payload["e1_metrics"],
        }
        for split, payload in split_payloads.items()
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize compact E1 rerank evidence.")
    parser.add_argument("--provider-region", default=DEFAULT_PROVIDER_REGION)
    args = parser.parse_args([] if argv is None else argv)

    result = materialize_rerank_evidence(provider_region=str(args.provider_region))
    print("E1 compact rerank evidence materialized.")
    for split in ("train", "dev"):
        paired = result[split]
        print(
            f"{split}: top5 fixed/regressed="
            f"{paired['hit5_fixed']}/{paired['hit5_regressed']}, "
            f"full retrieval p95={paired['full_retrieval_p95_ms']:.3f} ms"
        )
    print("provider_calls = 0")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
