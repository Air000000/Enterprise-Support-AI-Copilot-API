from __future__ import annotations

from importlib import import_module

import pytest

from experiments.evals.ir.rrf import fuse_rrf

import hashlib

from pathlib import Path
import json


def _load_module():
    try:
        return import_module("experiments.evals.eval_techqa_hybrid")
    except ModuleNotFoundError as exc:
        if exc.name == "experiments.evals.eval_techqa_hybrid":
            pytest.fail("R3 hybrid evaluator is not implemented yet")
        raise


def _require_attr(module, name: str):
    value = getattr(module, name, None)

    if value is None:
        pytest.fail(f"{name} is not implemented yet")

    return value


def test_collapse_document_ids_preserves_first_occurrence() -> None:
    module = _load_module()
    collapse_document_ids = _require_attr(
        module,
        "collapse_document_ids",
    )

    assert collapse_document_ids(
        ["a", "a", "b", "c", "b"]
    ) == [
        "a",
        "b",
        "c",
    ]


def test_hybrid_summary_counts_complementary_hits_and_metrics() -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    build_hybrid_summary = _require_attr(
        module,
        "build_hybrid_summary",
    )

    dense_q1 = ("g1", "x")
    bm25_q1 = ("y", "z")

    dense_q2 = ("x", "y")
    bm25_q2 = ("g2", "z")

    hybrid_q1 = tuple(
        fuse_rrf(
            [dense_q1, bm25_q1],
            rrf_k=60,
            top_k=100,
        )
    )
    hybrid_q2 = tuple(
        fuse_rrf(
            [dense_q2, bm25_q2],
            rrf_k=60,
            top_k=100,
        )
    )

    results = [
        result_type(
            question_id="q1",
            relevant_document_ids=("g1",),
            dense_document_ids=dense_q1,
            bm25_document_ids=bm25_q1,
            hybrid_document_ids=hybrid_q1,
            bm25_latency_ms=10.0,
            fusion_latency_ms=1.0,
        ),
        result_type(
            question_id="q2",
            relevant_document_ids=("g2",),
            dense_document_ids=dense_q2,
            bm25_document_ids=bm25_q2,
            hybrid_document_ids=hybrid_q2,
            bm25_latency_ms=20.0,
            fusion_latency_ms=2.0,
        ),
    ]

    summary = build_hybrid_summary(results)

    assert summary.query_count == 2

    assert summary.dense_hit100 == 1
    assert summary.bm25_hit100 == 1
    assert summary.hybrid_hit100 == 2

    assert summary.dense_only_hits == 1
    assert summary.bm25_only_hits == 1
    assert summary.both_hits == 0
    assert summary.neither_hits == 0

    assert summary.recovered_dense_misses == 1
    assert summary.lost_dense_hits == 0

    assert summary.dense_metrics["recall@100"] == 0.5
    assert summary.bm25_metrics["recall@100"] == 0.5

    assert set(summary.hybrid_metrics) == {
        "recall@20",
        "recall@100",
        "mrr@10",
    }
    assert summary.hybrid_metrics["recall@20"] == 1.0
    assert summary.hybrid_metrics["recall@100"] == 1.0

def test_build_r3_manifest_locks_frozen_pilot_identity() -> None:
    module = _load_module()
    build_r3_manifest = _require_attr(
        module,
        "build_r3_manifest",
    )

    manifest = build_r3_manifest(
        dense_results_sha256="dense-results-sha",
        gate_sha256="gate-sha",
        corpus_sha256="corpus-sha",
        queries_sha256="queries-sha",
        qrels_sha256="qrels-sha",
    )

    assert manifest["benchmark"] == "TechQA-RAG-Eval"
    assert manifest["run"] == "r3_hybrid_pilot"
    assert manifest["split"] == "train"
    assert manifest["query_count"] == 450

    assert manifest["dense_source"] == {
        "candidate_chunk_k": 100,
        "document_rule": (
            "first occurrence over frozen E0 raw_document_ids"
        ),
        "results_sha256": "dense-results-sha",
    }

    assert manifest["bm25"] == {
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
    }

    assert manifest["rrf"] == {
        "rrf_k": 60,
        "top_k": 100,
        "weights": "equal",
    }

    assert manifest["retrieval_dataset"] == {
        "corpus_sha256": "corpus-sha",
        "queries_sha256": "queries-sha",
        "qrels_sha256": "qrels-sha",
    }

    assert manifest["gate_sha256"] == "gate-sha"
    assert manifest["provider_calls"] == 0


def test_evaluate_r3_gate_requires_both_thresholds() -> None:
    module = _load_module()
    evaluate_r3_gate = _require_attr(
        module,
        "evaluate_r3_gate",
    )

    gate = {
        "required_recovered_dense_misses": 10,
        "required_net_gain_cases": 7,
    }

    admitted = evaluate_r3_gate(
        dense_hit100=387,
        hybrid_hit100=394,
        recovered_dense_misses=10,
        gate=gate,
    )

    assert admitted == {
        "recovered_dense_misses": 10,
        "net_gain_cases": 7,
        "admitted": True,
        "status": "ADMIT_PAID_R4",
    }

    rejected = evaluate_r3_gate(
        dense_hit100=387,
        hybrid_hit100=395,
        recovered_dense_misses=9,
        gate=gate,
    )

    assert rejected == {
        "recovered_dense_misses": 9,
        "net_gain_cases": 8,
        "admitted": False,
        "status": "SKIP_PAID_R4",
    }

def test_evaluate_hybrid_row_uses_frozen_retrieval_contract() -> None:
    module = _load_module()
    evaluate_hybrid_row = _require_attr(
        module,
        "evaluate_hybrid_row",
    )

    search_calls: list[tuple[str, int]] = []

    def bm25_searcher(
        query: str,
        *,
        top_k: int,
    ) -> list[str]:
        search_calls.append((query, top_k))
        return ["b", "gold", "c"]

    clock_values = iter(
        [
            1.000,
            1.010,
            2.000,
            2.002,
        ]
    )

    result = evaluate_hybrid_row(
        {
            "question_id": "TRAIN_Q1",
            "question": "Error 0x80070005   \n",
            "relevant_document_ids": ["gold"],
            "raw_document_ids": [
                "a",
                "a",
                "gold",
                "x",
            ],
        },
        bm25_searcher=bm25_searcher,
        clock=lambda: next(clock_values),
    )

    assert search_calls == [
        ("Error 0x80070005", 100),
    ]

    assert result.question_id == "TRAIN_Q1"
    assert result.relevant_document_ids == ("gold",)

    assert result.dense_document_ids == (
        "a",
        "gold",
        "x",
    )
    assert result.bm25_document_ids == (
        "b",
        "gold",
        "c",
    )
    assert result.hybrid_document_ids == (
        "gold",
        "a",
        "b",
        "c",
        "x",
    )

    assert result.bm25_latency_ms == pytest.approx(10.0)
    assert result.fusion_latency_ms == pytest.approx(2.0)

def test_render_admission_decision_keeps_gate_frozen() -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    build_hybrid_summary = _require_attr(
        module,
        "build_hybrid_summary",
    )
    render_admission_decision = _require_attr(
        module,
        "render_admission_decision",
    )

    results = [
        result_type(
            question_id="q1",
            relevant_document_ids=("g1",),
            dense_document_ids=("g1",),
            bm25_document_ids=("x",),
            hybrid_document_ids=("g1",),
            bm25_latency_ms=10.0,
            fusion_latency_ms=1.0,
        ),
        result_type(
            question_id="q2",
            relevant_document_ids=("g2",),
            dense_document_ids=("x",),
            bm25_document_ids=("g2",),
            hybrid_document_ids=("g2",),
            bm25_latency_ms=20.0,
            fusion_latency_ms=2.0,
        ),
    ]
    summary = build_hybrid_summary(results)

    gate = {
        "required_recovered_dense_misses": 1,
        "required_net_gain_cases": 1,
    }

    decision = render_admission_decision(
        summary=summary,
        gate=gate,
    )

    threshold_pos = decision.index(
        "required_recovered_dense_misses = 1"
    )
    observed_pos = decision.index(
        "recovered_dense_misses = 1"
    )
    status_pos = decision.index("ADMIT_PAID_R4")

    assert threshold_pos < observed_pos < status_pos

    assert "required_net_gain_cases = 1" in decision
    assert "dense_hit100 = 1" in decision
    assert "hybrid_hit100 = 2" in decision
    assert "net_gain_cases = 1" in decision

    assert "recall@20" in decision
    assert "recall@100" in decision
    assert "mrr@10" in decision

    assert decision.count("ADMIT_PAID_R4") == 1
    assert "SKIP_PAID_R4" not in decision

    assert gate == {
        "required_recovered_dense_misses": 1,
        "required_net_gain_cases": 1,
    }

def test_build_preflight_report_locks_zero_provider_contract() -> None:
    module = _load_module()
    build_preflight_report = _require_attr(
        module,
        "build_preflight_report",
    )

    report = build_preflight_report(
        train_query_count=450,
        corpus_count=28481,
        dense_results_sha256="dense-sha",
        gate_sha256="gate-sha",
        bm25_version="0.3.10",
    )

    assert report == {
        "split": "train",
        "train_query_count": 450,
        "corpus_count": 28481,
        "dense_results_sha256": "dense-sha",
        "gate_sha256": "gate-sha",
        "bm25_version": "0.3.10",
        "provider_calls": 0,
        "dev_artifact_opened": False,
    }

def test_run_preflight_reads_train_inputs_and_hashes_files(
    tmp_path,
) -> None:
    module = _load_module()
    run_preflight = _require_attr(
        module,
        "run_preflight",
    )

    dense_results_path = tmp_path / "train_results.jsonl"
    dense_results_path.write_text(
        "".join(
            f'{{"question_id":"TRAIN_{index}"}}\n'
            for index in range(450)
        ),
        encoding="utf-8",
    )

    gate_path = tmp_path / "r3_gate.json"
    gate_path.write_text(
        (
            "{"
            '"query_count":450,'
            '"required_recovered_dense_misses":10,'
            '"required_net_gain_cases":7'
            "}\n"
        ),
        encoding="utf-8",
    )

    document_loader_calls = 0

    def document_loader():
        nonlocal document_loader_calls
        document_loader_calls += 1
        return [object()] * 28481

    version_calls: list[str] = []

    def version_loader(package: str) -> str:
        version_calls.append(package)
        return "0.3.10"

    report = run_preflight(
        dense_results_path=dense_results_path,
        gate_path=gate_path,
        document_loader=document_loader,
        version_loader=version_loader,
    )

    assert document_loader_calls == 1
    assert version_calls == ["bm25s"]

    assert report == {
        "split": "train",
        "train_query_count": 450,
        "corpus_count": 28481,
        "dense_results_sha256": hashlib.sha256(
            dense_results_path.read_bytes()
        ).hexdigest(),
        "gate_sha256": hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest(),
        "bm25_version": "0.3.10",
        "provider_calls": 0,
        "dev_artifact_opened": False,
    }

@pytest.mark.parametrize(
    ("mismatch", "expected_message"),
    [
        ("train_count", "TRAIN query count mismatch"),
        ("corpus_count", "TechQA corpus count mismatch"),
        ("gate_query_count", "R3 gate query count mismatch"),
        ("bm25_version", "bm25s version mismatch"),
    ],
)
def test_run_preflight_fails_closed_on_frozen_identity_mismatch(
    tmp_path,
    mismatch: str,
    expected_message: str,
) -> None:
    module = _load_module()
    run_preflight = _require_attr(
        module,
        "run_preflight",
    )

    train_count = 449 if mismatch == "train_count" else 450
    corpus_count = 28480 if mismatch == "corpus_count" else 28481
    gate_query_count = (
        449
        if mismatch == "gate_query_count"
        else 450
    )
    bm25_version = (
        "0.3.11"
        if mismatch == "bm25_version"
        else "0.3.10"
    )

    dense_results_path = tmp_path / "train_results.jsonl"
    dense_results_path.write_text(
        "".join(
            f'{{"question_id":"TRAIN_{index}"}}\n'
            for index in range(train_count)
        ),
        encoding="utf-8",
    )

    gate_path = tmp_path / "r3_gate.json"
    gate_path.write_text(
        (
            "{"
            f'"query_count":{gate_query_count},'
            '"required_recovered_dense_misses":10,'
            '"required_net_gain_cases":7'
            "}\n"
        ),
        encoding="utf-8",
    )

    def document_loader():
        return range(corpus_count)

    def version_loader(package: str) -> str:
        assert package == "bm25s"
        return bm25_version

    with pytest.raises(
        RuntimeError,
        match=expected_message,
    ):
        run_preflight(
            dense_results_path=dense_results_path,
            gate_path=gate_path,
            document_loader=document_loader,
            version_loader=version_loader,
        )

def test_main_preflight_uses_frozen_train_inputs_and_reports_contract(
    capsys,
) -> None:
    module = _load_module()
    main = _require_attr(
        module,
        "main",
    )

    calls: dict[str, object] = {}

    def preflight_runner(
        *,
        dense_results_path,
        gate_path,
        document_loader,
        version_loader,
    ):
        calls["dense_results_path"] = dense_results_path
        calls["gate_path"] = gate_path
        calls["document_loader"] = document_loader
        calls["version_loader"] = version_loader

        return {
            "split": "train",
            "train_query_count": 450,
            "corpus_count": 28481,
            "dense_results_sha256": "dense-sha",
            "gate_sha256": "gate-sha",
            "bm25_version": "0.3.10",
            "provider_calls": 0,
            "dev_artifact_opened": False,
        }

    document_loader = object()
    version_loader = object()

    main(
        ["--preflight"],
        preflight_runner=preflight_runner,
        document_loader=document_loader,
        version_loader=version_loader,
    )


    assert calls == {
        "dense_results_path": Path(
            "experiments/evals/reports/e0_dense/"
            "train_results.jsonl"
        ),
        "gate_path": Path(
            "experiments/evals/reports/e1_rerank/"
            "r3_gate.json"
        ),
        "document_loader": document_loader,
        "version_loader": version_loader,
    }

    output = capsys.readouterr().out

    assert "TRAIN count = 450" in output
    assert "TechQA corpus count = 28481" in output
    assert "frozen E0 input sha256 = dense-sha" in output
    assert "frozen R3 gate sha256 = gate-sha" in output
    assert "bm25s = 0.3.10" in output
    assert "provider_calls = 0" in output
    assert "dev_artifact_opened = False" in output

def test_evaluate_hybrid_rows_builds_results_and_summary() -> None:
    module = _load_module()
    evaluate_hybrid_rows = _require_attr(
        module,
        "evaluate_hybrid_rows",
    )

    rows = [
        {
            "question_id": "TRAIN_Q1",
            "question": "query one   \n",
            "relevant_document_ids": ["gold-1"],
            "raw_document_ids": [
                "gold-1",
                "dense-extra",
            ],
        },
        {
            "question_id": "TRAIN_Q2",
            "question": "query two\n",
            "relevant_document_ids": ["gold-2"],
            "raw_document_ids": [
                "dense-miss",
            ],
        },
    ]

    search_calls: list[tuple[str, int]] = []

    def bm25_searcher(
        query: str,
        *,
        top_k: int,
    ) -> list[str]:
        search_calls.append((query, top_k))
        if query == "query one":
            return ["bm25-miss"]
        if query == "query two":
            return ["gold-2"]
        raise AssertionError(f"unexpected query: {query}")

    results, summary = evaluate_hybrid_rows(
        rows,
        bm25_searcher=bm25_searcher,
        clock=lambda: 0.0,
    )

    assert search_calls == [
        ("query one", 100),
        ("query two", 100),
    ]

    assert tuple(
        result.question_id
        for result in results
    ) == (
        "TRAIN_Q1",
        "TRAIN_Q2",
    )

    assert summary.query_count == 2
    assert summary.dense_hit100 == 1
    assert summary.bm25_hit100 == 1
    assert summary.hybrid_hit100 == 2
    assert summary.dense_only_hits == 1
    assert summary.bm25_only_hits == 1
    assert summary.recovered_dense_misses == 1
    assert summary.lost_dense_hits == 0

def test_write_train_results_jsonl_persists_rankings_and_sha256(
    tmp_path,
) -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    write_train_results_jsonl = _require_attr(
        module,
        "write_train_results_jsonl",
    )

    results = [
        result_type(
            question_id="TRAIN_Q1",
            relevant_document_ids=("gold-1",),
            dense_document_ids=("gold-1", "dense-1"),
            bm25_document_ids=("bm25-1",),
            hybrid_document_ids=("gold-1", "bm25-1", "dense-1"),
            bm25_latency_ms=10.5,
            fusion_latency_ms=1.25,
        ),
        result_type(
            question_id="TRAIN_Q2",
            relevant_document_ids=("gold-2",),
            dense_document_ids=("dense-2",),
            bm25_document_ids=("gold-2",),
            hybrid_document_ids=("gold-2", "dense-2"),
            bm25_latency_ms=20.0,
            fusion_latency_ms=2.0,
        ),
    ]

    output_path = tmp_path / "train_results.jsonl"

    results_sha256 = write_train_results_jsonl(
        results,
        output_path,
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]

    assert rows == [
        {
            "question_id": "TRAIN_Q1",
            "relevant_document_ids": ["gold-1"],
            "dense_document_ids": ["gold-1", "dense-1"],
            "bm25_document_ids": ["bm25-1"],
            "hybrid_document_ids": [
                "gold-1",
                "bm25-1",
                "dense-1",
            ],
            "bm25_latency_ms": 10.5,
            "fusion_latency_ms": 1.25,
        },
        {
            "question_id": "TRAIN_Q2",
            "relevant_document_ids": ["gold-2"],
            "dense_document_ids": ["dense-2"],
            "bm25_document_ids": ["gold-2"],
            "hybrid_document_ids": [
                "gold-2",
                "dense-2",
            ],
            "bm25_latency_ms": 20.0,
            "fusion_latency_ms": 2.0,
        },
    ]

    assert results_sha256 == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

def test_write_train_metrics_json_persists_full_summary(
    tmp_path,
) -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    build_hybrid_summary = _require_attr(
        module,
        "build_hybrid_summary",
    )
    write_train_metrics_json = _require_attr(
        module,
        "write_train_metrics_json",
    )

    results = [
        result_type(
            question_id="TRAIN_Q1",
            relevant_document_ids=("gold-1",),
            dense_document_ids=("gold-1",),
            bm25_document_ids=("miss-1",),
            hybrid_document_ids=("gold-1", "miss-1"),
            bm25_latency_ms=10.0,
            fusion_latency_ms=1.0,
        ),
        result_type(
            question_id="TRAIN_Q2",
            relevant_document_ids=("gold-2",),
            dense_document_ids=("miss-2",),
            bm25_document_ids=("gold-2",),
            hybrid_document_ids=("gold-2", "miss-2"),
            bm25_latency_ms=20.0,
            fusion_latency_ms=2.0,
        ),
    ]

    summary = build_hybrid_summary(results)

    output_path = tmp_path / "train_metrics.json"

    write_train_metrics_json(
        summary,
        output_path,
    )

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload == {
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

def test_materialize_r3_artifacts_writes_complete_evidence_bundle(
    tmp_path,
) -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    build_hybrid_summary = _require_attr(
        module,
        "build_hybrid_summary",
    )
    materialize_r3_artifacts = _require_attr(
        module,
        "materialize_r3_artifacts",
    )

    results = [
        result_type(
            question_id="TRAIN_Q1",
            relevant_document_ids=("gold-1",),
            dense_document_ids=("gold-1",),
            bm25_document_ids=("miss-1",),
            hybrid_document_ids=("gold-1", "miss-1"),
            bm25_latency_ms=10.0,
            fusion_latency_ms=1.0,
        ),
        result_type(
            question_id="TRAIN_Q2",
            relevant_document_ids=("gold-2",),
            dense_document_ids=("miss-2",),
            bm25_document_ids=("gold-2",),
            hybrid_document_ids=("gold-2", "miss-2"),
            bm25_latency_ms=20.0,
            fusion_latency_ms=2.0,
        ),
    ]
    summary = build_hybrid_summary(results)

    gate = {
        "required_recovered_dense_misses": 1,
        "required_net_gain_cases": 1,
    }

    manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r3_hybrid_pilot",
        "split": "train",
        "query_count": 2,
        "provider_calls": 0,
    }

    materialize_r3_artifacts(
        results=results,
        summary=summary,
        gate=gate,
        manifest=manifest,
        output_dir=tmp_path,
    )

    results_path = tmp_path / "train_results.jsonl"
    metrics_path = tmp_path / "train_metrics.json"
    manifest_path = tmp_path / "train_manifest.json"
    decision_path = tmp_path / "admission_decision.md"

    assert results_path.exists()
    assert metrics_path.exists()
    assert manifest_path.exists()
    assert decision_path.exists()

    persisted_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert persisted_manifest[
        "train_results_sha256"
    ] == hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()

    assert persisted_manifest["benchmark"] == "TechQA-RAG-Eval"
    assert persisted_manifest["run"] == "r3_hybrid_pilot"
    assert persisted_manifest["split"] == "train"
    assert persisted_manifest["provider_calls"] == 0

    decision = decision_path.read_text(encoding="utf-8")

    assert decision.count("ADMIT_PAID_R4") == 1
    assert "SKIP_PAID_R4" not in decision
    assert decision.endswith("ADMIT_PAID_R4\n")

def test_load_e0_train_rows_preserves_required_frozen_fields(
    tmp_path,
) -> None:
    module = _load_module()
    load_e0_train_rows = _require_attr(
        module,
        "load_e0_train_rows",
    )

    input_path = tmp_path / "train_results.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "TRAIN_Q1",
                        "question": "query one   \n",
                        "relevant_document_ids": ["gold-1"],
                        "raw_chunk_ids": ["c1", "c2"],
                        "raw_document_ids": [
                            "doc-a",
                            "doc-a",
                            "gold-1",
                        ],
                        "document_ranking": [
                            "doc-a",
                            "gold-1",
                        ],
                        "latency_ms": 123.0,
                    }
                ),
                json.dumps(
                    {
                        "question_id": "TRAIN_Q2",
                        "question": "query two",
                        "relevant_document_ids": ["gold-2"],
                        "raw_chunk_ids": ["c3"],
                        "raw_document_ids": ["gold-2"],
                        "document_ranking": ["gold-2"],
                        "latency_ms": 456.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_e0_train_rows(
        input_path,
        expected_count=2,
    )

    assert rows == [
        {
            "question_id": "TRAIN_Q1",
            "question": "query one   \n",
            "relevant_document_ids": ["gold-1"],
            "raw_document_ids": [
                "doc-a",
                "doc-a",
                "gold-1",
            ],
        },
        {
            "question_id": "TRAIN_Q2",
            "question": "query two",
            "relevant_document_ids": ["gold-2"],
            "raw_document_ids": ["gold-2"],
        },
    ]

def test_load_e0_train_rows_fails_closed_on_count_mismatch(
    tmp_path,
) -> None:
    module = _load_module()
    load_e0_train_rows = _require_attr(
        module,
        "load_e0_train_rows",
    )

    input_path = tmp_path / "train_results.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "question_id": "TRAIN_Q1",
                "question": "query one",
                "relevant_document_ids": ["gold-1"],
                "raw_document_ids": ["gold-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="E0 TRAIN row count mismatch",
    ):
        load_e0_train_rows(
            input_path,
            expected_count=2,
        )

def test_build_r3_manifest_from_frozen_inputs_hashes_sources(
    tmp_path,
) -> None:
    module = _load_module()
    build_r3_manifest_from_frozen_inputs = _require_attr(
        module,
        "build_r3_manifest_from_frozen_inputs",
    )

    dense_results_path = tmp_path / "train_results.jsonl"
    dense_results_path.write_text(
        '{"question_id":"TRAIN_Q1"}\n',
        encoding="utf-8",
    )

    gate_path = tmp_path / "r3_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "split": "train",
                "query_count": 450,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "corpus_sha256": "corpus-hash",
                    "queries_sha256": "queries-hash",
                    "qrels_sha256": "qrels-hash",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_r3_manifest_from_frozen_inputs(
        dense_results_path=dense_results_path,
        gate_path=gate_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    assert manifest[
        "dense_source"
    ]["results_sha256"] == hashlib.sha256(
        dense_results_path.read_bytes()
    ).hexdigest()

    assert manifest["gate_sha256"] == hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()

    assert manifest["retrieval_dataset"] == {
        "corpus_sha256": "corpus-hash",
        "queries_sha256": "queries-hash",
        "qrels_sha256": "qrels-hash",
    }

    assert manifest["provider_calls"] == 0

def test_run_r3_hybrid_pilot_executes_full_local_pipeline(
    tmp_path,
) -> None:
    module = _load_module()
    run_r3_hybrid_pilot = _require_attr(
        module,
        "run_r3_hybrid_pilot",
    )

    dense_results_path = tmp_path / "e0_train_results.jsonl"
    dense_results_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "question_id": "TRAIN_Q1",
                        "question": "query one\n",
                        "relevant_document_ids": ["gold-1"],
                        "raw_document_ids": [
                            "gold-1",
                            "dense-extra",
                        ],
                    }
                ),
                json.dumps(
                    {
                        "question_id": "TRAIN_Q2",
                        "question": "query two   \n",
                        "relevant_document_ids": ["gold-2"],
                        "raw_document_ids": [
                            "dense-miss",
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    gate_path = tmp_path / "r3_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "split": "train",
                "query_count": 2,
                "required_recovered_dense_misses": 1,
                "required_net_gain_cases": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "corpus_sha256": "corpus-hash",
                    "queries_sha256": "queries-hash",
                    "qrels_sha256": "qrels-hash",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    documents = [
        object(),
        object(),
    ]
    loader_calls = 0

    def document_loader():
        nonlocal loader_calls
        loader_calls += 1
        return documents

    factory_calls = 0

    class FakeBM25Retriever:
        def search(
            self,
            query: str,
            *,
            top_k: int,
        ) -> list[str]:
            assert top_k == 100

            if query == "query one":
                return ["bm25-miss"]

            if query == "query two":
                return ["gold-2"]

            raise AssertionError(
                f"unexpected query: {query}"
            )

    def bm25_factory(received_documents):
        nonlocal factory_calls
        factory_calls += 1

        assert received_documents is documents

        return FakeBM25Retriever()

    output_dir = tmp_path / "r3_hybrid"

    summary = run_r3_hybrid_pilot(
        dense_results_path=dense_results_path,
        gate_path=gate_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=output_dir,
        expected_query_count=2,
        document_loader=document_loader,
        bm25_factory=bm25_factory,
        clock=lambda: 0.0,
    )

    assert loader_calls == 1
    assert factory_calls == 1

    assert summary.query_count == 2
    assert summary.dense_hit100 == 1
    assert summary.bm25_hit100 == 1
    assert summary.hybrid_hit100 == 2
    assert summary.recovered_dense_misses == 1
    assert summary.lost_dense_hits == 0

    assert (
        output_dir / "train_results.jsonl"
    ).exists()
    assert (
        output_dir / "train_metrics.json"
    ).exists()
    assert (
        output_dir / "train_manifest.json"
    ).exists()
    assert (
        output_dir / "admission_decision.md"
    ).exists()

    decision = (
        output_dir / "admission_decision.md"
    ).read_text(encoding="utf-8")

    assert decision.endswith("ADMIT_PAID_R4\n")

def test_main_without_flags_runs_formal_r3_pilot() -> None:
    module = _load_module()
    main = _require_attr(
        module,
        "main",
    )

    calls = []

    def fake_r3_runner(**kwargs):
        calls.append(kwargs)
        return None

    fake_document_loader = object()
    fake_bm25_factory = object()
    fake_clock = object()

    main(
        [],
        r3_runner=fake_r3_runner,
        document_loader=fake_document_loader,
        bm25_factory=fake_bm25_factory,
        clock=fake_clock,
    )

    assert len(calls) == 1

    call = calls[0]

    assert call["dense_results_path"] == Path(
        "experiments/evals/reports/e0_dense/"
        "train_results.jsonl"
    )
    assert call["gate_path"] == Path(
        "experiments/evals/reports/e1_rerank/"
        "r3_gate.json"
    )
    assert call["dataset_manifest_path"] == Path(
        "experiments/evals/datasets/techqa/"
        "manifest.json"
    )
    assert call["output_dir"] == Path(
        "experiments/evals/reports/r3_hybrid"
    )
    assert call["expected_query_count"] == 450

    assert call["document_loader"] is fake_document_loader
    assert call["bm25_factory"] is fake_bm25_factory
    assert call["clock"] is fake_clock

def test_verify_r3_artifacts_recomputes_metrics_gate_and_hashes(
    tmp_path,
) -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    build_hybrid_summary = _require_attr(
        module,
        "build_hybrid_summary",
    )
    materialize_r3_artifacts = _require_attr(
        module,
        "materialize_r3_artifacts",
    )
    verify_r3_artifacts = _require_attr(
        module,
        "verify_r3_artifacts",
    )

    dense_results_path = tmp_path / "e0_train_results.jsonl"
    dense_results_path.write_text(
        '{"frozen":"dense"}\n',
        encoding="utf-8",
    )

    gate = {
        "required_recovered_dense_misses": 1,
        "required_net_gain_cases": 1,
    }
    gate_path = tmp_path / "r3_gate.json"
    gate_path.write_text(
        json.dumps(gate) + "\n",
        encoding="utf-8",
    )

    dataset_manifest = {
        "retrieval_dataset": {
            "corpus_sha256": "corpus-hash",
            "queries_sha256": "queries-hash",
            "qrels_sha256": "qrels-hash",
        }
    }
    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(dataset_manifest) + "\n",
        encoding="utf-8",
    )

    results = [
        result_type(
            question_id="TRAIN_Q1",
            relevant_document_ids=("gold-1",),
            dense_document_ids=("gold-1",),
            bm25_document_ids=("miss-1",),
            hybrid_document_ids=("gold-1", "miss-1"),
            bm25_latency_ms=10.0,
            fusion_latency_ms=1.0,
        ),
        result_type(
            question_id="TRAIN_Q2",
            relevant_document_ids=("gold-2",),
            dense_document_ids=("miss-2",),
            bm25_document_ids=("gold-2",),
            hybrid_document_ids=("gold-2", "miss-2"),
            bm25_latency_ms=20.0,
            fusion_latency_ms=2.0,
        ),
    ]
    summary = build_hybrid_summary(results)

    manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r3_hybrid_pilot",
        "split": "train",
        "query_count": 2,
        "dense_source": {
            "results_sha256": hashlib.sha256(
                dense_results_path.read_bytes()
            ).hexdigest(),
        },
        "retrieval_dataset": {
            "corpus_sha256": "corpus-hash",
            "queries_sha256": "queries-hash",
            "qrels_sha256": "qrels-hash",
        },
        "gate_sha256": hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest(),
        "provider_calls": 0,
    }

    output_dir = tmp_path / "r3_hybrid"

    materialize_r3_artifacts(
        results=results,
        summary=summary,
        gate=gate,
        manifest=manifest,
        output_dir=output_dir,
    )

    status = verify_r3_artifacts(
        output_dir=output_dir,
        dense_results_path=dense_results_path,
        gate_path=gate_path,
        dataset_manifest_path=dataset_manifest_path,
    )

    assert status == (
        "R3 HYBRID ARTIFACT VERIFICATION = OK"
    )

def test_verify_r3_artifacts_fails_closed_on_results_tamper(
    tmp_path,
) -> None:
    module = _load_module()

    result_type = _require_attr(
        module,
        "TechQAHybridResult",
    )
    build_hybrid_summary = _require_attr(
        module,
        "build_hybrid_summary",
    )
    materialize_r3_artifacts = _require_attr(
        module,
        "materialize_r3_artifacts",
    )
    verify_r3_artifacts = _require_attr(
        module,
        "verify_r3_artifacts",
    )

    dense_results_path = tmp_path / "e0_train_results.jsonl"
    dense_results_path.write_text(
        '{"frozen":"dense"}\n',
        encoding="utf-8",
    )

    gate = {
        "required_recovered_dense_misses": 0,
        "required_net_gain_cases": 0,
    }
    gate_path = tmp_path / "r3_gate.json"
    gate_path.write_text(
        json.dumps(gate) + "\n",
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "corpus_sha256": "corpus-hash",
                    "queries_sha256": "queries-hash",
                    "qrels_sha256": "qrels-hash",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    results = [
        result_type(
            question_id="TRAIN_Q1",
            relevant_document_ids=("gold-1",),
            dense_document_ids=("gold-1",),
            bm25_document_ids=("gold-1",),
            hybrid_document_ids=("gold-1",),
            bm25_latency_ms=1.0,
            fusion_latency_ms=1.0,
        ),
    ]

    summary = build_hybrid_summary(results)

    manifest = {
        "benchmark": "TechQA-RAG-Eval",
        "run": "r3_hybrid_pilot",
        "split": "train",
        "query_count": 1,
        "dense_source": {
            "results_sha256": hashlib.sha256(
                dense_results_path.read_bytes()
            ).hexdigest(),
        },
        "retrieval_dataset": {
            "corpus_sha256": "corpus-hash",
            "queries_sha256": "queries-hash",
            "qrels_sha256": "qrels-hash",
        },
        "gate_sha256": hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest(),
        "provider_calls": 0,
    }

    output_dir = tmp_path / "r3_hybrid"

    materialize_r3_artifacts(
        results=results,
        summary=summary,
        gate=gate,
        manifest=manifest,
        output_dir=output_dir,
    )

    results_path = output_dir / "train_results.jsonl"

    original = results_path.read_text(
        encoding="utf-8",
    )

    results_path.write_text(
        original + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="train_results SHA256 verification failed",
    ):
        verify_r3_artifacts(
            output_dir=output_dir,
            dense_results_path=dense_results_path,
            gate_path=gate_path,
            dataset_manifest_path=dataset_manifest_path,
        )

def test_main_verify_runs_artifact_verification(
    capsys,
) -> None:
    module = _load_module()
    main = _require_attr(
        module,
        "main",
    )

    calls = []

    def fake_verify_runner(**kwargs):
        calls.append(kwargs)
        return "R3 HYBRID ARTIFACT VERIFICATION = OK"

    main(
        ["--verify"],
        verify_runner=fake_verify_runner,
    )

    assert len(calls) == 1

    call = calls[0]

    assert call["output_dir"] == Path(
        "experiments/evals/reports/r3_hybrid"
    )
    assert call["dense_results_path"] == Path(
        "experiments/evals/reports/e0_dense/"
        "train_results.jsonl"
    )
    assert call["gate_path"] == Path(
        "experiments/evals/reports/e1_rerank/"
        "r3_gate.json"
    )
    assert call["dataset_manifest_path"] == Path(
        "experiments/evals/datasets/techqa/"
        "manifest.json"
    )

    captured = capsys.readouterr()

    assert captured.out == (
        "R3 HYBRID ARTIFACT VERIFICATION = OK\n"
    )