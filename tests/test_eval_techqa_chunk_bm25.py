from importlib import import_module

import pytest
from experiments.evals.adapters.techqa import TechQARetrievalCase
import json
import hashlib

def _load_module():
    try:
        return import_module("experiments.evals.eval_techqa_chunk_bm25")
    except ModuleNotFoundError:
        pytest.fail("chunk BM25 audit evaluator is not implemented yet")


def test_cutoff_observation_measures_duplicate_slot_pressure() -> None:
    module = _load_module()

    observation = module.build_cutoff_observation(
        raw_document_ids=[
            "a",
            "a",
            "a",
            "b",
            "c",
            "gold",
        ],
        relevant_document_ids=["gold"],
        cutoff=5,
    )

    assert observation.returned_chunk_count == 5
    assert observation.unique_document_count == 3
    assert observation.duplicate_slot_count == 2
    assert observation.duplicate_ratio == pytest.approx(0.4)
    assert observation.gold_document_hit_within_chunk_k is False

def test_paired_ranks_identify_crowding_rescue() -> None:
    module = _load_module()

    raw_document_ids = [
        "a",
        "a",
        "a",
        "b",
        "a",
        "c",
        "d",
        "e",
        "f",
        "gold",
    ]

    collapsed = module.collapse_document_ids(raw_document_ids)

    assert module.first_relevant_rank(
        raw_document_ids,
        ["gold"],
    ) == 10

    assert module.first_relevant_rank(
        collapsed,
        ["gold"],
    ) == 7

    observation = module.build_cutoff_observation(
        raw_document_ids=raw_document_ids,
        relevant_document_ids=["gold"],
        cutoff=8,
    )

    assert observation.gold_document_hit_within_chunk_k is False
    assert observation.crowding_rescue is True

def test_adaptive_depth_expands_without_qrels_until_100_unique_documents() -> None:
    module = _load_module()
    calls: list[int] = []

    def searcher(
        query: str,
        top_k: int,
    ):
        calls.append(top_k)

        if top_k == 500:
            return [
                module.TechQAChunkRef(
                    f"a_chunk_{index}",
                    "a",
                    index,
                )
                for index in range(500)
            ]

        return [
            module.TechQAChunkRef(
                f"doc{index}_chunk_0",
                f"doc{index}",
                0,
            )
            for index in range(100)
        ]

    clock_values = iter(
        [
            1.000,
            1.010,
            2.000,
            2.030,
        ]
    )

    result = module.retrieve_with_unique_document_depth(
        "query",
        searcher=searcher,
        clock=lambda: next(clock_values),
        max_depth=2000,
    )

    assert calls == [500, 1000]
    assert result.final_search_depth == 1000
    assert len(
        module.collapse_document_ids(
            [
                item.document_id
                for item in result.candidates
            ]
        )
    ) == 100
    assert result.latency_ms == pytest.approx(40.0)

def test_evaluate_audit_case_keeps_raw_chunk_and_unique_document_budgets_separate() -> None:
    module = _load_module()

    case = TechQARetrievalCase(
        question_id="TRAIN_Q1",
        question="Error 0x80070005   \n",
        relevant_document_ids=("gold",),
        split="train",
    )

    candidates = [
        module.TechQAChunkRef("a0", "a", 0),
        module.TechQAChunkRef("a1", "a", 1),
        module.TechQAChunkRef("b0", "b", 0),
        module.TechQAChunkRef("gold0", "gold", 0),
    ] + [
        module.TechQAChunkRef(
            f"d{index}_chunk_0",
            f"d{index}",
            0,
        )
        for index in range(100)
    ]

    def searcher(
        query: str,
        top_k: int,
    ):
        assert query == "Error 0x80070005"
        return candidates[:top_k]

    clock_values = iter([1.000, 1.010])

    result = module.evaluate_audit_case(
        case,
        searcher=searcher,
        clock=lambda: next(clock_values),
        max_depth=len(candidates),
    )

    assert result.raw_top100_document_ids[:4] == (
        "a",
        "a",
        "b",
        "gold",
    )

    assert result.document_top100[:3] == (
        "a",
        "b",
        "gold",
    )

    assert result.first_gold_chunk_rank == 4
    assert result.first_gold_document_rank == 3
    assert result.crowding_gap == 1

    assert result.cutoff_observations[0].cutoff == 20

def test_build_audit_summary_aggregates_diversity_coverage_and_document_recall() -> None:
    module = _load_module()

    def observation(
        cutoff: int,
        *,
        unique_document_count: int,
        duplicate_ratio: float,
        gold_hit: bool,
        crowding_rescue: bool,
    ):
        returned_chunk_count = cutoff
        duplicate_slot_count = round(
            returned_chunk_count * duplicate_ratio
        )

        return module.ChunkCutoffObservation(
            cutoff=cutoff,
            returned_chunk_count=returned_chunk_count,
            unique_document_count=unique_document_count,
            duplicate_slot_count=duplicate_slot_count,
            duplicate_ratio=duplicate_ratio,
            gold_document_hit_within_chunk_k=gold_hit,
            crowding_rescue=crowding_rescue,
        )

    results = [
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q1",
            question="q1",
            relevant_document_ids=("gold1",),
            audit_search_depth=500,
            latency_ms=10.0,
            raw_top100_chunk_ids=("c1",),
            raw_top100_document_ids=("gold1",),
            document_top100=("gold1", "x"),
            first_gold_chunk_rank=1,
            first_gold_document_rank=1,
            crowding_gap=0,
            cutoff_observations=(
                observation(
                    20,
                    unique_document_count=18,
                    duplicate_ratio=0.10,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
                observation(
                    50,
                    unique_document_count=40,
                    duplicate_ratio=0.20,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
                observation(
                    100,
                    unique_document_count=75,
                    duplicate_ratio=0.25,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
            ),
        ),
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q2",
            question="q2",
            relevant_document_ids=("gold2",),
            audit_search_depth=1000,
            latency_ms=30.0,
            raw_top100_chunk_ids=("c2",),
            raw_top100_document_ids=("x",),
            document_top100=("x", "gold2"),
            first_gold_chunk_rank=25,
            first_gold_document_rank=2,
            crowding_gap=23,
            cutoff_observations=(
                observation(
                    20,
                    unique_document_count=10,
                    duplicate_ratio=0.50,
                    gold_hit=False,
                    crowding_rescue=True,
                ),
                observation(
                    50,
                    unique_document_count=30,
                    duplicate_ratio=0.40,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
                observation(
                    100,
                    unique_document_count=60,
                    duplicate_ratio=0.40,
                    gold_hit=True,
                    crowding_rescue=False,
                ),
            ),
        ),
    ]

    summary = module.build_audit_summary(results)

    assert summary["query_count"] == 2

    cutoff20 = summary["cutoffs"]["20"]

    assert (
        cutoff20["unique_document_count_p05"]
        <= cutoff20["unique_document_count_p50"]
    )
    assert (
        cutoff20["duplicate_ratio_p95"]
        >= cutoff20["duplicate_ratio_p50"]
    )

    assert cutoff20["gold_document_hit_count"] == 1
    assert cutoff20["gold_document_hit_rate"] == pytest.approx(0.5)

    assert cutoff20["crowding_rescue_count"] == 1
    assert cutoff20["crowding_rescue_rate"] == pytest.approx(0.5)

    assert summary["collapsed_document_recall"] == {
        "recall@20": pytest.approx(1.0),
        "recall@50": pytest.approx(1.0),
        "recall@100": pytest.approx(1.0),
    }

    assert summary["crowding_gap"]["observed_count"] == 2
    assert summary["latency_ms"]["p50"] == pytest.approx(20.0)
    assert summary["latency_ms"]["p95"] > summary["latency_ms"]["p50"]

def test_build_diagnostic_cases_orders_high_duplication_and_crowding_rescue_deterministically() -> None:
    module = _load_module()

    def make_observation(
        cutoff: int,
        *,
        duplicate_ratio: float,
        crowding_rescue: bool,
    ):
        returned_chunk_count = cutoff
        unique_document_count = round(
            returned_chunk_count * (1.0 - duplicate_ratio)
        )

        return module.ChunkCutoffObservation(
            cutoff=cutoff,
            returned_chunk_count=returned_chunk_count,
            unique_document_count=unique_document_count,
            duplicate_slot_count=(
                returned_chunk_count - unique_document_count
            ),
            duplicate_ratio=duplicate_ratio,
            gold_document_hit_within_chunk_k=(
                not crowding_rescue
            ),
            crowding_rescue=crowding_rescue,
        )

    results = [
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q1",
            question="q1",
            relevant_document_ids=("gold1",),
            audit_search_depth=500,
            latency_ms=10.0,
            raw_top100_chunk_ids=("q1_c1",),
            raw_top100_document_ids=("a",),
            document_top100=("a", "gold1"),
            first_gold_chunk_rank=80,
            first_gold_document_rank=10,
            crowding_gap=70,
            cutoff_observations=(
                make_observation(
                    20,
                    duplicate_ratio=0.20,
                    crowding_rescue=True,
                ),
                make_observation(
                    50,
                    duplicate_ratio=0.30,
                    crowding_rescue=False,
                ),
                make_observation(
                    100,
                    duplicate_ratio=0.40,
                    crowding_rescue=False,
                ),
            ),
        ),
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q2",
            question="q2",
            relevant_document_ids=("gold2",),
            audit_search_depth=500,
            latency_ms=20.0,
            raw_top100_chunk_ids=("q2_c1",),
            raw_top100_document_ids=("b",),
            document_top100=("b", "gold2"),
            first_gold_chunk_rank=95,
            first_gold_document_rank=15,
            crowding_gap=80,
            cutoff_observations=(
                make_observation(
                    20,
                    duplicate_ratio=0.50,
                    crowding_rescue=True,
                ),
                make_observation(
                    50,
                    duplicate_ratio=0.40,
                    crowding_rescue=False,
                ),
                make_observation(
                    100,
                    duplicate_ratio=0.60,
                    crowding_rescue=False,
                ),
            ),
        ),
        module.TechQAChunkBM25AuditResult(
            question_id="TRAIN_Q0",
            question="q0",
            relevant_document_ids=("gold0",),
            audit_search_depth=500,
            latency_ms=30.0,
            raw_top100_chunk_ids=("q0_c1",),
            raw_top100_document_ids=("c",),
            document_top100=("c", "gold0"),
            first_gold_chunk_rank=90,
            first_gold_document_rank=10,
            crowding_gap=80,
            cutoff_observations=(
                make_observation(
                    20,
                    duplicate_ratio=0.10,
                    crowding_rescue=True,
                ),
                make_observation(
                    50,
                    duplicate_ratio=0.20,
                    crowding_rescue=False,
                ),
                make_observation(
                    100,
                    duplicate_ratio=0.60,
                    crowding_rescue=False,
                ),
            ),
        ),
    ]

    diagnostics = module.build_diagnostic_cases(
        results,
        limit_per_group=10,
    )

    assert list(diagnostics) == [
        "high_duplication_cases",
        "crowding_rescue_cases",
    ]

    assert [
        row["question_id"]
        for row in diagnostics["high_duplication_cases"]
    ] == [
        "TRAIN_Q0",
        "TRAIN_Q2",
        "TRAIN_Q1",
    ]

    assert [
        row["question_id"]
        for row in diagnostics["crowding_rescue_cases"]
    ] == [
        "TRAIN_Q0",
        "TRAIN_Q2",
        "TRAIN_Q1",
    ]

    assert all(
        row["crowding_rescue"] is True
        for row in diagnostics["crowding_rescue_cases"]
    )

    assert diagnostics["crowding_rescue_cases"][0]["cutoff"] == 20
    assert diagnostics["crowding_rescue_cases"][0]["crowding_gap"] == 80

def test_load_frozen_train_cases_rejects_dev_rows(
    tmp_path,
) -> None:
    module = _load_module()

    path = tmp_path / "train_results.jsonl"
    path.write_text(
        "\n".join(
            [
                (
                    '{"question_id":"TRAIN_Q0",'
                    '"question":"q",'
                    '"relevant_document_ids":["g"]}'
                ),
                (
                    '{"question_id":"DEV_Q0",'
                    '"question":"dev",'
                    '"relevant_document_ids":["g"]}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="TRAIN-only input",
    ):
        module.load_frozen_train_cases_from_e0(
            path,
            expected_count=None,
        )

def test_load_frozen_train_cases_requires_expected_query_count(
    tmp_path,
) -> None:
    module = _load_module()

    path = tmp_path / "train_results.jsonl"
    path.write_text(
        "".join(
            (
                f'{{"question_id":"TRAIN_Q{index:03d}",'
                f'"question":"q{index}",'
                '"relevant_document_ids":["gold"]}\n'
            )
            for index in range(449)
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="TRAIN query count mismatch",
    ):
        module.load_frozen_train_cases_from_e0(
            path,
            expected_count=450,
        )

def test_load_frozen_train_cases_rejects_duplicate_question_ids(
    tmp_path,
) -> None:
    module = _load_module()

    path = tmp_path / "train_results.jsonl"
    path.write_text(
        "\n".join(
            [
                (
                    '{"question_id":"TRAIN_Q001",'
                    '"question":"q1",'
                    '"relevant_document_ids":["gold1"]}'
                ),
                (
                    '{"question_id":"TRAIN_Q001",'
                    '"question":"q1 duplicate",'
                    '"relevant_document_ids":["gold1"]}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="duplicate TRAIN question_id",
    ):
        module.load_frozen_train_cases_from_e0(
            path,
            expected_count=None,
        )

@pytest.mark.parametrize(
    "relevant_document_ids",
    [
        [],
        ["gold1", "gold2"],
    ],
)
def test_load_frozen_train_cases_requires_exactly_one_relevant_document(
    tmp_path,
    relevant_document_ids,
) -> None:
    module = _load_module()

    path = tmp_path / "train_results.jsonl"
    path.write_text(
        (
            '{"question_id":"TRAIN_Q001",'
            '"question":"q",'
            f'"relevant_document_ids":{relevant_document_ids!r}'
            "}\n"
        ).replace("'", '"'),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="exactly one relevant document",
    ):
        module.load_frozen_train_cases_from_e0(
            path,
            expected_count=None,
        )

def test_run_preflight_records_historical_e0_sha_mismatch(
    tmp_path,
) -> None:
    module = _load_module()

    train_results_path = tmp_path / "train_results.jsonl"
    train_results_path.write_text(
        '{"question_id":"TRAIN_Q001"}\n',
        encoding="utf-8",
    )

    r3_manifest_path = tmp_path / "r3_manifest.json"
    r3_manifest_path.write_text(
        json.dumps(
            {
                "dense_source": {
                    "results_sha256": "wrong-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "frozen-corpus-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    report = module.run_preflight(
        train_results_path=train_results_path,
        r3_manifest_path=r3_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
        train_query_count=450,
        observed_chunk_count=172614,
        splitter_blob_loader=lambda: (
            "64026b4434f1eea46b95bfce9f667680a37a2103"
        ),
        version_loader=lambda package: "0.3.10",
    )

    historical_sha = json.loads(
        r3_manifest_path.read_text(
            encoding="utf-8",
        )
    )["dense_source"]["results_sha256"]

    assert report[
        "historical_e0_train_results_sha256"
    ] == historical_sha

    assert report[
        "input_e0_train_results_sha256"
    ] != historical_sha

    assert report[
        "e0_train_results_sha_matches_historical"
    ] is False

    manifest = module.build_run_manifest(report)

    assert manifest[
        "historical_e0_train_results_sha256"
    ] == historical_sha

    assert manifest[
        "input_e0_train_results_sha256"
    ] == report[
        "input_e0_train_results_sha256"
    ]

    assert manifest[
        "e0_train_results_sha_matches_historical"
    ] is False

def test_run_preflight_rejects_bm25_version_mismatch(
    tmp_path,
) -> None:
    module = _load_module()

    train_results_path = tmp_path / "train_results.jsonl"
    train_results_path.write_text(
        '{"question_id":"TRAIN_Q001"}\n',
        encoding="utf-8",
    )

    train_results_sha = hashlib.sha256(
        train_results_path.read_bytes()
    ).hexdigest()

    r3_manifest_path = tmp_path / "r3_manifest.json"
    r3_manifest_path.write_text(
        json.dumps(
            {
                "dense_source": {
                    "results_sha256": train_results_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "frozen-corpus-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="bm25s version mismatch",
    ):
        module.run_preflight(
            train_results_path=train_results_path,
            r3_manifest_path=r3_manifest_path,
            dataset_manifest_path=dataset_manifest_path,
            train_query_count=450,
            observed_chunk_count=172614,
            splitter_blob_loader=lambda: (
                "64026b4434f1eea46b95bfce9f667680a37a2103"
            ),
            version_loader=lambda package: "0.3.11",
        )

def test_run_preflight_rejects_splitter_blob_mismatch(
    tmp_path,
) -> None:
    module = _load_module()

    train_results_path = tmp_path / "train_results.jsonl"
    train_results_path.write_text(
        '{"question_id":"TRAIN_Q001"}\n',
        encoding="utf-8",
    )

    train_results_sha = hashlib.sha256(
        train_results_path.read_bytes()
    ).hexdigest()

    r3_manifest_path = tmp_path / "r3_manifest.json"
    r3_manifest_path.write_text(
        json.dumps(
            {
                "dense_source": {
                    "results_sha256": train_results_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "frozen-corpus-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="splitter blob mismatch",
    ):
        module.run_preflight(
            train_results_path=train_results_path,
            r3_manifest_path=r3_manifest_path,
            dataset_manifest_path=dataset_manifest_path,
            train_query_count=450,
            observed_chunk_count=172614,
            splitter_blob_loader=lambda: "wrong-splitter-blob",
            version_loader=lambda package: "0.3.10",
        )

def test_run_preflight_rejects_chunk_count_mismatch(
    tmp_path,
) -> None:
    module = _load_module()

    train_results_path = tmp_path / "train_results.jsonl"
    train_results_path.write_text(
        '{"question_id":"TRAIN_Q001"}\n',
        encoding="utf-8",
    )

    train_results_sha = hashlib.sha256(
        train_results_path.read_bytes()
    ).hexdigest()

    r3_manifest_path = tmp_path / "r3_manifest.json"
    r3_manifest_path.write_text(
        json.dumps(
            {
                "dense_source": {
                    "results_sha256": train_results_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "frozen-corpus-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="TechQA chunk count mismatch",
    ):
        module.run_preflight(
            train_results_path=train_results_path,
            r3_manifest_path=r3_manifest_path,
            dataset_manifest_path=dataset_manifest_path,
            train_query_count=450,
            observed_chunk_count=172613,
            splitter_blob_loader=lambda: (
                "64026b4434f1eea46b95bfce9f667680a37a2103"
            ),
            version_loader=lambda package: "0.3.10",
        )

def test_run_preflight_rejects_train_query_count_mismatch(
    tmp_path,
) -> None:
    module = _load_module()

    train_results_path = tmp_path / "train_results.jsonl"
    train_results_path.write_text(
        '{"question_id":"TRAIN_Q001"}\n',
        encoding="utf-8",
    )

    train_results_sha = hashlib.sha256(
        train_results_path.read_bytes()
    ).hexdigest()

    r3_manifest_path = tmp_path / "r3_manifest.json"
    r3_manifest_path.write_text(
        json.dumps(
            {
                "dense_source": {
                    "results_sha256": train_results_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "frozen-corpus-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="TRAIN query count mismatch",
    ):
        module.run_preflight(
            train_results_path=train_results_path,
            r3_manifest_path=r3_manifest_path,
            dataset_manifest_path=dataset_manifest_path,
            train_query_count=449,
            observed_chunk_count=172614,
            splitter_blob_loader=lambda: (
                "64026b4434f1eea46b95bfce9f667680a37a2103"
            ),
            version_loader=lambda package: "0.3.10",
        )

def test_run_preflight_returns_frozen_audit_evidence(
    tmp_path,
) -> None:
    module = _load_module()

    train_results_path = tmp_path / "train_results.jsonl"
    train_results_path.write_text(
        '{"question_id":"TRAIN_Q001"}\n',
        encoding="utf-8",
    )

    train_results_sha = hashlib.sha256(
        train_results_path.read_bytes()
    ).hexdigest()

    r3_manifest_path = tmp_path / "r3_manifest.json"
    r3_manifest_path.write_text(
        json.dumps(
            {
                "dense_source": {
                    "results_sha256": train_results_sha,
                }
            }
        ),
        encoding="utf-8",
    )

    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "retrieval_dataset": {
                    "repo": "bowang0911/TechQA-RAG-Eval",
                    "revision": "frozen-revision",
                    "corpus_sha256": "frozen-corpus-sha",
                    "queries_sha256": "frozen-queries-sha",
                    "qrels_sha256": "frozen-qrels-sha",
                }
            }
        ),
        encoding="utf-8",
    )

    report = module.run_preflight(
        train_results_path=train_results_path,
        r3_manifest_path=r3_manifest_path,
        dataset_manifest_path=dataset_manifest_path,
        train_query_count=450,
        observed_chunk_count=172614,
        splitter_blob_loader=lambda: (
            "64026b4434f1eea46b95bfce9f667680a37a2103"
        ),
        version_loader=lambda package: "0.3.10",
    )

    assert report["split"] == "train"
    assert report["query_count"] == 450
    assert report["chunk_count"] == 172614
    assert report["bm25_version"] == "0.3.10"
    assert report["splitter_blob_sha"] == (
        "64026b4434f1eea46b95bfce9f667680a37a2103"
    )
    assert report["provider_calls"] == 0
    assert report["dev_artifact_opened"] is False
    assert (
        report["input_e0_train_results_sha256"]
        == train_results_sha
    )
    assert report["retrieval_dataset"] == {
        "repo": "bowang0911/TechQA-RAG-Eval",
        "revision": "frozen-revision",
        "corpus_sha256": "frozen-corpus-sha",
        "queries_sha256": "frozen-queries-sha",
        "qrels_sha256": "frozen-qrels-sha",
    }

def test_build_run_manifest_freezes_chunk_bm25_audit_contract() -> None:
    module = _load_module()

    preflight_report = {
        "split": "train",
        "query_count": 450,
        "chunk_count": 172614,
        "bm25_version": "0.3.10",
        "splitter_blob_sha": (
            "64026b4434f1eea46b95bfce9f667680a37a2103"
        ),
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "input_e0_train_results_sha256": "frozen-e0-train-sha",
        "retrieval_dataset": {
            "repo": "bowang0911/TechQA-RAG-Eval",
            "revision": "frozen-revision",
            "corpus_sha256": "frozen-corpus-sha",
            "queries_sha256": "frozen-queries-sha",
            "qrels_sha256": "frozen-qrels-sha",
        },
    }

    manifest = module.build_run_manifest(preflight_report)

    assert manifest["benchmark"] == "TechQA-RAG-Eval"
    assert manifest["run"] == "r4_chunk_bm25_candidate_audit"
    assert manifest["split"] == "train"
    assert manifest["query_count"] == 450
    assert manifest["provider_calls"] == 0
    assert manifest["dev_artifact_opened"] is False

    assert manifest["chunking"] == {
        "strategy": "paragraph_aware_character",
        "chunk_size": 800,
        "chunk_overlap": 120,
        "min_chunk_size": 150,
        "splitter_blob_sha": (
            "64026b4434f1eea46b95bfce9f667680a37a2103"
        ),
        "observed_chunk_count": 172614,
    }

    assert manifest["bm25"] == {
        "library": "bm25s",
        "version": "0.3.10",
        "method": "lucene",
        "k1": 1.5,
        "b": 0.75,
        "backend": "numpy",
        "indexed_unit": "chunk",
        "tokenizer_regex": (
            r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*"
        ),
        "query_normalization": "rstrip",
    }

    assert manifest["audit"] == {
        "raw_chunk_cutoffs": [20, 50, 100],
        "initial_search_depth": 500,
        "required_unique_documents": 100,
        "depth_growth_factor": 2,
        "max_search_depth": 172614,
    }

    assert manifest["retrieval_dataset"] == (
        preflight_report["retrieval_dataset"]
    )
    assert manifest["input_e0_train_results_sha256"] == (
        "frozen-e0-train-sha"
    )

    assert "chunk_corpus_sha256" not in manifest
    assert "rerank_model" not in manifest
    assert "rrf" not in manifest

def test_run_audit_cases_builds_retriever_once_for_all_queries() -> None:
    module = _load_module()

    chunks = [
        module.TechQAChunkRef(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            chunk_index=index,
        )
        for index in range(100)
    ]

    cases = [
        module.TechQARetrievalCase(
            question_id="TRAIN_Q001",
            question="first query",
            relevant_document_ids=("doc-0",),
            split="train",
        ),
        module.TechQARetrievalCase(
            question_id="TRAIN_Q002",
            question="second query",
            relevant_document_ids=("doc-0",),
            split="train",
        ),
    ]

    build_count = 0

    class FakeRetriever:
        def search(
            self,
            query: str,
            *,
            top_k: int = 100,
        ):
            del query
            return chunks[:top_k]

    def retriever_factory(input_chunks):
        nonlocal build_count
        build_count += 1
        assert input_chunks is chunks
        return FakeRetriever()

    results, index_build_seconds = module.run_audit_cases(
        cases=cases,
        chunks=chunks,
        retriever_factory=retriever_factory,
    )

    assert build_count == 1
    assert len(results) == 2
    assert [
        result.question_id
        for result in results
    ] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]
    assert index_build_seconds >= 0.0

def test_write_audit_artifacts_writes_complete_report_set(
    tmp_path,
) -> None:
    module = _load_module()

    result = module.TechQAChunkBM25AuditResult(
        question_id="TRAIN_Q001",
        question="test query",
        relevant_document_ids=("gold-doc",),
        audit_search_depth=500,
        latency_ms=12.5,
        raw_top100_chunk_ids=["chunk-1"],
        raw_top100_document_ids=["gold-doc"],
        document_top100=["gold-doc"],
        first_gold_chunk_rank=1,
        first_gold_document_rank=1,
        crowding_gap=0,
        cutoff_observations=[
            module.ChunkCutoffObservation(
                cutoff=20,
                returned_chunk_count=1,
                unique_document_count=1,
                duplicate_slot_count=0,
                duplicate_ratio=0.0,
                gold_document_hit_within_chunk_k=True,
                crowding_rescue=False,
            )
        ],
    )

    manifest = {
        "run": "r4_chunk_bm25_candidate_audit",
        "split": "train",
        "provider_calls": 0,
        "dev_artifact_opened": False,
    }
    metrics = {
        "query_count": 1,
        "index_build_seconds": 1.25,
    }
    diagnostics = {
        "high_duplication_cases": [],
        "crowding_rescue_cases": [],
    }

    module.write_audit_artifacts(
        output_dir=tmp_path,
        manifest=manifest,
        metrics=metrics,
        results=[result],
        diagnostics=diagnostics,
    )

    manifest_path = tmp_path / "train_manifest.json"
    metrics_path = tmp_path / "train_metrics.json"
    results_path = tmp_path / "train_results.jsonl"
    diagnostics_path = tmp_path / "diagnostic_cases.json"

    assert manifest_path.exists()
    assert metrics_path.exists()
    assert results_path.exists()
    assert diagnostics_path.exists()

    assert json.loads(
        manifest_path.read_text(encoding="utf-8")
    ) == manifest

    assert json.loads(
        metrics_path.read_text(encoding="utf-8")
    ) == metrics

    result_payload = json.loads(
        results_path.read_text(
            encoding="utf-8",
        ).strip()
    )
    assert result_payload["question_id"] == "TRAIN_Q001"
    assert result_payload["relevant_document_ids"] == [
        "gold-doc"
    ]
    assert result_payload["raw_top100_chunk_ids"] == [
        "chunk-1"
    ]

    assert json.loads(
        diagnostics_path.read_text(encoding="utf-8")
    ) == diagnostics

def test_execute_audit_writes_zero_provider_report(
    tmp_path,
) -> None:
    module = _load_module()

    chunks = [
        module.TechQAChunkRef(
            chunk_id=f"chunk-{index}",
            document_id=f"doc-{index}",
            chunk_index=index,
        )
        for index in range(100)
    ]

    case = module.TechQARetrievalCase(
        question_id="TRAIN_Q001",
        question="test query",
        relevant_document_ids=("doc-0",),
        split="train",
    )

    class FakeRetriever:
        def search(
            self,
            query: str,
            *,
            top_k: int = 100,
        ):
            del query
            return chunks[:top_k]

    preflight_report = {
        "split": "train",
        "query_count": 1,
        "chunk_count": 100,
        "bm25_version": "0.3.10",
        "splitter_blob_sha": (
            "64026b4434f1eea46b95bfce9f667680a37a2103"
        ),
        "provider_calls": 0,
        "dev_artifact_opened": False,
        "input_e0_train_results_sha256": "test-sha",
        "retrieval_dataset": {
            "repo": "test-repo",
            "revision": "test-revision",
            "corpus_sha256": "corpus-sha",
            "queries_sha256": "queries-sha",
            "qrels_sha256": "qrels-sha",
        },
    }

    metrics = module.execute_audit(
        cases=[case],
        chunks=chunks,
        preflight_report=preflight_report,
        output_dir=tmp_path,
        retriever_factory=lambda input_chunks: FakeRetriever(),
    )

    assert metrics["query_count"] == 1
    assert metrics["index_build_seconds"] >= 0.0

    manifest = json.loads(
        (tmp_path / "train_manifest.json").read_text(
            encoding="utf-8",
        )
    )

    assert manifest["provider_calls"] == 0
    assert manifest["dev_artifact_opened"] is False

    assert {
        path.name
        for path in tmp_path.iterdir()
    } == {
        "train_manifest.json",
        "train_metrics.json",
        "train_results.jsonl",
        "diagnostic_cases.json",
    }