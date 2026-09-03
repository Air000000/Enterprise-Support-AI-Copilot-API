import json

import pytest

import experiments.evals.eval_techqa_generation as generation_eval
from experiments.evals.adapters.techqa import TechQAGenerationCase
from experiments.evals.eval_techqa_generation import (
    TechQAGenerationEvalResult,
    load_generation_checkpoint,
    run_resumable_generation_eval,
    write_generation_reports,
)
from experiments.evals.rerankers.qwen3_reranker import (
    RerankCandidate,
    RerankResult,
    RerankedCandidate,
)


def _candidate_document_selector():
    selector = getattr(
        generation_eval,
        "select_candidate_document_ids",
        None,
    )
    assert selector is not None, "select_candidate_document_ids is not implemented"
    return selector


def _document_local_pool_builder():
    builder = getattr(
        generation_eval,
        "build_document_local_candidate_pool",
        None,
    )
    assert builder is not None, "build_document_local_candidate_pool is not implemented"
    return builder


def _document_local_evidence_selector():
    selector = getattr(
        generation_eval,
        "select_document_local_evidence",
        None,
    )
    assert selector is not None, "select_document_local_evidence is not implemented"
    return selector


def _selected_evidence_context_assembler():
    assembler = getattr(
        generation_eval,
        "assemble_selected_evidence_context",
        None,
    )
    assert assembler is not None, "assemble_selected_evidence_context is not implemented"
    return assembler


def _document_local_diagnostic_runner():
    runner = getattr(
        generation_eval,
        "run_document_local_evidence_diagnostic",
        None,
    )
    assert runner is not None, "run_document_local_evidence_diagnostic is not implemented"
    return runner


def _retrieved_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    distance: float,
) -> generation_eval.G0RetrievedChunk:
    return generation_eval.G0RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=f"content {chunk_id}",
        distance=distance,
    )


def _case(question_id: str, *, answerable: bool = True) -> TechQAGenerationCase:
    return TechQAGenerationCase(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer" if answerable else "",
        answerable=answerable,
        split="train",
    )


def _result(
    question_id: str,
    *,
    answerable: bool = True,
    latency_ms: float = 100.0,
) -> TechQAGenerationEvalResult:
    return TechQAGenerationEvalResult(
        question_id=question_id,
        question=f"question {question_id}",
        gold_answer="gold answer" if answerable else "",
        answerable=answerable,
        retrieved_chunk_ids=("doc_chunk_0",),
        retrieved_document_ids=("doc",),
        retrieval_context=("retrieved evidence",),
        generated_answer=(
            "generated answer"
            if answerable
            else "我在已提供资料中没有找到足够依据。"
        ),
        retrieval_status="ok" if answerable else "refused_low_relevance",
        top_distance=0.2 if answerable else 1.1,
        abstained=not answerable,
        hallucinated=False,
        correctness_score=0.8 if answerable else None,
        correctness_reason="correct" if answerable else None,
        faithfulness_score=0.9 if answerable else None,
        faithfulness_reason="grounded" if answerable else None,
        e2e_latency_ms=latency_ms,
    )


def test_resumable_runner_skips_checkpointed_cases_and_appends_only_missing(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    existing = _result("TRAIN_Q001", latency_ms=100.0)
    checkpoint_path.write_text(
        json.dumps(existing.__dict__, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    evaluated: list[str] = []

    def fake_evaluator(case: TechQAGenerationCase) -> TechQAGenerationEvalResult:
        evaluated.append(case.question_id)
        return _result(case.question_id, latency_ms=200.0)

    summary = run_resumable_generation_eval(
        [_case("TRAIN_Q001"), _case("TRAIN_Q002")],
        evaluator=fake_evaluator,
        checkpoint_path=checkpoint_path,
        split="train",
    )

    assert evaluated == ["TRAIN_Q002"]
    assert summary.query_count == 2
    assert [result.question_id for result in summary.results] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]

    persisted = load_generation_checkpoint(checkpoint_path)
    assert [result.question_id for result in persisted] == [
        "TRAIN_Q001",
        "TRAIN_Q002",
    ]


def test_resumable_runner_persists_each_result_before_later_failure(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    calls: list[str] = []

    def flaky_evaluator(case: TechQAGenerationCase) -> TechQAGenerationEvalResult:
        calls.append(case.question_id)
        if case.question_id == "TRAIN_Q002":
            raise RuntimeError("simulated provider failure")
        return _result(case.question_id)

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        run_resumable_generation_eval(
            [_case("TRAIN_Q001"), _case("TRAIN_Q002")],
            evaluator=flaky_evaluator,
            checkpoint_path=checkpoint_path,
            split="train",
        )

    assert calls == ["TRAIN_Q001", "TRAIN_Q002"]
    persisted = load_generation_checkpoint(checkpoint_path)
    assert [result.question_id for result in persisted] == ["TRAIN_Q001"]


def test_completed_checkpoint_rebuilds_summary_and_reports_without_evaluator(tmp_path):
    checkpoint_path = tmp_path / "train_checkpoint.jsonl"
    results = [
        _result("TRAIN_Q001", latency_ms=100.0),
        _result("TRAIN_I001", answerable=False, latency_ms=300.0),
    ]
    checkpoint_path.write_text(
        "".join(
            json.dumps(result.__dict__, ensure_ascii=False) + "\n"
            for result in results
        ),
        encoding="utf-8",
    )

    def forbidden_evaluator(case: TechQAGenerationCase) -> TechQAGenerationEvalResult:
        raise AssertionError("completed checkpoint must not call evaluator")

    summary = run_resumable_generation_eval(
        [_case("TRAIN_Q001"), _case("TRAIN_I001", answerable=False)],
        evaluator=forbidden_evaluator,
        checkpoint_path=checkpoint_path,
        split="train",
    )

    assert summary.query_count == 2
    assert summary.answerable_count == 1
    assert summary.impossible_count == 1
    assert summary.correctness_mean == pytest.approx(0.8)
    assert summary.faithfulness_mean == pytest.approx(0.9)
    assert summary.abstention_accuracy == pytest.approx(1.0)
    assert summary.hallucination_rate == pytest.approx(0.0)
    assert summary.e2e_latency_p50_ms == pytest.approx(200.0)
    assert summary.e2e_latency_p95_ms == pytest.approx(290.0)

    report_dir = tmp_path / "reports"
    write_generation_reports(summary, report_dir=report_dir)

    metrics = json.loads(
        (report_dir / "train_generation_metrics.json").read_text(encoding="utf-8")
    )
    result_lines = (report_dir / "train_generation_results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()

    assert metrics["query_count"] == 2
    assert metrics["correctness_mean"] == pytest.approx(0.8)
    assert metrics["faithfulness_mean"] == pytest.approx(0.9)
    assert len(result_lines) == 2


def test_candidate_document_selection_deduplicates_repeated_document_slots():
    selector = _candidate_document_selector()
    ranked = (
        _retrieved_chunk("A_chunk_3", "A", 3, 0.10),
        _retrieved_chunk("A_chunk_5", "A", 5, 0.12),
        _retrieved_chunk("B_chunk_2", "B", 2, 0.15),
        _retrieved_chunk("A_chunk_1", "A", 1, 0.18),
        _retrieved_chunk("C_chunk_4", "C", 4, 0.20),
        _retrieved_chunk("D_chunk_2", "D", 2, 0.25),
    )

    assert selector(ranked, limit=3) == ("A", "B", "C")


@pytest.mark.parametrize("limit", [0, -1])
def test_candidate_document_selection_requires_positive_limit(limit):
    selector = _candidate_document_selector()

    with pytest.raises(ValueError):
        selector((), limit=limit)


def test_document_local_candidate_pool_preserves_document_and_loader_order():
    builder = _document_local_pool_builder()
    chunks_by_document = {
        "B": (
            _retrieved_chunk("B_chunk_4", "B", 4, 0.20),
            _retrieved_chunk("B_chunk_1", "B", 1, 0.22),
            _retrieved_chunk("B_chunk_3", "B", 3, 0.25),
        ),
        "A": (
            _retrieved_chunk("A_chunk_5", "A", 5, 0.10),
            _retrieved_chunk("A_chunk_2", "A", 2, 0.12),
        ),
    }

    pool = builder(
        ("B", "A"),
        load_document_chunks=chunks_by_document.__getitem__,
        max_chunks=5,
    )

    assert tuple(chunk.chunk_id for chunk in pool) == (
        "B_chunk_4",
        "B_chunk_1",
        "B_chunk_3",
        "A_chunk_5",
        "A_chunk_2",
    )


def test_document_local_candidate_pool_deduplicates_chunk_ids_globally():
    builder = _document_local_pool_builder()
    chunks_by_document = {
        "A": (
            _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
            _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
            _retrieved_chunk("A_chunk_2", "A", 2, 0.12),
        ),
        "B": (_retrieved_chunk("B_chunk_1", "B", 1, 0.20),),
    }

    pool = builder(
        ("A", "B"),
        load_document_chunks=chunks_by_document.__getitem__,
        max_chunks=4,
    )

    assert tuple(chunk.chunk_id for chunk in pool) == (
        "A_chunk_1",
        "A_chunk_2",
        "B_chunk_1",
    )


def test_document_local_candidate_pool_stops_at_total_budget():
    builder = _document_local_pool_builder()
    calls: list[str] = []
    chunks_by_document = {
        "A": (
            _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
            _retrieved_chunk("A_chunk_2", "A", 2, 0.12),
        ),
        "B": (
            _retrieved_chunk("B_chunk_1", "B", 1, 0.20),
            _retrieved_chunk("B_chunk_2", "B", 2, 0.25),
        ),
        "C": (_retrieved_chunk("C_chunk_1", "C", 1, 0.30),),
    }

    def load_document_chunks(document_id: str):
        calls.append(document_id)
        return chunks_by_document[document_id]

    pool = builder(
        ("A", "B", "C"),
        load_document_chunks=load_document_chunks,
        max_chunks=3,
    )

    assert tuple(chunk.chunk_id for chunk in pool) == (
        "A_chunk_1",
        "A_chunk_2",
        "B_chunk_1",
    )
    assert calls == ["A", "B"]


def test_document_local_candidate_pool_rejects_wrong_document_membership():
    builder = _document_local_pool_builder()
    chunks_by_document = {
        "A": (_retrieved_chunk("shared_chunk", "A", 1, 0.10),),
        "B": (_retrieved_chunk("shared_chunk", "A", 1, 0.10),),
    }

    with pytest.raises(RuntimeError, match="document_id"):
        builder(
            ("A", "B"),
            load_document_chunks=chunks_by_document.__getitem__,
            max_chunks=2,
        )


@pytest.mark.parametrize("max_chunks", [0, -1])
def test_document_local_candidate_pool_requires_positive_budget(max_chunks):
    builder = _document_local_pool_builder()

    with pytest.raises(ValueError):
        builder(
            (),
            load_document_chunks=lambda _: (),
            max_chunks=max_chunks,
        )


def test_evidence_selection_uses_relevance_order_over_pool_position():
    selector = _document_local_evidence_selector()
    pool = (
        _retrieved_chunk("A_chunk_1", "A", 1, 0.40),
        _retrieved_chunk("A_chunk_5", "A", 5, 0.20),
        _retrieved_chunk("A_chunk_9", "A", 9, 0.30),
        _retrieved_chunk("B_chunk_2", "B", 2, 0.25),
    )
    calls = []

    def reranker(question, candidates):
        calls.append((question, candidates))
        return RerankResult(
            results=(
                RerankedCandidate("A_chunk_1", "A", "", 0, 0.99),
                RerankedCandidate("B_chunk_2", "B", "", 3, 0.98),
                RerankedCandidate("A_chunk_5", "A", "", 1, 0.97),
                RerankedCandidate("A_chunk_9", "A", "", 2, 0.96),
            ),
            request_id=None,
            total_tokens=None,
        )

    selected = selector("Which evidence matters?   ", pool, reranker=reranker, limit=2)

    assert tuple(chunk.chunk_id for chunk in selected) == ("A_chunk_1", "B_chunk_2")
    assert len(calls) == 1
    assert calls[0][0] == "Which evidence matters?"
    assert calls[0][1] == (
        RerankCandidate("A_chunk_1", "A", "content A_chunk_1"),
        RerankCandidate("A_chunk_5", "A", "content A_chunk_5"),
        RerankCandidate("A_chunk_9", "A", "content A_chunk_9"),
        RerankCandidate("B_chunk_2", "B", "content B_chunk_2"),
    )


def test_evidence_selection_reranks_one_merged_document_pool():
    selector = _document_local_evidence_selector()
    pool = (
        _retrieved_chunk("A_chunk_1", "A", 1, 0.10),
        _retrieved_chunk("B_chunk_1", "B", 1, 0.20),
    )
    calls = []

    def reranker(question, candidates):
        calls.append((question, candidates))
        return RerankResult(
            results=tuple(
                RerankedCandidate(
                    candidate.chunk_id,
                    candidate.document_id,
                    candidate.content,
                    index,
                    1.0 - index,
                )
                for index, candidate in enumerate(candidates)
            ),
            request_id=None,
            total_tokens=None,
        )

    selector("question", pool, reranker=reranker, limit=2)

    assert len(calls) == 1
    assert tuple(candidate.chunk_id for candidate in calls[0][1]) == (
        "A_chunk_1",
        "B_chunk_1",
    )


def test_evidence_selection_returns_original_chunk_provenance():
    selector = _document_local_evidence_selector()
    source = generation_eval.G0RetrievedChunk(
        chunk_id="A_chunk_1",
        document_id="A",
        chunk_index=1,
        content="decisive evidence",
        distance=0.417,
    )

    def reranker(question, candidates):
        return RerankResult(
            results=(
                RerankedCandidate("A_chunk_1", "other", "rewritten", 99, 0.981),
            ),
            request_id=None,
            total_tokens=None,
        )

    selected = selector("question", (source,), reranker=reranker, limit=1)

    assert selected == (source,)
    assert selected[0].distance == pytest.approx(0.417)
    assert selected[0].distance != pytest.approx(0.981)


def test_evidence_selection_rejects_duplicate_candidate_chunk_id():
    selector = _document_local_evidence_selector()
    pool = (
        _retrieved_chunk("shared_chunk", "A", 1, 0.10),
        _retrieved_chunk("shared_chunk", "B", 2, 0.20),
    )

    def forbidden_reranker(*args):
        raise AssertionError("duplicate candidate identity must fail before reranking")

    with pytest.raises(RuntimeError, match="duplicate candidate"):
        selector("question", pool, reranker=forbidden_reranker, limit=1)


def test_evidence_selection_rejects_unknown_reranked_chunk_id():
    selector = _document_local_evidence_selector()

    def reranker(question, candidates):
        return RerankResult(
            results=(RerankedCandidate("unknown", "A", "", 0, 1.0),),
            request_id=None,
            total_tokens=None,
        )

    with pytest.raises(RuntimeError, match="candidate pool"):
        selector(
            "question",
            (_retrieved_chunk("A_chunk_1", "A", 1, 0.10),),
            reranker=reranker,
            limit=1,
        )


def test_evidence_selection_rejects_duplicate_reranked_chunk_id():
    selector = _document_local_evidence_selector()

    def reranker(question, candidates):
        return RerankResult(
            results=(
                RerankedCandidate("A_chunk_1", "A", "", 0, 1.0),
                RerankedCandidate("A_chunk_1", "A", "", 0, 0.9),
            ),
            request_id=None,
            total_tokens=None,
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        selector(
            "question",
            (_retrieved_chunk("A_chunk_1", "A", 1, 0.10),),
            reranker=reranker,
            limit=2,
        )


@pytest.mark.parametrize("limit", [0, -1])
def test_evidence_selection_requires_positive_limit(limit):
    selector = _document_local_evidence_selector()

    with pytest.raises(ValueError):
        selector("question", (), reranker=lambda *_: None, limit=limit)


def test_evidence_selection_empty_pool_avoids_reranker():
    selector = _document_local_evidence_selector()

    def forbidden_reranker(question, candidates):
        raise AssertionError("empty pool must not call reranker")

    assert selector("question", (), reranker=forbidden_reranker, limit=1) == ()


def test_selected_evidence_context_preserves_relevance_order():
    assembler = _selected_evidence_context_assembler()
    selected_evidence = (
        _retrieved_chunk("A_chunk_5", "A", 5, 0.20),
        _retrieved_chunk("A_chunk_1", "A", 1, 0.40),
        _retrieved_chunk("B_chunk_2", "B", 2, 0.25),
    )

    context = assembler(selected_evidence, max_context_chunks=3)

    assert tuple(chunk.chunk_id for chunk in context) == (
        "A_chunk_5",
        "A_chunk_1",
        "B_chunk_2",
    )


def test_selected_evidence_context_deduplicates_chunk_ids():
    assembler = _selected_evidence_context_assembler()
    a1 = _retrieved_chunk("A_chunk_1", "A", 1, 0.10)

    context = assembler(
        (a1, a1, _retrieved_chunk("B_chunk_2", "B", 2, 0.20)),
        max_context_chunks=3,
    )

    assert tuple(chunk.chunk_id for chunk in context) == ("A_chunk_1", "B_chunk_2")


def test_selected_evidence_context_budget_counts_unique_chunks():
    assembler = _selected_evidence_context_assembler()
    a1 = _retrieved_chunk("A_chunk_1", "A", 1, 0.10)

    context = assembler(
        (
            a1,
            a1,
            _retrieved_chunk("B_chunk_2", "B", 2, 0.20),
            _retrieved_chunk("C_chunk_3", "C", 3, 0.30),
        ),
        max_context_chunks=2,
    )

    assert tuple(chunk.chunk_id for chunk in context) == ("A_chunk_1", "B_chunk_2")


@pytest.mark.parametrize("max_context_chunks", [0, -1])
def test_selected_evidence_context_requires_positive_budget(max_context_chunks):
    assembler = _selected_evidence_context_assembler()

    with pytest.raises(ValueError):
        assembler((), max_context_chunks=max_context_chunks)


def test_document_local_diagnostic_records_full_four_stage_provenance():
    runner = _document_local_diagnostic_runner()
    ranked_chunks = (
        _retrieved_chunk("A_chunk_5", "A", 5, 0.10),
        _retrieved_chunk("A_chunk_3", "A", 3, 0.12),
        _retrieved_chunk("B_chunk_2", "B", 2, 0.20),
        _retrieved_chunk("C_chunk_1", "C", 1, 0.30),
    )
    chunks_by_document = {
        "A": (
            _retrieved_chunk("A_chunk_1", "A", 1, 0.11),
            _retrieved_chunk("A_chunk_5", "A", 5, 0.10),
            _retrieved_chunk("A_chunk_9", "A", 9, 0.13),
        ),
        "B": (
            _retrieved_chunk("B_chunk_2", "B", 2, 0.20),
            _retrieved_chunk("B_chunk_4", "B", 4, 0.21),
        ),
    }

    def reranker(question, candidates):
        return RerankResult(
            results=(
                RerankedCandidate("B_chunk_4", "B", "", 4, 0.99),
                RerankedCandidate("A_chunk_1", "A", "", 0, 0.98),
                RerankedCandidate("A_chunk_5", "A", "", 1, 0.97),
                RerankedCandidate("B_chunk_2", "B", "", 3, 0.96),
                RerankedCandidate("A_chunk_9", "A", "", 2, 0.95),
            ),
            request_id=None,
            total_tokens=None,
        )

    diagnostic = runner(
        "Q001",
        "diagnostic question",
        ranked_chunks,
        candidate_document_limit=2,
        load_document_chunks=chunks_by_document.__getitem__,
        candidate_pool_max_chunks=5,
        reranker=reranker,
        evidence_limit=3,
        max_context_chunks=2,
    )

    assert diagnostic.question_id == "Q001"
    assert diagnostic.question == "diagnostic question"
    assert tuple(chunk.chunk_id for chunk in diagnostic.ranked_chunks) == (
        "A_chunk_5",
        "A_chunk_3",
        "B_chunk_2",
        "C_chunk_1",
    )
    assert diagnostic.candidate_document_ids == ("A", "B")
    assert tuple(chunk.chunk_id for chunk in diagnostic.candidate_pool) == (
        "A_chunk_1",
        "A_chunk_5",
        "A_chunk_9",
        "B_chunk_2",
        "B_chunk_4",
    )
    assert tuple(chunk.chunk_id for chunk in diagnostic.selected_evidence) == (
        "B_chunk_4",
        "A_chunk_1",
        "A_chunk_5",
    )
    assert tuple(chunk.chunk_id for chunk in diagnostic.final_context) == (
        "B_chunk_4",
        "A_chunk_1",
    )
    assert diagnostic.final_context[0].document_id == "B"
    assert diagnostic.final_context[0].chunk_index == 4
    assert diagnostic.final_context[0].distance == pytest.approx(0.21)


def test_document_local_diagnostic_records_stage_invocations():
    runner = _document_local_diagnostic_runner()
    loader_calls = []
    reranker_calls = []
    chunks_by_document = {
        "A": (_retrieved_chunk("A_chunk_1", "A", 1, 0.10),),
        "B": (_retrieved_chunk("B_chunk_1", "B", 1, 0.20),),
    }

    def load_document_chunks(document_id):
        loader_calls.append(document_id)
        return chunks_by_document[document_id]

    def reranker(question, candidates):
        reranker_calls.append(candidates)
        return RerankResult(
            results=tuple(
                RerankedCandidate(
                    candidate.chunk_id,
                    candidate.document_id,
                    candidate.content,
                    index,
                    1.0 - index,
                )
                for index, candidate in enumerate(candidates)
            ),
            request_id=None,
            total_tokens=None,
        )

    diagnostic = runner(
        "Q002",
        "stage count question",
        (
            _retrieved_chunk("A_ranked", "A", 1, 0.10),
            _retrieved_chunk("B_ranked", "B", 1, 0.20),
        ),
        candidate_document_limit=2,
        load_document_chunks=load_document_chunks,
        candidate_pool_max_chunks=2,
        reranker=reranker,
        evidence_limit=2,
        max_context_chunks=2,
    )

    assert loader_calls == ["A", "B"]
    assert len(reranker_calls) == 1
    assert tuple(candidate.chunk_id for candidate in reranker_calls[0]) == (
        "A_chunk_1",
        "B_chunk_1",
    )
    assert diagnostic.stage_invocations.dense_search == 0
    assert diagnostic.stage_invocations.document_chunk_loader == len(loader_calls)
    assert diagnostic.stage_invocations.evidence_reranker == 1
    assert diagnostic.stage_invocations.generation == 0
    assert diagnostic.stage_invocations.judge == 0


def test_document_local_diagnostic_empty_ranked_input_has_zero_downstream_calls():
    runner = _document_local_diagnostic_runner()

    def forbidden_loader(*args):
        raise AssertionError("empty ranked input must not load document chunks")

    def forbidden_reranker(*args):
        raise AssertionError("empty ranked input must not rerank evidence")

    diagnostic = runner(
        "Q003",
        "empty question",
        (),
        candidate_document_limit=1,
        load_document_chunks=forbidden_loader,
        candidate_pool_max_chunks=1,
        reranker=forbidden_reranker,
        evidence_limit=1,
        max_context_chunks=1,
    )

    assert diagnostic.candidate_document_ids == ()
    assert diagnostic.candidate_pool == ()
    assert diagnostic.selected_evidence == ()
    assert diagnostic.final_context == ()
    assert diagnostic.stage_invocations.dense_search == 0
    assert diagnostic.stage_invocations.document_chunk_loader == 0
    assert diagnostic.stage_invocations.evidence_reranker == 0
    assert diagnostic.stage_invocations.generation == 0
    assert diagnostic.stage_invocations.judge == 0


@pytest.mark.parametrize(
    "invalid_budget",
    (
        "candidate_document_limit",
        "candidate_pool_max_chunks",
        "evidence_limit",
        "max_context_chunks",
    ),
)
def test_document_local_diagnostic_validates_all_budgets_before_side_effects(
    invalid_budget,
):
    runner = _document_local_diagnostic_runner()
    calls = []
    budgets = {
        "candidate_document_limit": 1,
        "candidate_pool_max_chunks": 1,
        "evidence_limit": 1,
        "max_context_chunks": 1,
    }
    budgets[invalid_budget] = 0

    def forbidden_loader(*args):
        calls.append("loader")
        raise AssertionError("invalid budgets must fail before loading")

    def forbidden_reranker(*args):
        calls.append("reranker")
        raise AssertionError("invalid budgets must fail before reranking")

    with pytest.raises(ValueError):
        runner(
            "Q004",
            "invalid budget question",
            (_retrieved_chunk("A_chunk_1", "A", 1, 0.10),),
            load_document_chunks=forbidden_loader,
            reranker=forbidden_reranker,
            **budgets,
        )

    assert calls == []


def test_default_single_case_evaluator_uses_frozen_g0_retriever(monkeypatch):
    from types import SimpleNamespace

    import experiments.evals.eval_techqa_generation as generation_eval

    case = _case("TRAIN_Q001")
    expected_result = _result("TRAIN_Q001")

    observed: dict[str, object] = {}

    def fake_evaluate(cases, **kwargs):
        observed["cases"] = list(cases)
        observed["split"] = kwargs.get("split")
        observed["retriever"] = kwargs.get("retriever")

        return SimpleNamespace(
            results=(expected_result,)
        )

    monkeypatch.setattr(
        generation_eval,
        "evaluate_techqa_generation_cases",
        fake_evaluate,
    )

    actual = generation_eval._evaluate_single_generation_case(
        case
    )

    assert actual is expected_result
    assert observed["cases"] == [case]
    assert observed["split"] == "train"

    assert (
        observed["retriever"]
        is generation_eval.retrieve_g0_e1_context
    )


def test_g0_train_pilot_selection_is_deterministic_and_balanced():
    import hashlib

    from experiments.evals import eval_techqa_generation as generation_eval

    train_answerable = [
        TechQAGenerationCase(
            question_id=f"TRAIN_Q{index:03d}",
            question=f"answerable question {index}",
            gold_answer=f"answer {index}",
            answerable=True,
            split="train",
        )
        for index in range(12)
    ]

    train_impossible = [
        TechQAGenerationCase(
            question_id=f"TRAIN_I{index:03d}",
            question=f"impossible question {index}",
            gold_answer="",
            answerable=False,
            split="train",
        )
        for index in range(12)
    ]

    dev_noise = [
        TechQAGenerationCase(
            question_id=f"DEV_Q{index:03d}",
            question=f"dev question {index}",
            gold_answer=f"dev answer {index}",
            answerable=True,
            split="dev",
        )
        for index in range(4)
    ]

    all_cases = (
        train_answerable
        + train_impossible
        + dev_noise
    )

    selected = generation_eval.select_g0_train_pilot_cases(
        all_cases
    )

    selected_from_reversed = (
        generation_eval.select_g0_train_pilot_cases(
            list(reversed(all_cases))
        )
    )

    assert generation_eval.DEFAULT_G0_PILOT_SIZE == 12
    assert generation_eval.DEFAULT_G0_PILOT_PER_CLASS == 6
    assert (
        generation_eval.DEFAULT_G0_PILOT_SEED
        == "techqa-g0-generation-pilot-v1"
    )

    assert len(selected) == 12

    assert sum(
        case.answerable
        for case in selected
    ) == 6

    assert sum(
        not case.answerable
        for case in selected
    ) == 6

    assert all(
        case.split == "train"
        for case in selected
    )

    selected_ids = [
        case.question_id
        for case in selected
    ]

    reversed_ids = [
        case.question_id
        for case in selected_from_reversed
    ]

    assert selected_ids == reversed_ids

    def stable_key(case):
        return hashlib.sha256(
            (
                f"{generation_eval.DEFAULT_G0_PILOT_SEED}:"
                f"{case.question_id}"
            ).encode("utf-8")
        ).hexdigest()

    expected_answerable = sorted(
        train_answerable,
        key=stable_key,
    )[:6]

    expected_impossible = sorted(
        train_impossible,
        key=stable_key,
    )[:6]

    expected_ids = {
        case.question_id
        for case in (
            expected_answerable
            + expected_impossible
        )
    }

    assert set(selected_ids) == expected_ids
