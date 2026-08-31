import pytest

from experiments.evals.adapters.techqa import TechQAGenerationCase


def _case(
    question_id: str,
    *,
    answerable: bool = True,
    split: str = "train",
) -> TechQAGenerationCase:
    return TechQAGenerationCase(
        question_id=question_id,
        question=f"question for {question_id}",
        gold_answer=f"answer for {question_id}" if answerable else "",
        answerable=answerable,
        split=split,
    )


def test_representative_sample_is_deterministic_and_train_answerable_only():
    try:
        from experiments.evals.techqa_evidence_audit import (
            DEFAULT_EVIDENCE_SAMPLE_SEED,
            select_representative_train_cases,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.fail("TechQA evidence audit sampling is not implemented yet")

    train_answerable = [
        _case(f"TRAIN_Q{index:03d}")
        for index in range(450)
    ]
    distractors = [
        _case("TRAIN_IMPOSSIBLE", answerable=False),
        _case("DEV_Q001", split="dev"),
    ]
    cases = train_answerable + distractors

    selected = select_representative_train_cases(
        cases,
        sample_size=60,
    )
    selected_reversed = select_representative_train_cases(
        list(reversed(cases)),
        sample_size=60,
    )

    selected_ids = [case.question_id for case in selected]
    reversed_ids = [
        case.question_id
        for case in selected_reversed
    ]

    assert DEFAULT_EVIDENCE_SAMPLE_SEED == "techqa-evidence-audit-v1"
    assert len(selected_ids) == 60
    assert len(set(selected_ids)) == 60

    assert all(case.split == "train" for case in selected)
    assert all(case.answerable for case in selected)

    assert selected_ids == reversed_ids
    assert selected_ids[:10] == [
        "TRAIN_Q408",
        "TRAIN_Q068",
        "TRAIN_Q133",
        "TRAIN_Q217",
        "TRAIN_Q353",
        "TRAIN_Q243",
        "TRAIN_Q026",
        "TRAIN_Q204",
        "TRAIN_Q256",
        "TRAIN_Q221",
    ]


def test_annotation_candidates_expand_exact_answer_match_with_neighbors():
    try:
        from experiments.evals.techqa_evidence_audit import (
            build_annotation_candidates,
        )
        from experiments.evals.retrievers.bm25_techqa_chunks import (
            TechQAChunk,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.fail("TechQA evidence annotation candidates are not implemented yet")

    chunks = [
        TechQAChunk(
            chunk_id="doc_chunk_0",
            document_id="doc",
            chunk_index=0,
            content="Unrelated introduction.",
        ),
        TechQAChunk(
            chunk_id="doc_chunk_1",
            document_id="doc",
            chunk_index=1,
            content="Configuration background before the answer.",
        ),
        TechQAChunk(
            chunk_id="doc_chunk_2",
            document_id="doc",
            chunk_index=2,
            content=(
                "Resolution: Restart the service after changing "
                "the configuration."
            ),
        ),
        TechQAChunk(
            chunk_id="doc_chunk_3",
            document_id="doc",
            chunk_index=3,
            content="Additional notes after the resolution.",
        ),
        TechQAChunk(
            chunk_id="doc_chunk_4",
            document_id="doc",
            chunk_index=4,
            content="Unrelated appendix.",
        ),
    ]

    candidates = build_annotation_candidates(
        chunks,
        gold_answer="Restart the service after changing the configuration.",
    )

    assert [chunk.chunk_id for chunk in candidates] == [
        "doc_chunk_1",
        "doc_chunk_2",
        "doc_chunk_3",
    ]


def test_annotation_candidates_normalize_whitespace_for_answer_localization():
    from experiments.evals.techqa_evidence_audit import (
        build_annotation_candidates,
    )
    from experiments.evals.retrievers.bm25_techqa_chunks import (
        TechQAChunk,
    )

    chunks = [
        TechQAChunk(
            chunk_id="doc_chunk_0",
            document_id="doc",
            chunk_index=0,
            content="Background before the resolution.",
        ),
        TechQAChunk(
            chunk_id="doc_chunk_1",
            document_id="doc",
            chunk_index=1,
            content=(
                "Resolution:\n"
                "Restart the service   after changing\n"
                "the configuration."
            ),
        ),
        TechQAChunk(
            chunk_id="doc_chunk_2",
            document_id="doc",
            chunk_index=2,
            content="Additional notes.",
        ),
    ]

    candidates = build_annotation_candidates(
        chunks,
        gold_answer=(
            "Restart the service after changing the configuration."
        ),
    )

    assert [chunk.chunk_id for chunk in candidates] == [
        "doc_chunk_0",
        "doc_chunk_1",
        "doc_chunk_2",
    ]


def test_annotation_candidates_use_local_bm25_when_answer_spans_chunks():
    from experiments.evals.techqa_evidence_audit import (
        build_annotation_candidates,
    )
    from experiments.evals.retrievers.bm25_techqa_chunks import (
        TechQAChunk,
    )

    chunks = [
        TechQAChunk(
            chunk_id=f"doc_chunk_{index}",
            document_id="doc",
            chunk_index=index,
            content=content,
        )
        for index, content in enumerate(
            [
                "unrelated introduction",
                "unrelated background",
                "neighbor before evidence",
                "alpha alpha alpha configuration cause",
                "beta beta beta resolution mechanism",
                "gamma gamma gamma remediation step",
                "neighbor after evidence",
                "unrelated appendix",
            ]
        )
    ]

    candidates = build_annotation_candidates(
        chunks,
        gold_answer="alpha beta gamma",
    )

    assert [chunk.chunk_id for chunk in candidates] == [
        "doc_chunk_2",
        "doc_chunk_3",
        "doc_chunk_4",
        "doc_chunk_5",
        "doc_chunk_6",
    ]


def test_build_annotation_pack_cases_uses_sampled_train_gold_context_only():
    try:
        from experiments.evals.techqa_evidence_audit import (
            build_annotation_pack_cases,
        )
    except ImportError:
        pytest.fail("TechQA evidence annotation pack is not implemented yet")

    rows = [
        {
            "id": "TRAIN_Q001",
            "question": "How should service A be recovered?",
            "answer": "Restart service A after updating the configuration.",
            "is_impossible": False,
            "contexts": [
                {
                    "filename": "doc_a.txt",
                    "text": (
                        "Title: Service A recovery\n\n"
                        "Restart service A after updating the configuration. "
                        "Then verify that the service is healthy."
                    ),
                }
            ],
        },
        {
            "id": "TRAIN_Q002",
            "question": "Why does component B fail?",
            "answer": "Component B fails because the cache is stale.",
            "is_impossible": False,
            "contexts": [
                {
                    "filename": "doc_b.txt",
                    "text": (
                        "Title: Component B failure\n\n"
                        "Component B fails because the cache is stale. "
                        "Clear the cache before retrying."
                    ),
                }
            ],
        },
        {
            "id": "TRAIN_I001",
            "question": "Unsupported question",
            "answer": "",
            "is_impossible": True,
            "contexts": [],
        },
        {
            "id": "DEV_Q001",
            "question": "DEV must not enter the audit sample",
            "answer": "DEV answer",
            "is_impossible": False,
            "contexts": [
                {
                    "filename": "dev_doc.txt",
                    "text": "DEV answer",
                }
            ],
        },
    ]

    pack_cases = build_annotation_pack_cases(
        rows,
        sample_size=2,
    )

    assert len(pack_cases) == 2
    assert {
        case.question_id
        for case in pack_cases
    } == {
        "TRAIN_Q001",
        "TRAIN_Q002",
    }

    by_id = {
        case.question_id: case
        for case in pack_cases
    }

    case_a = by_id["TRAIN_Q001"]

    assert case_a.question == "How should service A be recovered?"
    assert (
        case_a.gold_answer
        == "Restart service A after updating the configuration."
    )
    assert case_a.gold_document_id == "doc_a.txt"

    assert case_a.candidate_chunks
    assert all(
        chunk.document_id == "doc_a.txt"
        for chunk in case_a.candidate_chunks
    )
    assert any(
        "Restart service A after updating the configuration."
        in chunk.content
        for chunk in case_a.candidate_chunks
    )


def test_annotation_pack_markdown_is_train_only_and_annotation_ready():
    try:
        from experiments.evals.techqa_evidence_audit import (
            build_annotation_pack_markdown,
        )
    except ImportError:
        pytest.fail("TechQA evidence annotation Markdown is not implemented yet")

    rows = [
        {
            "id": "TRAIN_Q001",
            "question": "How should service A be recovered?",
            "answer": "Restart service A after updating the configuration.",
            "is_impossible": False,
            "contexts": [
                {
                    "filename": "doc_a.txt",
                    "text": (
                        "Title: Service A recovery\n\n"
                        "Restart service A after updating the configuration. "
                        "Then verify that the service is healthy."
                    ),
                }
            ],
        },
        {
            "id": "DEV_Q001",
            "question": "DEV must never enter the annotation pack.",
            "answer": "DEV answer",
            "is_impossible": False,
            "contexts": [
                {
                    "filename": "dev_doc.txt",
                    "text": "DEV answer",
                }
            ],
        },
    ]

    markdown = build_annotation_pack_markdown(
        rows,
        sample_size=1,
    )

    assert "## 01 TRAIN_Q001" in markdown
    assert "DEV_Q001" not in markdown
    assert "DEV must never enter the annotation pack." not in markdown

    assert "Question:" in markdown
    assert "How should service A be recovered?" in markdown

    assert "Gold answer:" in markdown
    assert (
        "Restart service A after updating the configuration."
        in markdown
    )

    assert "Gold document:" in markdown
    assert "doc_a.txt" in markdown

    assert "Localization:" in markdown
    assert "exact_anchor" in markdown

    assert "### Candidate 1" in markdown
    assert "chunk_id:" in markdown
    assert "evidence_label: ___" in markdown

    assert "<details>" in markdown
    assert "Full gold document" in markdown
    assert "Title: Service A recovery" in markdown
    assert "</details>" in markdown

    assert "questionable_gold: ___" in markdown
    assert "notes:" in markdown
