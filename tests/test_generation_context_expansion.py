from __future__ import annotations

from dataclasses import dataclass

import experiments.evals.eval_techqa_generation as generation


@dataclass(frozen=True)
class StubChunk:
    chunk_id: str
    document_id: str
    content: str


def test_expand_g0_generation_context_uses_bounded_forward_document_context():
    expand = getattr(
        generation,
        "expand_g0_generation_context",
        None,
    )

    assert callable(expand), (
        "expected expand_g0_generation_context to exist"
    )

    anchors = (
        StubChunk(
            chunk_id="doc-a_chunk_1",
            document_id="doc-a",
            content="a1",
        ),
        # Same document appears again in the anchor set.
        # The higher-ranked first anchor should own expansion.
        StubChunk(
            chunk_id="doc-a_chunk_3",
            document_id="doc-a",
            content="a3",
        ),
        StubChunk(
            chunk_id="doc-b_chunk_0",
            document_id="doc-b",
            content="b0",
        ),
        StubChunk(
            chunk_id="doc-c_chunk_0",
            document_id="doc-c",
            content="c0",
        ),
    )

    forward_siblings = {
        "doc-a_chunk_1": (
            StubChunk(
                chunk_id="doc-a_chunk_2",
                document_id="doc-a",
                content="a2",
            ),
            StubChunk(
                chunk_id="doc-a_chunk_3",
                document_id="doc-a",
                content="a3",
            ),
            StubChunk(
                chunk_id="doc-a_chunk_4",
                document_id="doc-a",
                content="a4",
            ),
        ),
        "doc-b_chunk_0": (
            StubChunk(
                chunk_id="doc-b_chunk_1",
                document_id="doc-b",
                content="b1",
            ),
            StubChunk(
                chunk_id="doc-b_chunk_2",
                document_id="doc-b",
                content="b2",
            ),
            StubChunk(
                chunk_id="doc-b_chunk_3",
                document_id="doc-b",
                content="b3",
            ),
        ),
        "doc-c_chunk_0": (
            StubChunk(
                chunk_id="doc-c_chunk_1",
                document_id="doc-c",
                content="c1",
            ),
        ),
    }

    def load_forward_siblings(
        anchor: StubChunk,
        limit: int,
    ) -> tuple[StubChunk, ...]:
        return forward_siblings.get(
            anchor.chunk_id,
            (),
        )[:limit]

    expanded = expand(
        anchors,
        load_forward_siblings=load_forward_siblings,
        max_forward_chunks=3,
        max_context_chunks=6,
    )

    assert tuple(
        chunk.chunk_id
        for chunk in expanded
    ) == (
        "doc-a_chunk_1",
        "doc-a_chunk_2",
        "doc-a_chunk_3",
        "doc-a_chunk_4",
        "doc-b_chunk_0",
        "doc-b_chunk_1",
    )
