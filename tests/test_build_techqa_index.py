import pytest

from experiments.evals.adapters.techqa import TechQADocument
from experiments.evals.build_techqa_index import (
    TechQAIndexBuildSummary,
    build_techqa_index,
    load_frozen_techqa_documents,
    search_techqa_index,
)
from rag_runtime.build_chroma_index import get_chroma_client


def _fake_embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for text in texts:
        lowered = text.lower()

        if "alpha" in lowered:
            embeddings.append([1.0, 0.0])
        elif "beta" in lowered:
            embeddings.append([0.0, 1.0])
        else:
            embeddings.append([0.5, 0.5])

    return embeddings


def _small_corpus() -> list[TechQADocument]:
    return [
        TechQADocument(document_id="doc_b", text="beta support resolution"),
        TechQADocument(document_id="doc_a", text="alpha support resolution"),
    ]


def test_load_frozen_corpus_uses_manifest_identity():
    observed: dict[str, object] = {}

    def fake_load_dataset(path, name, *, split, revision):
        observed.update(
            path=path,
            name=name,
            split=split,
            revision=revision,
        )
        return (
            {"_id": f"doc_{index:05d}", "text": f"document {index}"}
            for index in range(28481)
        )

    documents = load_frozen_techqa_documents(dataset_loader=fake_load_dataset)

    assert observed == {
        "path": "bowang0911/TechQA-RAG-Eval",
        "name": "corpus",
        "split": "train",
        "revision": "68323f8f191fd5df93e2b2673d79a5da3a805638",
    }
    assert len(documents) == 28481
    assert documents[0].document_id == "doc_00000"


def test_load_frozen_corpus_rejects_count_mismatch():
    def fake_load_dataset(path, name, *, split, revision):
        return [{"_id": "doc_only", "text": "only one document"}]

    with pytest.raises(RuntimeError, match="expected=28481"):
        load_frozen_techqa_documents(dataset_loader=fake_load_dataset)


def test_build_small_corpus_is_deterministic_and_records_metadata(tmp_path):
    chroma_dir = tmp_path / "techqa_chroma"
    collection_name = "techqa_test"

    summary = build_techqa_index(
        _small_corpus(),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=_fake_embed_texts,
    )

    assert isinstance(summary, TechQAIndexBuildSummary)
    assert summary.document_count == 2
    assert summary.chunk_count == 2
    assert summary.indexing_seconds >= 0

    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )
    stored = collection.get(include=["metadatas"])
    metadata_by_id = dict(zip(stored["ids"], stored["metadatas"]))

    assert set(metadata_by_id) == {"doc_a_chunk_0", "doc_b_chunk_0"}
    assert metadata_by_id["doc_a_chunk_0"] == {
        "benchmark": "techqa",
        "chunk_id": "doc_a_chunk_0",
        "chunk_index": 0,
        "chunk_strategy": "paragraph_aware_character",
        "corpus_sha256": "corpus-sha-demo",
        "document_id": "doc_a",
    }


def test_duplicate_build_is_idempotent(tmp_path):
    chroma_dir = tmp_path / "techqa_chroma"
    collection_name = "techqa_test"

    first = build_techqa_index(
        _small_corpus(),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=_fake_embed_texts,
    )
    second = build_techqa_index(
        _small_corpus(),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=_fake_embed_texts,
    )

    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    assert first.chunk_count == 2
    assert second.chunk_count == 2
    assert collection.count() == 2


def test_resume_skips_existing_chunks_and_embeds_only_missing(tmp_path):
    chroma_dir = tmp_path / "techqa_chroma"
    collection_name = "techqa_test"

    build_techqa_index(
        [TechQADocument(document_id="doc_a", text="alpha support resolution")],
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=_fake_embed_texts,
    )

    embedded_texts: list[str] = []

    def tracking_embedder(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return _fake_embed_texts(texts)

    summary = build_techqa_index(
        _small_corpus(),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=tracking_embedder,
    )

    client = get_chroma_client(chroma_dir)
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,
    )

    assert summary.chunk_count == 2
    assert embedded_texts == ["beta support resolution"]
    assert collection.count() == 2


def test_resume_rejects_incompatible_corpus_identity(tmp_path):
    chroma_dir = tmp_path / "techqa_chroma"
    collection_name = "techqa_test"

    build_techqa_index(
        _small_corpus(),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-one",
        embedder=_fake_embed_texts,
    )

    with pytest.raises(RuntimeError, match="corpus_sha256"):
        build_techqa_index(
            _small_corpus(),
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            corpus_sha256="corpus-sha-two",
            embedder=_fake_embed_texts,
        )


def test_fresh_rebuild_reembeds_existing_chunks(tmp_path):
    chroma_dir = tmp_path / "techqa_chroma"
    collection_name = "techqa_test"
    documents = [
        TechQADocument(document_id="doc_a", text="alpha support resolution")
    ]

    build_techqa_index(
        documents,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=_fake_embed_texts,
    )

    embedded_texts: list[str] = []

    def tracking_embedder(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return _fake_embed_texts(texts)

    build_techqa_index(
        documents,
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=tracking_embedder,
        fresh=True,
    )

    assert embedded_texts == ["alpha support resolution"]


def test_query_returns_canonical_document_id(tmp_path):
    chroma_dir = tmp_path / "techqa_chroma"
    collection_name = "techqa_test"

    build_techqa_index(
        _small_corpus(),
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        corpus_sha256="corpus-sha-demo",
        embedder=_fake_embed_texts,
    )

    results = search_techqa_index(
        "alpha",
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        top_k=1,
        embedder=_fake_embed_texts,
    )

    assert len(results) == 1
    assert results[0].document_id == "doc_a"
    assert results[0].chunk_id == "doc_a_chunk_0"
