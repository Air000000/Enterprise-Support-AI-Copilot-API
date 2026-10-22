"""Real tiny local Chroma/SQL; external network and private assets are unnecessary."""

import copy
import hashlib
import json

import chromadb
import pytest
from sqlmodel import Session, create_engine, select

from experiments.evals import import_runtime_snapshot as importer
from models.document import Document, DocumentChunk
import rag_runtime.hybrid_search as hybrid


def sample():
    return dict(ids=["sample_chunk_0", "sample_chunk_1"], documents=["first body", "second body"],
                embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                metadatas=[dict(document_id="sample", chunk_index=i) for i in range(2)])


@pytest.mark.parametrize("change", [
    lambda b: b["metadatas"][0].update(document_id="../escape"),
    lambda b: b["metadatas"][0].update(chunk_index=True),
    lambda b: b["ids"].__setitem__(0, "stale_chunk_0"),
    lambda b: b["documents"].__setitem__(0, " "),
    lambda b: b["documents"].__setitem__(0, "x" * 8001),
    lambda b: b["embeddings"].__setitem__(0, [1.0]),
    lambda b: b["embeddings"][0].__setitem__(0, float("nan")),
    lambda b: b["ids"].pop(),
])
def test_malformed_archival_batch_is_rejected(change):
    batch = copy.deepcopy(sample())
    change(batch)
    with pytest.raises(ValueError):
        importer.managed_batch(batch, 3)


def test_real_import_is_isolated_and_runtime_visibility_stays_strict(tmp_path, monkeypatch):
    source = tmp_path / "original"
    original = chromadb.PersistentClient(path=str(source)).create_collection("techqa_e0_dense", embedding_function=None)
    original.add(**sample())
    del original

    def audit(path, **kwargs):
        return dict(all_asset_checks_passed=True, collection="techqa_e0_dense", chunks=2, documents=1,
                    dimension=3, embedding_model_metadata=None, embedding_endpoint_metadata=None,
                    fingerprints={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in path.rglob("*") if p.is_file()})
    monkeypatch.setattr(importer, "audit_index", audit)
    before = audit(source)["fingerprints"]
    output = tmp_path / "isolated"
    receipt = importer.import_snapshot(source, output)
    assert receipt == json.loads((output / "import_receipt.json").read_text(encoding="utf-8"))
    assert receipt["provider_calls"] == 0 and receipt["runtime_quality_evaluation_ready"] is False
    assert receipt["embedding_model_metadata"] is None
    assert receipt["all_imported_vectors_bodies_metadata_readback_equal"] is True
    assert audit(source)["fingerprints"] == before

    engine = create_engine("sqlite:///" + str(output / "runtime.db"))
    monkeypatch.setattr(hybrid, "engine", engine)
    try:
        visible = hybrid.load_visible_chunks(importer.TENANT, "other")
        assert set(visible) == {"doc:sample:v1:chunk:0", "doc:sample:v1:chunk:1"}
        assert hybrid.load_visible_chunks("different_tenant", "other") == {}
        assert hybrid.load_visible_chunks(importer.TENANT, "it") == {}
        with Session(engine) as session:
            doc = session.get(Document, "sample")
            assert doc.chunk_count == 2
            chunks = session.exec(select(DocumentChunk).order_by(DocumentChunk.chunk_index)).all()
            assert [c.content for c in chunks] == sample()["documents"]
            assert all(json.loads(c.metadata_json)["index_profile"] == importer.PROFILE for c in chunks)
            doc.version = 2
            session.add(doc)
            session.commit()
        assert hybrid.load_visible_chunks(importer.TENANT, "other") == {}
    finally:
        engine.dispose()
    with pytest.raises(FileExistsError):
        importer.import_snapshot(source, output)
    with pytest.raises(ValueError, match="disjoint"):
        importer.import_snapshot(source, source / "nested")


def test_failed_source_audit_creates_no_output(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    monkeypatch.setattr(importer, "audit_index", lambda *a, **kw: {"all_asset_checks_passed": False})
    with pytest.raises(ValueError, match="audit failed"):
        importer.import_snapshot(source, output)
    assert not output.exists()


def test_changed_copy_has_no_success_receipt(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "new"
    source.mkdir()
    calls = []
    def audit(path, **kwargs):
        calls.append(path)
        return {"all_asset_checks_passed": True, "fingerprints": {"file": len(calls)}}
    monkeypatch.setattr(importer, "audit_index", audit)
    with pytest.raises(ValueError, match="changed while copying"):
        importer.import_snapshot(source, output)
    assert not (output / "import_receipt.json").exists()
