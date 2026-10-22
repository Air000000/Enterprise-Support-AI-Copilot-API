"""Synthetic SQLite only; no model, vector client, DEV loader or private index."""

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from experiments.evals.audit_runtime_compatibility import audit_index


@pytest.fixture
def asset(tmp_path):
    root, index = tmp_path / "repo", tmp_path / "index"
    for relative, content in {
        "experiments/evals/datasets/techqa/manifest.json": json.dumps({"retrieval_dataset": {
            "corpus_sha256": "a" * 64, "loaded_corpus_documents": 1}}),
        "experiments/evals/reports/r4_c1_hybrid_rerank/train_preflight.json": '{"chunk_count":2}',
        "rag_runtime/hybrid_search.py": "MAX_VISIBLE_CHUNKS = 1\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    index.mkdir()
    db = index / "chroma.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.executescript("""
            CREATE TABLE collections(id TEXT,name TEXT,dimension INTEGER);
            CREATE TABLE collection_metadata(collection_id TEXT,key TEXT,str_value TEXT);
            CREATE TABLE segments(id TEXT,scope TEXT,collection TEXT);
            CREATE TABLE embeddings(id INTEGER,segment_id TEXT,embedding_id TEXT);
            CREATE TABLE embedding_metadata(id INTEGER,key TEXT,string_value TEXT,int_value INTEGER);
            INSERT INTO collections VALUES('collection','techqa_e0_dense',1024);
            INSERT INTO collection_metadata VALUES('collection','corpus_sha256','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
            INSERT INTO collection_metadata VALUES('collection','benchmark','techqa');
            INSERT INTO collection_metadata VALUES('collection','chunk_strategy','paragraph_aware_character');
            INSERT INTO segments VALUES('00000000-0000-0000-0000-000000000001','METADATA','collection');
            INSERT INTO segments VALUES('00000000-0000-0000-0000-000000000002','VECTOR','collection');
        """)
        for i in range(2):
            connection.execute("INSERT INTO embeddings VALUES(?,?,?)", (i, "00000000-0000-0000-0000-000000000001", f"doc_chunk_{i}"))
            connection.executemany("INSERT INTO embedding_metadata VALUES(?,?,?,?)", [
                (i, "document_id", "doc", None), (i, "chunk_index", None, i),
                (i, "chroma:document", "synthetic body", None)])
    vector = index / "00000000-0000-0000-0000-000000000002"
    vector.mkdir()
    for name in ("header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"):
        (vector / name).write_bytes(b"Not a vector or pickle; must never be deserialized")
    return root, index, db


def test_audit_is_read_only_and_never_promotes_metadata_checks(asset, monkeypatch):
    root, index, db = asset
    paths = [p for p in index.rglob("*") if p.is_file()]
    before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
    connect = sqlite3.connect
    def readonly(database, **kwargs):
        assert str(database).endswith("?mode=ro") and kwargs["uri"] is True
        return connect(database, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", readonly)
    report = audit_index(index, repo_root=root)
    assert report["all_asset_checks_passed"] is True
    assert report["runtime_quality_evaluation_ready"] is False
    assert report["fits_current_runtime_capacity"] is False
    assert report["embedding_model_metadata"] is None
    assert report["dimension"] == 1024 and report["provider_calls"] == 0
    assert before == {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}


@pytest.mark.parametrize("sql,check", [
    ("UPDATE collection_metadata SET str_value='wrong' WHERE key='corpus_sha256'", "declared_corpus_sha_matches"),
    ("DELETE FROM embeddings WHERE id=1", "chunk_count_matches"),
    ("UPDATE embeddings SET embedding_id='wrong' WHERE id=1", "chunk_id_content_metadata_consistent"),
    ("UPDATE collections SET dimension=0", "positive_recorded_dimension"),
])
def test_incompatible_assets_are_not_accepted(asset, sql, check):
    root, index, db = asset
    with sqlite3.connect(db) as connection:
        connection.execute(sql)
    report = audit_index(index, repo_root=root)
    assert report["asset_checks"][check] is False
    assert report["all_asset_checks_passed"] is False
    assert report["runtime_quality_evaluation_ready"] is False


def test_missing_index_is_not_created(asset):
    root, index, _ = asset
    missing = index / "absent"
    with pytest.raises(FileNotFoundError):
        audit_index(missing, repo_root=root)
    assert not missing.exists()


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_active_journal_requires_snapshot(asset, suffix):
    root, index, db = asset
    Path(str(db) + suffix).write_bytes(b"do not open the database")
    with pytest.raises(ValueError, match="Journal/WAL"):
        audit_index(index, repo_root=root)


def test_segment_path_cannot_escape_index(asset):
    root, index, db = asset
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE segments SET id='../outside' WHERE scope='VECTOR'")
    with pytest.raises(ValueError):
        audit_index(index, repo_root=root)
