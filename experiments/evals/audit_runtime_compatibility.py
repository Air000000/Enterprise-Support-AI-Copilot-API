"""Read-only phase-0 asset audit. No Chroma client, dotenv, DEV loader or model API."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import UUID


def _sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit_index(index_dir, *, repo_root=Path(".")):
    root, index_dir = Path(repo_root), Path(index_dir).resolve()
    manifest = json.loads((root / "experiments/evals/datasets/techqa/manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((root / "experiments/evals/reports/r4_c1_hybrid_rerank/train_preflight.json").read_text(encoding="utf-8"))
    source = root / "rag_runtime/hybrid_search.py"
    capacity = next(ast.literal_eval(node.value) for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAX_VISIBLE_CHUNKS" for t in node.targets))
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Unrecognized runtime capacity")
    db_path = index_dir / "chroma.sqlite3"
    if not db_path.is_file():
        raise FileNotFoundError("Existing chroma.sqlite3 required; no index is created")
    if db_path.is_symlink():
        raise ValueError("Linked index database is not accepted")
    if any(Path(str(db_path) + suffix).exists() for suffix in ("-wal", "-journal")):
        raise ValueError("Journal/WAL present; obtain a quiescent isolated snapshot before auditing")
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in index_dir.rglob("*") if p.is_file()}
    connection = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("SQLite quick_check failed")
        rows = connection.execute("SELECT id,dimension FROM collections WHERE name=?", ("techqa_e0_dense",)).fetchall()
        if len(rows) != 1:
            raise ValueError("Exactly one named TechQA collection is required")
        collection, dimension = rows[0]
        metadata = dict(connection.execute("SELECT key,str_value FROM collection_metadata WHERE collection_id=?", (collection,)))
        segments = connection.execute("SELECT id,scope FROM segments WHERE collection=?", (collection,)).fetchall()
        if len(segments) != 2 or {scope for _, scope in segments} != {"METADATA", "VECTOR"}:
            raise ValueError("Expected one persisted metadata and vector segment")
        for segment_id, _ in segments:
            if str(UUID(segment_id)) != segment_id:
                raise ValueError("Unsafe or unrecognized segment directory")
        metadata_segment = next(s for s, scope in segments if scope == "METADATA")
        vector_segment = next(s for s, scope in segments if scope == "VECTOR")
        chunk_count, unique_ids = connection.execute(
            "SELECT count(*),count(DISTINCT embedding_id) FROM embeddings WHERE segment_id=?", (metadata_segment,)).fetchone()
        document_count = connection.execute("""SELECT count(DISTINCT m.string_value) FROM embedding_metadata m
            JOIN embeddings e ON e.id=m.id WHERE e.segment_id=? AND m.key='document_id'""", (metadata_segment,)).fetchone()[0]
        invalid = connection.execute("""SELECT count(*) FROM embeddings e
            LEFT JOIN embedding_metadata d ON d.id=e.id AND d.key='document_id'
            LEFT JOIN embedding_metadata i ON i.id=e.id AND i.key='chunk_index'
            LEFT JOIN embedding_metadata c ON c.id=e.id AND c.key='chroma:document'
            WHERE e.segment_id=? AND (d.string_value IS NULL OR i.int_value IS NULL OR i.int_value<0
                OR c.string_value IS NULL OR trim(c.string_value)=''
                OR e.embedding_id != d.string_value || '_chunk_' || i.int_value)""", (metadata_segment,)).fetchone()[0]
    finally:
        connection.close()
    vector_dir = index_dir / vector_segment
    files = sorted(vector_dir.glob("*")) if vector_dir.is_dir() else []
    required = {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin", "index_metadata.pickle"}
    available = {p.name for p in files if p.is_file() and not p.is_symlink()}
    # ponytail: fingerprint persisted assets; never deserialize a pickle or mutate the source with Chroma.
    if db_path.is_symlink() or vector_dir.is_symlink() or available != {p.name for p in files}:
        raise ValueError("Linked or non-file index assets are not accepted")
    hashes = {"chroma.sqlite3": _sha256(db_path)}
    hashes.update({p.name: _sha256(p) for p in files})
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in index_dir.rglob("*") if p.is_file()}
    if before != after:
        raise ValueError("Index changed during read-only audit; discard this report")
    expected = manifest["retrieval_dataset"]
    checks = {
        "declared_corpus_sha_matches": metadata.get("corpus_sha256") == expected["corpus_sha256"],
        "declared_benchmark_matches": metadata.get("benchmark") == "techqa",
        "declared_chunk_strategy_matches": metadata.get("chunk_strategy") == "paragraph_aware_character",
        "chunk_count_matches": chunk_count == preflight["chunk_count"],
        "document_count_matches": document_count == expected["loaded_corpus_documents"],
        "chunk_ids_unique": unique_ids == chunk_count,
        "chunk_id_content_metadata_consistent": invalid == 0,
        "persisted_vector_files_present": required <= available,
        "positive_recorded_dimension": type(dimension) is int and dimension > 0,
    }
    return {
        "status": "PHASE0_ASSET_AUDIT_ONLY_NOT_RUNTIME_ACCEPTANCE",
        "provider_calls": 0,
        "dev_examples_opened": False,
        "source_modified": False,
        "asset_checks": checks,
        "all_asset_checks_passed": all(checks.values()),
        "collection": "techqa_e0_dense",
        "documents": document_count,
        "chunks": chunk_count,
        "dimension": dimension,
        "invalid_chunk_records": invalid,
        "embedding_model_metadata": metadata.get("embedding_model"),
        "embedding_endpoint_metadata": metadata.get("embedding_endpoint"),
        "fingerprints": hashes,
        "runtime_visible_chunk_limit": capacity,
        "runtime_source_sha256": _sha256(source),
        "fits_current_runtime_capacity": chunk_count <= capacity,
        "not_verified": [
            "Historical embedding model/endpoint are not attested by current defaults or dimension.",
            "Corpus SHA is a metadata declaration, not an independent body-to-source or vector provenance proof.",
            "Vector bytes are fingerprinted, not deserialized or queried; Chroma load/query compatibility is untested.",
            "Normal managed upload embedding input differs from the historical plain-content builder.",
            "SQL tenant/version/chunk mapping, full-corpus latency and bounded provider calls remain unverified.",
        ],
        "runtime_quality_evaluation_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, required=True, help="Existing quiescent TechQA index; read-only")
    args = parser.parse_args()
    report = audit_index(args.index_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_asset_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
