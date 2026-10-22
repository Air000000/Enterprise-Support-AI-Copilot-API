"""Zero-provider archival vector import into a NEW, isolated managed benchmark store.

Run with networking disabled. Never mount the output as the daily service store.
Failure leaves an incomplete directory; it cannot be resumed or overwritten.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from experiments.evals.audit_runtime_compatibility import audit_index

TENANT = "tenant_eval_techqa_archival"
PROFILE = "techqa_archival_content_vectors_managed_ids_v1"
COLLECTION = "local_doc_rag"
BATCH_SIZE = 256


def managed_batch(batch, dimension):
    """Keep original document IDs/content/vectors; rekey chunks, not embeddings."""
    ids, texts, vectors, metadatas = [], [], [], []
    for old_id, text, vector, meta in zip(
        batch["ids"], batch["documents"], batch["embeddings"], batch["metadatas"], strict=True
    ):
        doc_id, index = meta.get("document_id"), meta.get("chunk_index")
        if (not isinstance(doc_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", doc_id)
                or doc_id in {".", ".."} or type(index) is not int or index < 0
                or old_id != f"{doc_id}_chunk_{index}"):
            raise ValueError("Invalid archival chunk identity")
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError("Invalid archival chunk content")
        if len(vector) != dimension or not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("Invalid archival vector")
        cid = f"doc:{doc_id}:v1:chunk:{index}"
        ids.append(cid)
        texts.append(text)
        vectors.append(vector)
        metadatas.append({
            "chunk_id": cid, "document_id": doc_id, "document_db_id": doc_id,
            "chunk_index": index, "tenant_id": TENANT, "category": "other", "version": 1,
            "title": doc_id, "filename": f"{doc_id}.txt",
            "source_path": f"benchmark/techqa/{doc_id}.txt",
            "source_type": "archival_benchmark", "index_profile": PROFILE,
            "archival_chunk_id": old_id,
        })
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate archival chunk identity")
    return dict(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)


def import_snapshot(source_dir, output_dir, *, repo_root=Path(".")):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("Source and new output must be disjoint")
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite or resume an existing output")
    if any(p.is_symlink() for p in source_dir.rglob("*")):
        raise ValueError("Linked source assets are not accepted")
    audit = audit_index(source_dir, repo_root=repo_root)
    if not audit["all_asset_checks_passed"]:
        raise ValueError("Source asset audit failed")
    started = time.monotonic()
    output_dir.mkdir()  # Exclusive creation; parent must already exist.
    copy_dir = output_dir / "source_copy"
    shutil.copytree(source_dir, copy_dir)
    copied_audit = audit_index(copy_dir, repo_root=repo_root)
    if copied_audit["fingerprints"] != audit["fingerprints"]:
        raise ValueError("Source changed while copying")

    # Never import a provider client, application database, dotenv, or dataset loader.
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from chromadb import PersistentClient, __version__ as chroma_version
    from sqlalchemy import insert, text as sql_text
    from sqlmodel import SQLModel, create_engine
    from models.document import Document, DocumentChunk, utc_now

    source = PersistentClient(path=str(copy_dir)).get_collection(audit["collection"], embedding_function=None)
    if source.count() != audit["chunks"]:
        raise ValueError("Copied source count changed")
    source_space = source.configuration["hnsw"]["space"]
    if source_space != "l2":
        raise ValueError("Only the audited historical L2 profile is supported")
    target = PersistentClient(path=str(output_dir / "chroma")).create_collection(
        COLLECTION, embedding_function=None, configuration={"hnsw": {"space": source_space}},
        metadata={"index_profile": PROFILE, "historical_model_attested": False})
    engine = create_engine("sqlite:///" + str(output_dir / "runtime.db"))
    SQLModel.metadata.create_all(engine, tables=[Document.__table__, DocumentChunk.__table__])
    counts, seen = Counter(), set()
    digest = hashlib.sha256()
    try:
        with engine.begin() as connection:
            for offset in range(0, audit["chunks"], BATCH_SIZE):
                raw = source.get(limit=BATCH_SIZE, offset=offset, include=["documents", "metadatas", "embeddings"])
                batch = managed_batch(raw, audit["dimension"])
                if not batch["ids"] or seen.intersection(batch["ids"]):
                    raise ValueError("Missing or duplicate source batch")
                seen.update(batch["ids"])
                target.add(**batch)  # Existing vectors explicitly supplied: no embedding function.
                stored = target.get(ids=batch["ids"], include=["documents", "metadatas", "embeddings"])
                by_id = {cid: (body, meta, list(vector)) for cid, body, meta, vector in zip(
                    stored["ids"], stored["documents"], stored["metadatas"], stored["embeddings"], strict=True)}
                chunks = []
                for cid, body, meta, vector in zip(
                    batch["ids"], batch["documents"], batch["metadatas"], batch["embeddings"], strict=True
                ):
                    if by_id.get(cid) != (body, meta, list(vector)):
                        raise ValueError("Imported body/metadata/vector readback mismatch")
                    digest.update(json.dumps([cid, body, meta, list(map(float, vector))],
                                             sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
                    counts[meta["document_id"]] += 1
                    chunks.append(dict(id=cid, tenant_id=TENANT, document_id=meta["document_id"],
                                       chunk_index=meta["chunk_index"], content=body, category="other",
                                       metadata_json=json.dumps(meta), embedding_id=cid, created_at=utc_now()))
                connection.execute(insert(DocumentChunk), chunks)
                if offset % (BATCH_SIZE * 40) == 0:
                    print(f"Imported and readback-verified chunks: {len(seen)}", flush=True)
            if len(counts) != audit["documents"] or len(seen) != audit["chunks"] or target.count() != len(seen):
                raise ValueError("Imported corpus count mismatch")
            # ponytail: one SQL transaction, no resumable importer; rebuild a new output after failure.
            connection.execute(insert(Document), [dict(
                id=doc_id, tenant_id=TENANT, uploaded_by="offline_benchmark_import",
                filename=f"{doc_id}.txt", file_type="txt", category="other",
                source_path=f"benchmark/techqa/{doc_id}.txt", status="indexed", version=1,
                checksum="archival-vector-profile-not-upload-checksum", chunk_count=count,
                created_at=utc_now(), updated_at=utc_now(),
            ) for doc_id, count in counts.items()])
            if connection.execute(sql_text("PRAGMA quick_check")).scalar() != "ok":
                raise ValueError("Imported SQL integrity check failed")
        final_audit = audit_index(source_dir, repo_root=repo_root)
        if final_audit["fingerprints"] != audit["fingerprints"]:
            raise ValueError("Original source changed during import")
        report = {
            "status": "ARCHIVAL_IMPORT_VERIFIED_NOT_RUNTIME_ACCEPTANCE", "index_profile": PROFILE,
            "provider_calls": 0, "dev_examples_opened": False, "source_modified": False,
            "documents": len(counts), "chunks": len(seen), "dimension": audit["dimension"],
            "tenant_id": TENANT, "collection": COLLECTION, "distance_space": source_space,
            "chromadb": chroma_version, "elapsed_seconds": round(time.monotonic() - started, 3),
            "source_fingerprints": audit["fingerprints"], "managed_records_sha256": digest.hexdigest(),
            "all_imported_vectors_bodies_metadata_readback_equal": True,
            "embedding_model_metadata": audit["embedding_model_metadata"],
            "embedding_endpoint_metadata": audit["embedding_endpoint_metadata"],
            "embedding_input": "historical chunk content, no title prefix",
            "ordinary_upload_index_equivalent": False, "hnsw_rebuilt_from_existing_vectors": True,
            "limitations": ["Historical model/endpoint identity is not attested.",
                            "New HNSW construction/IDs can change approximate rankings and ties.",
                            "Synthetic filenames/paths identify benchmark records, not uploaded source files.",
                            "SQL contains document/chunk tables only, not a migrated serving database.",
                            "Full-corpus runtime capacity, latency and provider bounds remain unverified."],
            "runtime_quality_evaluation_ready": False,
        }
        with (output_dir / "import_receipt.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist; use an isolated eval directory")
    args = parser.parse_args()
    print(json.dumps(import_snapshot(args.source_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
