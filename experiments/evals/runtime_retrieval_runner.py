"""Explicit phase-1 prepare/execute/report entrypoint. Never starts on import.

Approval files bind scope; they do not manufacture actual human consent. The
operator must obtain that consent and recheck current prices before --execute.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from unittest.mock import patch

MANIFEST = Path(__file__).parent / "datasets/techqa/manifest.json"
DOCS, CHUNKS = 28481, 172614
MANAGED_SHA = "7ea91b0d516761bdf80e65397379c2e3a934e105a30ca956846e4a025e8e8fc5"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inventory(root):
    root = Path(root)
    if not root.is_dir() or root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Regular isolated snapshot directory required")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if any(path.name.endswith(("-wal", "-shm", "-journal")) for path in files):
        raise ValueError("Quiescent snapshot required")
    return {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files}


def stat_inventory(root):
    return {path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in Path(root).rglob("*") if path.is_file()}


def readonly(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Existing read-only database required")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    db.execute("PRAGMA query_only=ON")
    return db


def verify_snapshot(root):
    from experiments.evals.runtime_retrieval import TENANT, CATEGORY

    root = Path(root)
    receipt = read_json(root / "import_receipt.json")
    if (receipt.get("managed_records_sha256") != MANAGED_SHA or not receipt.get("all_imported_vectors_bodies_metadata_readback_equal")
            or (receipt.get("documents"), receipt.get("chunks"), receipt.get("dimension")) != (DOCS, CHUNKS, 1024)
            or receipt.get("index_profile") != "techqa_archival_content_vectors_managed_ids_v1"):
        raise ValueError("Wrong archival import receipt")
    with readonly(root / "runtime.db") as db:
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("Snapshot SQL integrity failed")
        documents = db.execute("SELECT id,tenant_id,category,status,version,chunk_count FROM documents").fetchall()
        chunks = db.execute("SELECT embedding_id,document_id,chunk_index,tenant_id,category FROM document_chunks").fetchall()
        if (len(documents) != DOCS or len(chunks) != CHUNKS or len({row[0] for row in chunks}) != CHUNKS
                or any(row[1:5] != (TENANT, CATEGORY, "indexed", 1) for row in documents)
                or any(row[0] != f"doc:{row[1]}:v1:chunk:{row[2]}" or row[2] < 0
                       or row[3:] != (TENANT, CATEGORY) for row in chunks)
                or dict(db.execute("SELECT document_id,count(*) FROM document_chunks GROUP BY document_id"))
                   != {row[0]: row[5] for row in documents}):
            raise ValueError("Snapshot managed SQL mapping failed")
    with readonly(root / "chroma/chroma.sqlite3") as db:
        collections = db.execute("SELECT id,dimension FROM collections WHERE name='local_doc_rag'").fetchall()
        if len(collections) != 1 or collections[0][1] != 1024:
            raise ValueError("Snapshot Chroma collection mismatch")
        ids = {row[0] for row in db.execute("SELECT e.embedding_id FROM embeddings e JOIN segments s ON s.id=e.segment_id WHERE s.collection=? AND s.scope='METADATA'", (collections[0][0],))}
        if ids != {row[0] for row in chunks}:
            raise ValueError("Snapshot SQL/Chroma identities differ")
    return {row[0] for row in documents}


def load_inputs(dataset_root):
    import pyarrow.parquet as pq

    root, spec = Path(dataset_root), read_json(MANIFEST)["retrieval_dataset"]
    paths = {key: root / spec[key + "_file"] for key in ("queries", "qrels")}
    if any(not path.is_file() or sha(path) != spec[key + "_sha256"] for key, path in paths.items()):
        raise ValueError("Frozen dataset fingerprint mismatch; no downloads allowed")
    ids = pq.read_table(paths["queries"], columns=["_id"]).column(0).to_pylist()
    selected = sorted(cid for cid in ids if isinstance(cid, str) and cid.startswith("DEV_"))
    if len(ids) != 610 or len(set(ids)) != 610 or len(selected) != 160:
        raise ValueError("Frozen DEV160 selection changed")
    rows = pq.read_table(paths["queries"], columns=["_id", "text"], filters=[("_id", "in", selected)]).to_pylist()
    pairs = sorted((row["_id"], row["text"].rstrip()) for row in rows)
    labels = pq.read_table(paths["qrels"], columns=["query-id", "corpus-id", "score"], filters=[("query-id", "in", selected)]).to_pylist()
    if ([cid for cid, _ in pairs] != selected or any(not text.strip() for _, text in pairs)
            or len(labels) != 160 or {row["query-id"] for row in labels} != set(selected)
            or any(row["score"] != 1 or not isinstance(row["corpus-id"], str) for row in labels)):
        raise ValueError("Frozen question/qrel contract failed")
    return pairs, {row["query-id"]: row["corpus-id"] for row in labels}


def prepare(dataset_root, snapshot_root, tokenizer, output):
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from experiments.evals import runtime_retrieval as evaluation

    dataset_root, snapshot_root, output = map(lambda p: Path(p).resolve(), (dataset_root, snapshot_root, output))
    if (output.exists() or any(output == root or output in root.parents or root in output.parents
                               for root in (dataset_root, snapshot_root))
            or output == Path(tokenizer).resolve() or output in Path(tokenizer).resolve().parents):
        raise ValueError("New output directory disjoint from all source assets required")
    before = inventory(snapshot_root)
    documents = verify_snapshot(snapshot_root)
    pairs, qrels = load_inputs(dataset_root)
    if not set(qrels.values()) <= documents:
        raise ValueError("Gold documents missing from snapshot")
    plan = evaluation.build_plan(pairs, evaluation.wire.digest(before), "PENDING_HUMAN_APPROVAL_NOT_AUTHORIZED",
                                 tokenizer_path=tokenizer, planned_mode="live")
    for _, query in pairs:
        if not evaluation.inputs.estimate_input("embedding", {"input": [query]}, plan["identity"]["reference_tokenizer"])["within_reference_profile"]:
            raise ValueError("Query exceeds reference input profile")
    if inventory(snapshot_root) != before:
        raise ValueError("Snapshot changed during preparation")
    prepared = {"plan": plan, "snapshot_root": str(snapshot_root), "dataset_root": str(dataset_root),
                "snapshot_inventory": before, "qrels_sha256": evaluation.wire.digest(qrels)}
    output.mkdir()
    evaluation.wire.write_json(output / "prepared.json", prepared)
    return {"status": "PREPARED_NO_PROVIDER_CALLS", "prepared_sha256": sha(output / "prepared.json"),
            "plan_sha256": evaluation.wire.digest(plan), "max_provider_calls": 480, "accounting_limit_cny": "35"}


def validate_approval(approval_path, expected_sha, prepared_path):
    if sha(approval_path) != expected_sha:
        raise ValueError("Approval file hash mismatch")
    approval = read_json(approval_path)
    if (set(approval) != {"human_approval_reference", "prepared_sha256", "scope", "max_provider_calls",
                          "accounting_limit_cny", "accept_reference_estimate", "current_prices_reconfirmed"}
            or not isinstance(approval["human_approval_reference"], str) or not approval["human_approval_reference"].strip()
            or approval["prepared_sha256"] != sha(prepared_path) or approval["scope"] != "DEV160_RETRIEVAL_ONLY"
            or type(approval["max_provider_calls"]) is not int or approval["max_provider_calls"] != 480
            or approval["accounting_limit_cny"] != "35" or approval["accept_reference_estimate"] is not True
            or approval["current_prices_reconfirmed"] is not True):
        raise ValueError("Explicit phase-1 scope/reference-risk/price approval required")
    return approval


def require_container():
    # Fail closed on unbounded/v1 resource profiles; no WSL/system modifications.
    if not Path("/.dockerenv").is_file():
        raise RuntimeError("Dedicated evaluation container required")
    quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    memory = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    if (quota == "max" or memory == "max" or not 0 < int(quota) / int(period) <= 4
            or not 0 < int(memory) <= 5 * 1024**3):
        raise RuntimeError("Explicit <=4CPU/5GiB cgroup-v2 limits required")


def execute(output, approval_path, approval_sha):
    # No --execute from a normal host, daily API worker or implicit SDK retry path.
    require_container()
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    from experiments.evals import runtime_retrieval as evaluation
    from experiments.evals.runtime_retrieval_report import build_report

    output = Path(output).resolve()
    approval = validate_approval(approval_path, approval_sha, output / "prepared.json")
    prepared = read_json(output / "prepared.json")
    source, plan = Path(prepared["snapshot_root"]), prepared["plan"]
    pairs, qrels = load_inputs(prepared["dataset_root"])
    if (inventory(source) != prepared["snapshot_inventory"] or not set(qrels.values()) <= verify_snapshot(source)
            or evaluation.wire.digest(qrels) != prepared["qrels_sha256"]
            or [[cid, text] for cid, text in pairs] != plan["identity"]["questions"]):
        raise ValueError("Prepared source assets changed")
    current = evaluation.build_plan(pairs, evaluation.wire.digest(prepared["snapshot_inventory"]),
        "PENDING_HUMAN_APPROVAL_NOT_AUTHORIZED", tokenizer_path=plan["identity"]["reference_tokenizer"]["path"], planned_mode="live")
    if current != plan:
        raise ValueError("Prepared code/dependencies/tokenizer/config changed")
    keys = {"embedding": os.getenv("DASHSCOPE_API_KEY"), "rerank": os.getenv("DASHSCOPE_RERANK_API_KEY")}
    if any(not isinstance(key, str) or not key.strip() for key in keys.values()):
        raise ValueError("Both explicit regional keys required; no dotenv or credential fallback")
    lock = output / "execute.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        if (output / "report.json").exists():
            raise RuntimeError("Completed report already exists; no automatic rerun")
        before = stat_inventory(source)
        with tempfile.TemporaryDirectory(prefix="phase1_owned_copy_", dir=output.parent) as temporary:
            work = Path(temporary).resolve()
            if work.parent != output.parent or source == work or source in work.parents:
                raise RuntimeError("Unsafe owned temporary copy path")
            shutil.copy2(source / "runtime.db", work / "runtime.db")
            shutil.copytree(source / "chroma", work / "chroma")
            for name, spec in prepared["snapshot_inventory"].items():
                if name == "runtime.db" or name.startswith("chroma/"):
                    if sha(work / name) != spec["sha256"]:
                        raise ValueError("Working copy differs from approved snapshot")
            from chromadb import PersistentClient
            from sqlmodel import create_engine
            engine = create_engine("sqlite:///" + str(work / "runtime.db"))
            client = None
            try:
                client = PersistentClient(path=str(work / "chroma"))
                collection = client.get_collection("local_doc_rag", embedding_function=None)
                evaluation.hybrid._bm25_index.cache_clear()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(evaluation.hybrid, "engine", engine))
                    stack.enter_context(patch.object(evaluation.hybrid, "get_collection", lambda: collection))
                    stack.enter_context(patch.dict(os.environ, {"RAG_HYBRID_MAX_VISIBLE_CHUNKS": "200000", "ANONYMIZED_TELEMETRY": "False"}))
                    cases = []
                    for index in range(160):
                        if stat_inventory(source) != before:
                            raise RuntimeError("Immutable source changed between cases")
                        cases.append(evaluation.save_case(plan, index, output / "run", keys,
                            {"embedding": None, "rerank": None}, allow_network=True))
                        print(json.dumps({"completed_cases": index + 1, "total_cases": 160}), flush=True)
                if inventory(source) != prepared["snapshot_inventory"]:
                    raise RuntimeError("Source snapshot fingerprint changed")
                events = evaluation.wire._read_jsonl(output / "run/wire/attempts.jsonl")
                report = build_report(plan, cases, qrels, events)
                report["human_approval_reference"] = approval["human_approval_reference"]
                evaluation.wire.write_json(output / "report.json", report)
            finally:
                evaluation.hybrid._bm25_index.cache_clear()
                engine.dispose()
                if client is not None:
                    client.close()
        return {"status": report["status"], "cases": 160, "provider_attempts": len(events),
                "accounting_cny_not_invoice": report["accounting_cny_not_invoice"]}
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--snapshot-root")
    parser.add_argument("--tokenizer")
    parser.add_argument("--authorization")
    parser.add_argument("--authorization-sha256")
    args = parser.parse_args()
    if args.prepare:
        if not all((args.dataset_root, args.snapshot_root, args.tokenizer)) or args.authorization or args.authorization_sha256:
            parser.error("prepare requires dataset/snapshot/tokenizer, not authorization")
        result = prepare(args.dataset_root, args.snapshot_root, args.tokenizer, args.output)
    else:
        if not args.authorization or not args.authorization_sha256 or any((args.dataset_root, args.snapshot_root, args.tokenizer)):
            parser.error("execute uses only the frozen prepared paths and explicit authorization/hash")
        result = execute(args.output, args.authorization, args.authorization_sha256)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Provider/SDK exception text can contain source bodies or credentials.
        print(json.dumps({"status": "STOPPED_NO_AUTOMATIC_RETRY", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
