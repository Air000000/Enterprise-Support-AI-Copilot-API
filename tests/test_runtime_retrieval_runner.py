"""Synthetic data only: full scoring, approval gates and isolated-copy execution."""

import hashlib
import json
from pathlib import Path
import shutil

import chromadb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlmodel import Session, select

from test_runtime_retrieval import KEYS, plan, providers, reference_asset, runtime  # noqa: F401
from experiments.evals import runtime_retrieval as evaluation
from experiments.evals import runtime_retrieval_runner as runner
from experiments.evals.runtime_retrieval_report import build_report
from models.document import Document, DocumentChunk


def example():
    value = plan()
    events = [dict(call_id=cid, stage=stage, status="complete", request_sha256="a" * 64,
                   response_sha256="b" * 64, prompt_tokens=5, total_tokens=5, latency_ms=10)
              for cid, stage in value["contract"]["calls"].items()]
    chunks = [f"doc:d000:v1:chunk:{i}" for i in range(20)] + [f"doc:d{i:03d}:v1:chunk:0" for i in range(1, 81)]
    promoted = ["doc:d004:v1:chunk:0"] + [key for key in chunks if key != "doc:d004:v1:chunk:0"]
    cases, qrels = [], {}
    for index, (cid, question) in enumerate(value["identity"]["questions"]):
        ranks = {"D": list(chunks), "DR": list(chunks), "H": list(promoted)}
        row = dict(question_id=cid, question_sha256=hashlib.sha256(question.encode()).hexdigest(),
                   chunk_rankings=ranks, document_rankings={arm: list(dict.fromkeys(key.split(":")[1] for key in keys)) for arm, keys in ranks.items()},
                   top14_chunk_ids=promoted[:14], fused_chunk_ids=chunks, bm25_chunk_ids=chunks)
        calls = [{key: event[key] for key in ("call_id", "request_sha256", "response_sha256")}
                 for event in events if event["call_id"].startswith(cid + ":")]
        cases.append(dict(plan_sha256=evaluation.wire.digest(value), index=index, result=row,
                          result_sha256=evaluation.wire.digest(row), calls=calls))
        qrels[cid] = "d004"
    return value, cases, qrels, events


def test_full160_report_is_document_level_paired_and_explicitly_synthetic():
    value, cases, qrels, events = example()
    report = build_report(value, cases, qrels, events)
    assert report["status"] == "SYNTHETIC_FULL160_COMPONENT_CHECK"
    assert report["metrics"]["D"]["mrr@10"] == pytest.approx(.2)
    assert report["metrics"]["H"]["mrr@10"] == 1
    assert report["metrics"]["D"]["recall@5"] == 1  # 20 duplicate-document chunks collapsed first.
    assert report["context_hit_at_chunk14"]["D"] == 0 and report["context_hit_at_chunk14"]["H"] == 1
    assert report["paired_differences"]["H-DR:mrr@10"]["ci95"] == pytest.approx([.8, .8])
    assert report["primary"]["conclusion"] == "PAIRED_IMPROVEMENT"
    assert not report["quality_acceptance_or_default_promotion"]
    assert report["provider_attempts"] == 480 and report["wire_attempt_latency"]["rerank"]["n"] == 320


@pytest.mark.parametrize("bad", ["partial", "qrels", "chunks", "collapse", "context", "receipt", "events", "plan"])
def test_incomplete_or_changed_inputs_never_produce_formal_metrics(bad):
    value, cases, qrels, events = example()
    if bad == "partial":
        cases.pop()
    elif bad == "qrels":
        qrels.pop(next(iter(qrels)))
    elif bad == "chunks":
        cases[0]["result"]["chunk_rankings"]["D"].pop()
    elif bad == "collapse":
        cases[0]["result"]["document_rankings"]["D"] = []
    elif bad == "context":
        cases[0]["result"]["top14_chunk_ids"] = []
    elif bad == "receipt":
        cases[0]["calls"][0]["response_sha256"] = "changed"
    elif bad == "events":
        events.pop()
    else:
        cases[0]["plan_sha256"] = "changed"
    if cases:
        cases[0]["result_sha256"] = evaluation.wire.digest(cases[0]["result"])
    with pytest.raises(ValueError):
        build_report(value, cases, qrels, events)


def test_equal_arms_cannot_be_promoted():
    value, cases, qrels, events = example()
    for artifact in cases:
        row = artifact["result"]
        row["chunk_rankings"]["H"] = list(row["chunk_rankings"]["D"])
        row["document_rankings"]["H"] = list(row["document_rankings"]["D"])
        row["top14_chunk_ids"] = row["chunk_rankings"]["H"][:14]
        artifact["result_sha256"] = evaluation.wire.digest(row)
    assert build_report(value, cases, qrels, events)["primary"]["conclusion"] == "IMPROVEMENT_NOT_ESTABLISHED"


def approval_for(prepared):
    return {"human_approval_reference": "SYNTHETIC_TEST_NOT_REAL_CONSENT", "prepared_sha256": runner.sha(prepared),
            "scope": "DEV160_RETRIEVAL_ONLY", "max_provider_calls": 480, "accounting_limit_cny": "35",
            "accept_reference_estimate": True, "current_prices_reconfirmed": True}


@pytest.mark.parametrize("bad", ["hash", "scope", "calls", "budget", "risk", "prices", "extra"])
def test_exact_approval_gates(tmp_path, bad):
    prepared, approval = tmp_path / "prepared.json", tmp_path / "approval.json"
    prepared.write_text("{}", encoding="utf-8")
    value = approval_for(prepared)
    changes = {"scope": ("scope", "OTHER"), "calls": ("max_provider_calls", 481),
               "budget": ("accounting_limit_cny", "36"), "risk": ("accept_reference_estimate", False),
               "prices": ("current_prices_reconfirmed", False), "extra": ("extra", True)}
    if bad in changes:
        key, replacement = changes[bad]
        value[key] = replacement
    approval.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.validate_approval(approval, "wrong" if bad == "hash" else runner.sha(approval), prepared)


def test_host_guard_runs_before_approval_or_credentials(monkeypatch):
    def forbidden():
        raise RuntimeError("container_required")
    monkeypatch.setattr(runner, "require_container", forbidden)
    monkeypatch.setattr(runner, "read_json", lambda *a: pytest.fail("No assets/credentials before container guard"))
    with pytest.raises(RuntimeError, match="container_required"):
        runner.execute("unused", "unused", "unused")


@pytest.mark.parametrize("cpu,memory", [("max 100000", "5368709120"), ("500000 100000", "5368709120"),
                                       ("400000 100000", "max"), ("400000 100000", "5368709121")])
def test_unbounded_or_oversized_cgroups_are_rejected(monkeypatch, cpu, memory):
    original_file, original_text = Path.is_file, Path.read_text
    monkeypatch.setattr(Path, "is_file", lambda p: True if p.as_posix() == "/.dockerenv" else original_file(p))
    def text(path, *a, **kw):
        if path.as_posix() == "/sys/fs/cgroup/cpu.max":
            return cpu
        if path.as_posix() == "/sys/fs/cgroup/memory.max":
            return memory
        return original_text(path, *a, **kw)
    monkeypatch.setattr(Path, "read_text", text)
    with pytest.raises(RuntimeError):
        runner.require_container()


def test_no_real_transport_without_mode_flag_and_reference(reference_asset, tmp_path, monkeypatch):  # noqa: F811
    value = evaluation.build_plan([(f"DEV_SYNTH_{i:03d}", "word") for i in range(160)], "a" * 64,
                                  "SYNTHETIC_ONLY", tokenizer_path=reference_asset, planned_mode="live")
    body = {"model": "text-embedding-v4", "input": ["word"], "dimensions": 1024, "encoding_format": "float"}
    seen, factories = [], []
    def factory(**kwargs):
        factories.append(kwargs)
        return providers(seen)["embedding"]  # Never a real connection in this test.
    monkeypatch.setattr(evaluation.httpx, "HTTPTransport", factory)
    with pytest.raises(ValueError):
        evaluation.make_client(value, tmp_path / "wire", "DEV_SYNTH_000:E", KEYS["embedding"], body, delegate=None)
    assert not factories
    with evaluation.make_client(value, tmp_path / "wire", "DEV_SYNTH_000:E", KEYS["embedding"], body,
                                delegate=None, allow_network=True) as client:
        client.embeddings.create(**body)
    assert factories == [{"retries": 0, "trust_env": False}] and len(seen) == 1


def test_isolated_runner_prepare_stop_resume_uses_real_sql_chroma_and_mock_sdk(runtime, reference_asset, tmp_path, monkeypatch):  # noqa: F811
    # Real tiny archival-style stores; production counts remain fixed, patched only for this fixture.
    snapshot, dataset, output = tmp_path / "snapshot", tmp_path / "dataset", tmp_path / "bundle"
    snapshot.mkdir()
    dataset.mkdir()
    engine = evaluation.hybrid.engine
    with Session(engine) as session:
        for document in session.exec(select(Document)).all():
            document.chunk_count = 1
            session.add(document)
        session.commit()
        rows = session.exec(select(DocumentChunk).order_by(DocumentChunk.embedding_id)).all()
    shutil.copy2(engine.url.database, snapshot / "runtime.db")
    client = chromadb.PersistentClient(path=str(snapshot / "chroma"))
    collection = client.create_collection("local_doc_rag", embedding_function=None)
    # Distinct distances: identical vectors leave Chroma tie order unspecified on reopen.
    collection.add(ids=[row.embedding_id for row in rows],
                   embeddings=[[1. + i / 100] + [1.] * 1023 for i in range(len(rows))],
                   documents=[row.content for row in rows], metadatas=[{"tenant_id": evaluation.TENANT,
                   "category": evaluation.CATEGORY, "source_type": "archival_benchmark"} for _ in rows])
    client.close()
    (snapshot / "import_receipt.json").write_text(json.dumps({"managed_records_sha256": runner.MANAGED_SHA,
        "all_imported_vectors_bodies_metadata_readback_equal": True, "documents": 120, "chunks": 120,
        "dimension": 1024, "index_profile": "techqa_archival_content_vectors_managed_ids_v1"}), encoding="utf-8")
    ids = [f"DEV_SYNTH_{i:03d}" for i in range(160)] + [f"TRAIN_SYNTH_{i:03d}" for i in range(450)]
    pq.write_table(pa.table({"_id": ids, "text": ["needle"] * 610}), dataset / "queries.parquet")
    pq.write_table(pa.table({"query-id": ids, "corpus-id": ["d119"] * 610, "score": [1] * 610}), dataset / "qrels.parquet")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"retrieval_dataset": {
        "queries_file": "queries.parquet", "qrels_file": "qrels.parquet",
        "queries_sha256": runner.sha(dataset / "queries.parquet"), "qrels_sha256": runner.sha(dataset / "qrels.parquet")}}), encoding="utf-8")
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    monkeypatch.setattr(runner, "DOCS", 120)
    monkeypatch.setattr(runner, "CHUNKS", 120)
    before = runner.inventory(snapshot)
    prepared = runner.prepare(dataset, snapshot, reference_asset, output)
    assert prepared["status"] == "PREPARED_NO_PROVIDER_CALLS" and runner.inventory(snapshot) == before
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(approval_for(output / "prepared.json")), encoding="utf-8")
    monkeypatch.setattr(runner, "require_container", lambda: None)  # Synthetic Windows test only.
    monkeypatch.setenv("DASHSCOPE_API_KEY", KEYS["embedding"])
    monkeypatch.setenv("DASHSCOPE_RERANK_API_KEY", KEYS["rerank"])
    original_client, original_save = evaluation.make_client, evaluation.save_case
    seen = []
    delegates = providers(seen)
    def offline_client(*args, **kwargs):
        stage = args[0]["contract"]["calls"][args[2]]
        kwargs.update(delegate=delegates[stage], allow_network=False)
        return original_client(*args, **kwargs)
    monkeypatch.setattr(evaluation, "make_client", offline_client)
    def stop_at(limit):
        def save(*args, **kwargs):
            if args[1] >= limit:
                raise RuntimeError("synthetic_interruption")
            return original_save(*args, **kwargs)
        return save
    for limit, count in ((2, 6), (3, 9)):
        monkeypatch.setattr(evaluation, "save_case", stop_at(limit))
        with pytest.raises(RuntimeError, match="synthetic_interruption"):
            runner.execute(output, approval, runner.sha(approval))
        assert len(seen) == count and runner.inventory(snapshot) == before
        assert not (output / "execute.lock").exists() and not (output / "report.json").exists()
    assert len(list((output / "run/cases").glob("*.json"))) == 3
    assert not list(tmp_path.glob("phase1_owned_copy_*"))
