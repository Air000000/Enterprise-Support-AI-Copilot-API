"""Explicit serial phase-2 HTTP / blind-review runner. Default CLI only prints help.

Run in a dedicated <=4CPU/5GiB container, with read-only source mounts. Never
mount .env, use the daily API, or interpret an approval JSON as human consent.
"""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from functools import partial
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI

from experiments.evals import runtime_qa as qa
from experiments.evals import runtime_retrieval_inputs as inputs
from experiments.evals import runtime_retrieval_runner as assets
from experiments.evals import runtime_transport as wire

TENANT, CATEGORY = "tenant_eval_techqa_archival", "other"
_STRICT_USAGE = wire.usage_record


def service_usage_record(stage, payload):
    """Account only a fully received, at-limit generation truncation; never accept it."""
    choices = payload.get("choices")
    if stage != "generation" or not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "length":
        return _STRICT_USAGE(stage, payload)
    choice, usage = choices[0], payload["usage"]
    message = choice.get("message", {})
    if (choice.get("index") != 0 or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or usage.get("completion_tokens") != wire.STAGES[stage][3]):
        raise ValueError("Unknown incomplete generation envelope")
    # Validate unchanged usage with the strict validator, retaining the actual finish reason.
    checked = _STRICT_USAGE(stage, payload | {"choices": [choice | {"finish_reason": "stop"}]})
    return checked | {"finish_reason": "length"}


def keep(path, value):
    """Write-once checkpoint; changed/partial observations require an audit."""
    path = Path(path)
    if path.exists():
        if assets.read_json(path) != value:
            raise RuntimeError("Checkpoint content/identity changed")
    else:
        wire.write_json(path, value)
    return value


def code_identity():
    root = Path(__file__).resolve().parents[2]
    paths = list(root.glob("*.py")) + [root / "alembic.ini"]
    for folder in ("models", "routers", "schemas", "services", "rag_runtime", "migrations", "experiments/evals", "scripts"):
        paths.extend((root / folder).rglob("*.py"))
    paths.append(root / "experiments/evals/reports/runtime_eval_phase0/qa_protocol_2026-10-23.md")
    paths.append(assets.MANIFEST)
    return {p.relative_to(root).as_posix(): assets.sha(p) for p in sorted(set(paths)) if p.is_file()}


def identity():
    return dict(sources=code_identity(), python=sys.version,
        dependencies={n: importlib.metadata.version(n) for n in ("openai", "httpx", "chromadb", "sqlmodel", "bm25s", "tokenizers", "uvicorn", "alembic", "numpy", "scipy", "fastapi", "pydantic", "h11")},
        profiles=qa.PROFILES, caps={k: str(v) for k, v in qa.CAPS.items()}, endpoints=qa.ENDPOINTS,
        judge_model=qa.JUDGE_MODEL, evidence_prompt=qa.EVIDENCE_PROMPT, answer_prompt=qa.ANSWER_PROMPT)


def load_questions(path):
    spec = assets.read_json(assets.MANIFEST)["generation_dataset"]
    if assets.sha(path) != spec["metadata_sha256"]:
        raise ValueError("Frozen generation metadata hash mismatch; no download/fallback")
    rows = assets.read_json(path)
    if not isinstance(rows, list) or len(rows) != 910 or len({r["id"] for r in rows}) != 910:
        raise ValueError("Frozen metadata count/IDs differ")
    selected = []
    for r in rows:
        if not isinstance(r["id"], str) or not r["id"].startswith(("TRAIN_", "DEV_")) or type(r["is_impossible"]) is not bool:
            raise ValueError("Unexpected split/stratum")
        if r["id"].startswith("DEV_"):
            if re.fullmatch(r"DEV_[A-Za-z0-9_]{1,90}", r["id"]) is None or not isinstance(r["question"], str) or not r["question"].strip():
                raise ValueError("Invalid question")
            selected.append(dict(id=r["id"], question=r["question"].rstrip(),
                                 stratum="impossible" if r["is_impossible"] else "answerable"))
    if len(selected) != 310 or sum(r["stratum"] == "answerable" for r in selected) != 160:
        raise ValueError("Full DEV310/160/150 required")
    return sorted(selected, key=lambda r: r["id"])


def prepare(metadata, snapshot, tokenizer, output):
    runtime_identity = identity()  # Check actual runtime dependencies before scanning large read-only assets.
    output, snapshot, metadata, tokenizer = map(lambda p: Path(p).resolve(), (output, snapshot, metadata, tokenizer))
    if output.exists() or any(output == p or output in p.parents or p in output.parents for p in (snapshot, metadata, tokenizer)):
        raise ValueError("New output path disjoint from source assets required")
    before = assets.inventory(snapshot)
    assets.verify_snapshot(snapshot)
    questions = load_questions(metadata)
    profile = inputs.reference_profile(tokenizer)
    for row in questions:
        from schemas.rag import RagAskRequest
        RagAskRequest(question=row["question"])
        if not inputs.estimate_input("embedding", {"input": [row["question"]]}, profile)["within_reference_profile"]:
            raise ValueError("Full question exceeds embedding reference profile")
    if assets.inventory(snapshot) != before:
        raise ValueError("Snapshot changed")
    plan = dict(identity=runtime_identity, questions=questions, snapshot_root=str(snapshot), metadata=str(metadata),
                metadata_sha256=assets.sha(metadata), snapshot_inventory=before, reference_tokenizer=profile,
                full_reserve_cny=qa.full_reserve(), scope="DEV310_HTTP_AND_TWO_BLIND_REVIEW_PHASES")
    output.mkdir()
    keep(output / "prepared.json", plan)
    return dict(status="PREPARED_NO_PROVIDER_CALLS", prepared_sha256=assets.sha(output / "prepared.json"),
                max_provider_calls=3100, accounting_limit_cny=str(sum(qa.CAPS.values())), full_reserve_cny=qa.full_reserve())


def approval(root, path, expected_sha):
    if assets.sha(path) != expected_sha:
        raise ValueError("Approval fingerprint differs")
    value, plan = assets.read_json(path), assets.read_json(root / "prepared.json")
    expected = dict(prepared_sha256=assets.sha(root / "prepared.json"), scope=plan["scope"],
        max_provider_calls=3100, accounting_limits_cny={k: str(v) for k, v in qa.CAPS.items()},
        accept_reference_estimate=True, current_prices_reconfirmed=True,
        allow_resume_completed_receipts_only=True)
    if (set(value) != set(expected) | {"human_approval_reference"}
            or any(value.get(k) != v or type(value.get(k)) is not type(v) for k, v in expected.items())
            or not isinstance(value["human_approval_reference"], str) or not value["human_approval_reference"].strip()):
        raise ValueError("New explicit phase-2 scope/budget/price approval required")
    # JSON round-trip normalizes tuple-valued profile constants.
    if plan["identity"] != json.loads(json.dumps(identity())):
        raise ValueError("Frozen code/config/dependencies changed")
    if inputs.reference_profile(plan["reference_tokenizer"]["path"]) != plan["reference_tokenizer"]:
        raise ValueError("Reference tokenizer changed")
    return plan, value["human_approval_reference"]


def contract(reference, stages, cid):
    return dict(authorization_reference=reference, stages={k: dict(base_url=v, max_request_bytes=32768 if k == "embedding" else 2*1024**2)
                for k, v in stages.items()}, calls={f"{cid}:{s}": s for s in stages}, max_total_request_bytes=8*1024**2)


class Book:
    """Verified receipts in RAM while one execution lock is held; reload on restart."""

    def __init__(self):
        self.rows = {}

    def add(self, event, kind=None):
        stage = kind or event["stage"]
        row = (stage, event["prompt_tokens"], event["completion_tokens"], event["request_sha256"], event["response_sha256"])
        qa.charge(stage, row[1], row[2])
        cid = event["call_id"]
        if cid in self.rows and self.rows[cid] != row:
            raise RuntimeError("Completed receipt changed")
        self.rows[cid] = row
        self.check()

    def check(self, next_stage=None):
        spent = {g: sum((qa.charge(r[0], r[1], r[2]) for r in self.rows.values() if qa.group(r[0]) == g), qa.Decimal(0)) for g in qa.CAPS}
        if any(sum(r[0] == s for r in self.rows.values()) > spec[4] for s, spec in qa.PROFILES.items()):
            raise RuntimeError("Call count exceeded")
        if next_stage is not None:
            if sum(r[0] == next_stage for r in self.rows.values()) >= qa.PROFILES[next_stage][4]:
                raise RuntimeError("No approved call slots remain")
            pi, po, *_ = qa.PROFILES[next_stage]
            spent[qa.group(next_stage)] += qa.charge(next_stage, pi, po)
        if any(spent[g] > qa.CAPS[g] for g in spent):
            raise RuntimeError("Accounting cap/reservation exceeded; not an invoice guarantee")
        return {g: str(v) for g, v in spent.items()}


class Transport(wire.RuntimeEvalTransport):
    def __init__(self, *args, book, validate, kind=None, profile=None, **kwargs):
        self.book, self.validate, self.kind, self.profile = book, validate, kind, profile
        super().__init__(*args, **kwargs)

    def handle_request(self, request):
        try:
            with patch.object(wire, "usage_record", service_usage_record if self.kind is None else _STRICT_USAGE):
                response = self._handle(request)
        except Exception:
            keep(self.root.parent / "halt.json", {"status": "AUDIT_REQUIRED_NO_AUTOMATIC_RETRY"})
            raise
        if self.stage == "generation" and response.json()["choices"][0]["finish_reason"] == "length":
            keep(self.root.parent / "generation_failure.json", dict(
                reason="generation_output_truncated", call_id=self.call_id, usage_recorded=True))
            response.close()
            # Outside the unknown-outcome handler: HTTP failure, no partial answer or resend.
            raise RuntimeError("Known generation truncation; accounted failure, never retry")
        return response

    def _handle(self, request):
        self._check_files()
        if (self.root / "pending.json").exists() or (self.root / "transport.lock").exists():
            raise RuntimeError("Unresolved pending/lock; audit before any restoration/send")
        body = json.loads(request.content)
        wire.validate_body(self.stage, body)
        self.validate(body)
        estimate(self.kind or self.stage, body, self.profile)
        events = self._history()
        for e in events:
            self.book.add(e, self.kind)
        old = next((e for e in events if e["call_id"] == self.call_id), None)
        if old:
            if (wire.digest(body) != old["request_sha256"] or request.method != "POST"
                    or str(request.url) != self.contract["stages"][self.stage]["base_url"] + wire.STAGES[self.stage][1]):
                raise RuntimeError("Replay request differs")
            saved = assets.read_json(self.root / old["response_file"])["wire_utf8"]
            return httpx.Response(200, content=saved.encode(), headers={"content-type": "application/json"})
        expected = list(self.contract["calls"])
        if [e["call_id"] for e in events] != expected[:len(events)] or self.call_id != expected[len(events)]:
            raise RuntimeError("Cannot skip or reorder a paid stage")
        self.book.check(self.kind or self.stage)
        response = super().handle_request(request)
        try:
            for e in self._history():
                self.book.add(e, self.kind)
            payload = response.json()
            if self.stage in {"embedding", "rerank"}:
                from experiments.evals.runtime_retrieval import validate_response
                validate_response(self.stage, body, payload)
        except Exception:
            keep(self.root / "pending.json", {"call_id": self.call_id, "status": "audit_required"})
            raise
        return response


def estimate(stage, body, profile):
    if profile is None:
        return  # Synthetic tests only; live approval requires a hash-bound reference profile.
    if stage in {"embedding", "rerank"}:
        result = inputs.estimate_input(stage, body, profile)
        if not result["within_reference_profile"]:
            raise ValueError("Input exceeds reference profile; never truncate/skip")
        return result["planning_tokens"]
    if inputs.reference_profile(profile["path"]) != profile:
        raise ValueError("Reference profile differs")
    tokenizer = inputs._tokenizer(profile["path"], profile["sha256"])
    raw = sum(len(v.ids) for v in tokenizer.encode_batch([m["content"] for m in body["messages"]], add_special_tokens=False))
    planning = (raw*5+3)//4 + 128
    if planning > qa.PROFILES[stage][0]:
        raise ValueError("Full chat context exceeds reference reservation; no truncation")
    return planning


def client_factory(c, root, book, keys, validate, profile, *, delegates=None, kind=None, allow_network=False):
    if delegates is None and (allow_network is not True or profile is None):
        raise ValueError("Offline delegates required unless live mode and reference profile explicitly supplied")
    if set(keys) < set(c["stages"]) or any(not isinstance(keys.get(s), str) or not keys[s].strip() for s in c["stages"]):
        raise ValueError("Explicit stage credentials required; no environment fallback")
    clients = []
    def factory(stage):
        cid = next(cid for cid, s in c["calls"].items() if s == stage)
        delegate = delegates[stage] if delegates is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        if delegates is not None and not isinstance(delegate, httpx.MockTransport):
            raise ValueError("Offline delegates must be MockTransport")
        transport = Transport(c, root, cid, delegate, book=book, validate=partial(validate, stage), kind=kind, profile=profile)
        cli = OpenAI(api_key=keys[stage], base_url=c["stages"][stage]["base_url"], organization="", project="",
            max_retries=0, timeout=wire.STAGES[stage][2], http_client=httpx.Client(transport=transport,
            trust_env=False, follow_redirects=False, timeout=wire.STAGES[stage][2]))
        if stage == "embedding":
            cli.embeddings.create = partial(cli.embeddings.create, dimensions=1024)
        elif stage == "generation":
            cli.chat.completions.create = partial(cli.chat.completions.create, max_tokens=1024, extra_body={"enable_thinking": False})
        clients.append(cli)
        return cli
    return factory, clients


def replay_history(root, book):
    for folder in ("service", "evidence", "answer"):
        for contract_path in sorted((root / folder).glob("*/contract.json")):
            case = contract_path.parent
            path = case / "wire/identity.json"
            if (case / "halt.json").exists():
                raise RuntimeError("Halt marker requires explicit audit, not automatic retry")
            if not path.is_file():
                # A crash before SDK construction is known-unsent; no result may claim completion.
                if (case / "result.json").exists():
                    raise RuntimeError("Completed case missing wire identity")
                continue
            c = assets.read_json(contract_path)
            with qa.wire_profile(folder != "service"), patch.object(
                    wire, "usage_record", service_usage_record if folder == "service" else _STRICT_USAGE):
                transport = wire.RuntimeEvalTransport(c, path.parent, next(iter(c["calls"])), httpx.MockTransport(lambda r: None))
                try:
                    if (path.parent / "pending.json").exists() or (path.parent / "transport.lock").exists():
                        raise RuntimeError("Unknown attempt/old lock requires audit; no auto clearing")
                    for e in transport._history():
                        book.add(e, None if folder == "service" else folder)
                finally:
                    transport.close()


@contextmanager
def no_provider_fallback():
    """Even a late server worker cannot escape the per-case evaluated clients."""
    from rag_runtime import build_rag_index, hybrid_ask, hybrid_search, query_chroma, query_rag_chroma
    def denied(*args, **kwargs):
        raise RuntimeError("Provider/index access outside the active evaluated case is forbidden")
    with ExitStack() as stack:
        for mod, name in ((build_rag_index, "get_client"), (hybrid_search, "get_rerank_client"),
                (hybrid_ask, "get_classifier_client"), (hybrid_ask, "get_llm_client"),
                (query_rag_chroma, "get_llm_client"), (hybrid_search, "get_collection"), (query_chroma, "get_collection")):
            stack.enter_context(patch.object(mod, name, denied))
        yield


def http_case(client, row, arm, root, reference, book, keys, collection, *, profile=None, delegates=None, allow_network=False):
    """Actual /rag/ask, actual JWT, captured selection and stage-by-stage SDK replay."""
    from auth import CurrentUser, create_access_token
    from rag_runtime import build_rag_index as embeddings, hybrid_ask, hybrid_search as hybrid
    from rag_runtime import query_chroma as dense, query_rag_chroma as ask_dense
    from experiments.evals.refusal_evidence_sufficiency import ClassifierInput, ContextSource

    root.mkdir(exist_ok=True)
    request = dict(question=row["question"], category=CATEGORY, retrieval_mode="dense" if arm == "D" else "hybrid", top_k=3 if arm == "D" else 14)
    request.update({"max_distance": .9} if arm == "D" else {"refusal_policy": "v5_3_1_experimental"})
    case_id = f"{row['id']}:{arm}"
    keep(root / "input.json", dict(id=row["id"], arm=arm, request=request, reference=reference))
    stages = {s: qa.ENDPOINTS[s] for s in (("embedding", "generation") if arm == "D" else ("embedding", "rerank", "classifier", "generation"))}
    c = keep(root / "contract.json", contract(reference, stages, case_id))
    result_path = root / "result.json"
    if (root / "halt.json").exists():
        raise RuntimeError("Halt requires explicit audit")
    if result_path.exists():
        saved = assets.read_json(result_path)
        if (saved["observation_sha256"] != wire.digest({k: v for k, v in saved.items() if k != "observation_sha256"})
                or saved["context_sha256"] != (assets.sha(root / "context.json") if (root / "context.json").exists() else None)
                or saved["selection_sha256"] != (assets.sha(root / "selection.json") if (root / "selection.json").exists() else None)):
            raise RuntimeError("Completed HTTP observation changed")
        return saved
    replayed = (root / "started.json").exists()
    if replayed:
        if assets.read_json(root / "started.json")["request_sha256"] != wire.digest(request):
            raise ValueError("Started HTTP request changed")
    else:
        keep(root / "started.json", {"request_sha256": wire.digest(request),
             "first_started_utc": datetime.now(timezone.utc).isoformat()})
    sources = []
    original_search = ask_dense.search_chroma if arm == "D" else hybrid_ask.search_hybrid

    def query(**kwargs):
        path = root / "selection.json"
        query_hash = wire.digest(kwargs)
        if path.exists():
            saved = assets.read_json(path)
            if saved["query_sha256"] != query_hash or saved["result_sha256"] != wire.digest(saved["result"]):
                raise RuntimeError("Frozen retrieval selection changed")
            return saved["result"]
        raw = collection.query(**kwargs)
        value = {k: raw[k] for k in (("ids", "documents", "metadatas", "distances") if arm == "D" else ("ids", "distances"))}
        keep(path, dict(query_sha256=query_hash, result=value, result_sha256=wire.digest(value)))
        return value

    def search(*args, **kwargs):
        selected = original_search(*args, **kwargs)
        visible = hybrid.load_visible_chunks(TENANT, CATEGORY, embedding_ids=[s.chunk_id for s in selected]) if selected else {}
        if any(s.chunk_id not in visible or s.content != visible[s.chunk_id].content or s.tenant_id != TENANT
               or s.document_id != visible[s.chunk_id].document_id or s.category != CATEGORY for s in selected):
            raise RuntimeError("Selected full context differs from authorized SQL snapshot")
        sources[:] = selected
        keep(root / "context.json", {"question": row["question"], "sources": [
            dict(source_id=f"Source {i}", content=s.content) for i, s in enumerate(sources, 1)]})
        return selected

    def validate(stage, body):
        if stage == "embedding":
            if body["input"] != [row["question"]]:
                raise ValueError("Embedding is not the frozen generation question")
        elif stage == "rerank":
            from experiments.evals.rerankers.qwen3_reranker import DEFAULT_RERANK_INSTRUCTION
            # Service constructs these from full current SQL-visible candidates, not metadata previews.
            candidate_path = root / "rerank_input.json"
            if body["query"] != row["question"] or body["instruct"] != DEFAULT_RERANK_INSTRUCTION:
                raise ValueError("Rerank question/instruction differs")
            if not candidate_path.exists() or assets.read_json(candidate_path) != body:
                raise ValueError("Rerank differs from captured eligible candidates")
        elif stage == "classifier":
            expected = hybrid_ask.POLICY_BUILDERS["v5_3_1_experimental"](ClassifierInput(row["question"], tuple(
                ContextSource(f"Source {i}", s.chunk_id, s.document_id, s.content) for i, s in enumerate(sources, 1))))
            if body["messages"] != expected:
                raise ValueError("Classifier differs from full selected context")
        elif arm == "D":
            expected = f"Question:\n{row['question']}\n\nContext:\n{ask_dense.build_context(sources)}\n\n请基于 Context 回答 Question。".strip()
            if body["messages"][1]["content"] != expected:
                raise ValueError("Dense generation differs from full context")
        else:
            payload = json.loads(body["messages"][1]["content"])
            expected = assets.read_json(root / "context.json")
            if (set(payload) != {"question", "sources", "allowed_citation_ids"}
                    or {k: payload[k] for k in expected} != expected
                    or not isinstance(payload["allowed_citation_ids"], list)
                    or not set(payload["allowed_citation_ids"]) <= {s["source_id"] for s in expected["sources"]}):
                raise ValueError("Hybrid generation differs from full context")

    rerank_original = hybrid.rerank_candidates
    def rerank(query, candidates, **kwargs):
        from experiments.evals.rerankers.qwen3_reranker import DEFAULT_RERANK_INSTRUCTION
        expected = hybrid.load_visible_chunks(TENANT, CATEGORY, embedding_ids=[s.chunk_id for s in candidates])
        if any(s.chunk_id not in expected or s.content != expected[s.chunk_id].content for s in candidates):
            raise ValueError("Rerank source scope mismatch")
        keep(root / "rerank_input.json", dict(model="qwen3-rerank", query=query, documents=[s.content for s in candidates],
             top_n=len(candidates), instruct=DEFAULT_RERANK_INSTRUCTION))
        return rerank_original(query, candidates, **kwargs)

    def guarded(function):
        def call(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception:
                keep(root / "halt.json", {"status": "AUDIT_REQUIRED_NO_AUTOMATIC_RETRY"})
                raise
        return call

    factory, clients = client_factory(c, root / "wire", book, keys, validate, profile, delegates=delegates, allow_network=allow_network)
    try:
        with ExitStack() as stack:
            for mod, name, value in ((dense, "get_collection", lambda: SimpleNamespace(query=guarded(query))),
                    (hybrid, "get_collection", lambda: SimpleNamespace(query=guarded(query))),
                    (embeddings, "get_client", lambda: factory("embedding")),
                    (hybrid, "get_rerank_client", lambda: factory("rerank")),
                    (hybrid, "rerank_candidates", guarded(rerank)),
                    (ask_dense if arm == "D" else hybrid_ask, "search_chroma" if arm == "D" else "search_hybrid", guarded(search)),
                    (ask_dense, "get_llm_client", lambda: factory("generation")),
                    (hybrid_ask, "get_llm_client", lambda: factory("generation")),
                    (hybrid_ask, "get_classifier_client", lambda: factory("classifier"))):
                stack.enter_context(patch.object(mod, name, value))
            started = time.perf_counter()
            token = create_access_token(CurrentUser(sub="phase2-eval", user_id="phase2-eval", tenant_id=TENANT, role="eval"))
            response = client.post("/rag/ask", json=request, headers={"Authorization": "Bearer " + token})
            elapsed = round((time.perf_counter()-started)*1000)
            # Known HTTP failures are retained; wire-unknown failures stop the whole run.
            if (root / "wire/pending.json").exists() or (root / "halt.json").exists():
                raise RuntimeError("Unknown model outcome; stop for audit")
            body = response.json() if response.status_code == 200 else {}
            if response.status_code == 200:
                if not (root / "context.json").exists() or [s["chunk_id"] for s in body["sources"]] != [s.chunk_id for s in sources]:
                    raise RuntimeError("HTTP sources and full captured context disagree")
            value = dict(http_status=response.status_code, body=body, latency_ms=None if replayed else elapsed,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                reconstructed_after_interruption=replayed,
                context_sha256=assets.sha(root / "context.json") if (root / "context.json").exists() else None,
                selection_sha256=assets.sha(root / "selection.json") if (root / "selection.json").exists() else None)
            return keep(result_path, value | {"observation_sha256": wire.digest(value)})
    finally:
        for cli in clients:
            cli.close()


def migrate_copy(source, work):
    """Never upgrade the archival SQL file: fresh schema, then only two source tables."""
    from alembic import command
    from alembic.config import Config
    command.upgrade(Config("alembic.ini"), "head")
    with sqlite3.connect(work / "runtime.db") as target, assets.readonly(source / "runtime.db") as origin:
        for table in ("documents", "document_chunks"):
            columns = [r[1] for r in origin.execute(f"PRAGMA table_info({table})")]
            if set(columns) != {r[1] for r in target.execute(f"PRAGMA table_info({table})")}:
                raise ValueError("Archival/current source table schema changed")
            placeholders = ",".join("?" for _ in columns)
            names = ",".join('"'+c+'"' for c in columns)
            target.executemany(f"INSERT INTO {table} ({names}) VALUES ({placeholders})", origin.execute(f"SELECT {names} FROM {table}"))


def service_fingerprints(root):
    return {p.relative_to(root / "service").as_posix(): assets.sha(p)
            for p in sorted((root / "service").rglob("*.json"))}


def execute_service(root, plan, reference, keys):
    if "database" in sys.modules:
        raise RuntimeError("Fresh process required before database import")
    if (root / "service_complete.json").exists():
        raise RuntimeError("Service already completed; no rerun")
    source, work = Path(plan["snapshot_root"]), root / "owned_work"
    if not os.statvfs(source).f_flag & os.ST_RDONLY:
        raise ValueError("Read-only archival source mount required")
    if work.is_symlink() or work.resolve().parent != root:
        raise ValueError("Owned work directory must stay inside this run")
    # Snapshot verification imports database via runtime_retrieval; configure it first.
    os.environ.update(DATABASE_URL="sqlite:///"+str(work / "runtime.db"), SQL_ECHO="false", PYTHON_DOTENV_DISABLED="1",
        DASHSCOPE_API_KEY="evaluation-clients-patched-no-env-key", DASHSCOPE_BASE_URL=qa.BEIJING,
        CHAT_MODEL="qwen3.5-plus", DASHSCOPE_MODEL="qwen3.5-plus", EMBEDDING_MODEL="text-embedding-v4",
        JWT_SECRET=secrets.token_hex(32), JWT_ACCESS_TOKEN_EXPIRE_MINUTES="10", AUTH_DEMO_USERS_JSON="",
        ANONYMIZED_TELEMETRY="False", RAG_HYBRID_SEARCH_ENABLED="true", RAG_HYBRID_ASK_ENABLED="true",
        RAG_HYBRID_MAX_VISIBLE_CHUNKS="200000", DOCUMENT_STORAGE_DIR=str(work / "uploads"))
    if assets.inventory(source) != plan["snapshot_inventory"] or load_questions(plan["metadata"]) != plan["questions"]:
        raise ValueError("Approved immutable input identity differs")
    assets.verify_snapshot(source)
    # Persistent native-volume copy is intentional: no Windows bind-mount HNSW/ledger scans.
    work.mkdir(exist_ok=True)
    marker = work / "ready.json"
    if not marker.exists():
        if any(work.iterdir()):
            raise RuntimeError("Incomplete local copy/migration; audit owned_work, do not delete provider receipts")
        shutil.copytree(source / "chroma", work / "chroma")
        for name, spec in plan["snapshot_inventory"].items():
            if name.startswith("chroma/") and assets.sha(work / name) != spec["sha256"]:
                raise ValueError("Working vector copy differs")
        migrate_copy(source, work)
        keep(marker, {"prepared_sha256": assets.sha(root / "prepared.json")})
    elif assets.read_json(marker) != {"prepared_sha256": assets.sha(root / "prepared.json")}:
        raise ValueError("Working copy identity differs")
    import chromadb
    import database
    import main
    from rag_runtime import hybrid_search as hybrid
    from scripts.smoke_hybrid_runtime import live_http_client
    book = Book()
    replay_history(root, book)
    client = chromadb.PersistentClient(path=str(work / "chroma"))
    try:
        collection = client.get_collection("local_doc_rag", embedding_function=None)
        (root / "service").mkdir(exist_ok=True)
        hybrid._bm25_index.cache_clear()
        with no_provider_fallback(), live_http_client(main.app) as http:
            # Local round-trip may include a cold BM25 build and four bounded model stages.
            http.timeout = httpx.Timeout(600)
            for index, row in enumerate(plan["questions"]):
                for arm in (("D", "H") if index % 2 == 0 else ("H", "D")):
                    http_case(http, row, arm, root / "service" / f"{index:03d}-{arm}", reference, book, keys,
                              collection, profile=plan["reference_tokenizer"], allow_network=True)
                print(json.dumps({"completed_pairs": index+1, "total": 310, "accounting_cny_not_invoice": book.check()}), flush=True)
        with sqlite3.connect(work / "runtime.db") as db:
            if any(db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("ticket", "agent_runs", "approval_requests", "tool_calls")):
                raise RuntimeError("Unexpected action side effect")
        if assets.inventory(source) != plan["snapshot_inventory"]:
            raise RuntimeError("Read-only source changed")
        keep(root / "service_complete.json", {"prepared_sha256": assets.sha(root / "prepared.json"),
             "accounting": book.check(), "files": service_fingerprints(root)})
    finally:
        hybrid._bm25_index.cache_clear()
        database.engine.dispose()
        main.llm_client.close()
        client.close()


def packets(root, kind):
    """Evidence projection never contains output, gate, arm, old labels or gold answers."""
    plan = assets.read_json(root / "prepared.json")
    if not (root / "service_complete.json").exists():
        raise ValueError("Complete service capture required before blind labeling")
    if service_fingerprints(root) != assets.read_json(root / "service_complete.json")["files"]:
        raise ValueError("Service observations changed after capture")
    if kind == "answer" and not (root / "evidence_frozen.json").exists():
        raise ValueError("Assistant-reviewed evidence freeze required before unblinding answers")
    tasks, mapping = {}, {}
    for index, row in enumerate(plan["questions"]):
        for arm in ("D", "H"):
            folder = root / "service" / f"{index:03d}-{arm}"
            path = folder / "context.json"
            if kind == "evidence" and not path.exists():
                # Failed before context observation is not an observed empty context.
                mapping[f"{index:03d}-{arm}"] = None
                continue
            packet = assets.read_json(path) if path.exists() else dict(question=row["question"], sources=[])
            if kind == "answer":
                result = assets.read_json(folder / "result.json")
                if result["http_status"] != 200 or result["body"]["answer"].strip() == qa.REFUSAL:
                    continue
                packet = packet | {"answer": result["body"]["answer"]}
            pid = wire.digest(packet)
            tasks[pid] = packet
            mapping[f"{index:03d}-{arm}"] = pid
    tasks = dict(sorted(tasks.items()))
    keep(root / f"{kind}_packets.json", tasks)
    keep(root / f"{kind}_private_mapping.json", mapping)
    return tasks


def execute_judges(root, plan, reference, key, kind):
    tasks = packets(root, kind)
    book = Book()
    replay_history(root, book)
    (root / kind).mkdir(exist_ok=True)
    with qa.wire_profile(True):
        for pid, packet in tasks.items():
            folder = root / kind / pid
            folder.mkdir(exist_ok=True)
            body = qa.judge_body(kind, packet)
            c = keep(folder / "contract.json", contract(reference, {"classifier": qa.BEIJING}, pid+":"+kind))
            if (folder / "result.json").exists():
                qa.validate_verdict(kind, assets.read_json(folder / "result.json"), packet)
                continue
            def validate(stage, actual):
                if actual != body:
                    raise ValueError("Blind judge payload differs")
            factory, clients = client_factory(c, folder / "wire", book, {"classifier": key}, validate,
                                               plan["reference_tokenizer"], kind=kind, allow_network=True)
            try:
                response = factory("classifier").chat.completions.create(**{k: v for k, v in body.items() if k != "enable_thinking"},
                    extra_body={"enable_thinking": False})
                try:
                    verdict = qa.parse_verdict(kind, response.choices[0].message.content, packet)
                except Exception:
                    keep(folder / "halt.json", {"status": "AUDIT_REQUIRED_NO_AUTOMATIC_RETRY"})
                    raise
                keep(folder / "result.json", verdict)
            finally:
                for cli in clients:
                    cli.close()
            print(json.dumps({"judge_phase": kind, "completed_unique": len(list((root / kind).glob("*/result.json"))),
                              "accounting_cny_not_invoice": book.check()}), flush=True)


def safe_error_chain(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        frames, tb = [], error.__traceback__
        while tb is not None:
            frames.append([Path(tb.tb_frame.f_code.co_filename).name, tb.tb_frame.f_code.co_name, tb.tb_lineno])
            tb = tb.tb_next
        chain.append({"type": type(error).__name__, "frames": frames})
        error = error.__cause__ or error.__context__
    return chain


def freeze(root, kind, reviewed, reviewer):
    tasks = packets(root, kind)
    values = json.loads(Path(reviewed).read_text(encoding="utf-8"), object_pairs_hook=qa._unique_object)
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(values, dict) or set(values) != set(tasks):
        raise ValueError("All unique packets need explicit assistant review and reviewer reference")
    for pid, packet in tasks.items():
        # Require completed original AI result; no silent replacement of failed judge calls.
        qa.validate_verdict(kind, assets.read_json(root / kind / pid / "result.json"), packet)
        qa.validate_verdict(kind, values[pid], packet)
    keep(root / f"{kind}_frozen.json", dict(reviewer=reviewer, packet_sha256=wire.digest(tasks),
         reviewed_sha256=wire.digest(values), values=values))
    return {"status": "FROZEN", "kind": kind, "unique_packets": len(tasks)}


def report(root):
    plan = assets.read_json(root / "prepared.json")
    reviews, mappings = {}, {}
    for kind in ("evidence", "answer"):
        tasks = packets(root, kind)
        frozen = assets.read_json(root / f"{kind}_frozen.json")
        if frozen["packet_sha256"] != wire.digest(tasks) or frozen["reviewed_sha256"] != wire.digest(frozen["values"]) or set(frozen["values"]) != set(tasks):
            raise ValueError("Reviewed packet freeze changed")
        for pid, packet in tasks.items():
            qa.validate_verdict(kind, frozen["values"][pid], packet)
        reviews[kind], mappings[kind] = frozen["values"], assets.read_json(root / f"{kind}_private_mapping.json")
    cases = []
    for index, row in enumerate(plan["questions"]):
        pair = dict(id=row["id"], stratum=row["stratum"], arms={}, latency={})
        for arm in ("D", "H"):
            key = f"{index:03d}-{arm}"
            result = assets.read_json(root / "service" / key / "result.json")
            pid = mappings["evidence"][key]
            evidence = reviews["evidence"][pid] if pid is not None else dict(verdict="AMBIGUOUS", context_available=False)
            verdict = reviews["answer"][mappings["answer"][key]] if key in mappings["answer"] else None
            pair["arms"][arm] = qa.score(result, evidence, verdict)
            pair["latency"][arm] = result["latency_ms"]
        cases.append(pair)
    book = Book()
    replay_history(root, book)
    value = qa.paired_report(cases) | dict(accounting_cny_not_invoice=book.check(), completed_unique_provider_calls=len(book.rows),
                                          prepared_sha256=assets.sha(root / "prepared.json"))
    keep(root / "report.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action")
    prep = sub.add_parser("prepare", help="Explicit local DEV read; no external model calls")
    for name in ("metadata", "snapshot", "tokenizer", "output"):
        prep.add_argument("--"+name, required=True)
    run = sub.add_parser("execute", help="Requires new explicit consent; credentials JSON on stdin only")
    run.add_argument("--phase", choices=("service", "evidence", "answer"), required=True)
    for name in ("output", "approval", "approval-sha256"):
        run.add_argument("--"+name, required=True)
    rev = sub.add_parser("freeze", help="Freeze ALL assistant-reviewed verdicts before next phase")
    for name in ("output", "kind", "reviewed", "reviewer"):
        rev.add_argument("--"+name, required=True, choices=("evidence", "answer") if name == "kind" else None)
    out = sub.add_parser("report", help="Local complete paired aggregate only; does not publish")
    out.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.action is None:
        parser.print_help()
        return
    root = Path(args.output).resolve()
    if args.action == "prepare":
        value = prepare(args.metadata, args.snapshot, args.tokenizer, root)
    elif args.action == "freeze":
        value = freeze(root, args.kind, args.reviewed, args.reviewer)
    elif args.action == "report":
        value = report(root)
    else:
        assets.require_container()
        plan, reference = approval(root, args.approval, args.approval_sha256)
        keys = json.load(sys.stdin)
        if set(keys) != {"embedding", "rerank", "classifier", "generation"} or any(not isinstance(k, str) or not k.strip() for k in keys.values()):
            raise ValueError("Only four selected regional keys accepted on stdin")
        # One process holds this lock for the entire phase. Never clear stale locks automatically.
        lock = root / "execute.lock"
        with lock.open("x", encoding="utf-8"):
            pass
        logging.disable(logging.CRITICAL)  # Never emit SDK exceptions, HTTP bodies or source text.
        try:
            if args.phase == "service":
                execute_service(root, plan, reference, keys)
                packets(root, "evidence")
            else:
                execute_judges(root, plan, reference, keys["generation"], args.phase)
            value = {"status": "PHASE_COMPLETE", "phase": args.phase}
        finally:
            lock.unlink()
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # No str(error), payloads, credentials or local variables in public terminal output.
        print(json.dumps({"status": "STOPPED_NO_AUTOMATIC_RETRY", "safe_error_chain": safe_error_chain(error)}), flush=True)
        raise SystemExit(1) from None
