"""Phase-1 observation/accounting API. No CLI, dataset/credential loader or consent.

Use only in a dedicated serial evaluation process, never in a serving worker.
Caller must independently verify the frozen snapshot, DEV identity and permission.
"""

from contextlib import ExitStack
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import OpenAI

from experiments.evals import runtime_transport as wire
from experiments.evals import runtime_retrieval_inputs as inputs
from experiments.evals.ir.ranx_adapter import collapse_chunk_results_to_document_ranking
from experiments.evals.rerankers.qwen3_reranker import DEFAULT_RERANK_INSTRUCTION, RerankCandidate
from rag_runtime import hybrid_search as hybrid

TENANT, CATEGORY = "tenant_eval_techqa_archival", "other"
ENDPOINTS = {"embedding": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "rerank": "https://dashscope-intl.aliyuncs.com/compatible-api/v1"}
TOKEN_RESERVATIONS = {"embedding": 8192, "rerank": 120000}
RATES = {"embedding": Decimal(".5"), "rerank": Decimal(".74942")}
BUDGET = Decimal("35")
_SLOT = Lock()
_SOURCE_FILES = tuple(Path(sys.modules[name].__file__) for name in (
    __name__, wire.__name__, inputs.__name__, hybrid.__name__, "experiments.evals.ir.rrf",
    "experiments.evals.ir.ranx_adapter", "experiments.evals.rerankers.qwen3_reranker",
    "experiments.evals.retrievers.bm25_techqa_chunks", "experiments.evals.retrievers.bm25_techqa",
    "rag_runtime.query_chroma", "rag_runtime.build_rag_index", "database", "models.document")) + (
    Path(__file__).parent / "reports/runtime_eval_phase0/evaluation_protocol_2026-10-23.md",
    Path(__file__).parent / "datasets/techqa/manifest.json",
    Path(__file__).parent / "runtime_retrieval_report.py",
    Path(__file__).parent / "runtime_retrieval_runner.py")


def _schedule(pairs):
    return {f"{cid}:{arm}": "embedding" if arm == "E" else "rerank"
            for index, (cid, _) in enumerate(pairs)
            for arm in (("E", "DR", "H") if index % 2 == 0 else ("E", "H", "DR"))}


def build_plan(questions, asset_identity, authorization_reference, *, tokenizer_path=None, planned_mode="synthetic"):
    """Accept only 160 explicit (ID, question) pairs, never qrels/answers/labels.

    asset_identity binds an independently verified snapshot/preflight manifest;
    its presence and authorization_reference are NOT proof of those checks/consent.
    """
    if planned_mode not in {"synthetic", "live"} or (planned_mode == "live" and tokenizer_path is None):
        raise ValueError("Explicit mode and reference tokenizer required for live planning")
    pairs = [list(pair) for pair in questions]
    if (len(pairs) != 160 or any(len(pair) != 2 or not all(isinstance(s, str) for s in pair)
            or re.fullmatch(r"DEV_[A-Za-z0-9_]{1,90}", pair[0]) is None or not pair[1].strip() for pair in pairs)
            or len({pair[0] for pair in pairs}) != 160
            or not isinstance(asset_identity, str) or re.fullmatch(r"[a-f0-9]{64}", asset_identity) is None
            or not isinstance(authorization_reference, str) or not authorization_reference.strip()):
        raise ValueError("Require 160 unique DEV pairs, asset identity and explicit authorization reference")
    pairs = sorted([[cid, text.rstrip()] for cid, text in pairs])
    identity = {"questions": pairs, "asset_identity": asset_identity,
                "python": sys.version, "planned_mode": planned_mode,
                "reference_tokenizer": None if tokenizer_path is None else inputs.reference_profile(tokenizer_path),
                "sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _SOURCE_FILES},
                "dependencies": {name: importlib.metadata.version(name) for name in (
                    "openai", "httpx", "chromadb", "bm25s", "numpy", "sqlmodel", "ranx", "tokenizers")},
                "rates": {key: str(value) for key, value in RATES.items()},
                "token_reservations": TOKEN_RESERVATIONS, "budget": str(BUDGET)}
    contract = {"authorization_reference": authorization_reference + ":" + wire.digest(identity),
                "stages": {key: {"base_url": url, "max_request_bytes": 32768 if key == "embedding" else 524288}
                           for key, url in ENDPOINTS.items()},
                "calls": _schedule(pairs), "max_total_request_bytes": 173015040}
    return {"identity": identity, "contract": contract}


def accounting(events, next_stage=None):
    spent = Decimal(0)
    for event in events:
        stage = event["stage"]
        if event["prompt_tokens"] > TOKEN_RESERVATIONS[stage]:
            raise RuntimeError("Usage exceeds reservation; audit before continuing")
        spent += Decimal(event["prompt_tokens"]) * RATES[stage] / 1_000_000
    reserve = Decimal(0) if next_stage is None else Decimal(TOKEN_RESERVATIONS[next_stage]) * RATES[next_stage] / 1_000_000
    if spent + reserve > BUDGET:
        raise RuntimeError("Phase-1 accounting budget exhausted")
    return spent


def validate_response(stage, body, payload):
    if stage == "embedding":
        rows = payload["data"]
        if (len(rows) != 1 or rows[0]["index"] != 0 or len(rows[0]["embedding"]) != 1024
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in rows[0]["embedding"])):
            raise RuntimeError("Invalid query embedding")
    else:
        rows = payload["results"]
        if (not isinstance(rows, list) or any(type(row.get("index")) is not int
                or type(row.get("relevance_score")) not in (int, float)
                or not math.isfinite(row["relevance_score"]) for row in rows)
                or sorted(row["index"] for row in rows) != list(range(len(body["documents"])))):
            raise RuntimeError("Invalid complete rerank permutation")


class Phase1Transport(wire.RuntimeEvalTransport):
    """Reuse the durable wire ledger; restore HTTP200 bodies, never redo paid calls."""

    def _history(self):
        events = super()._history()
        expected = list(self.contract["calls"])
        if [event["call_id"] for event in events] != expected[:len(events)]:
            raise RuntimeError("Phase-1 call order changed")
        for index, event in enumerate(events):
            body = json.loads(json.loads((self.root / f"requests/{index}.json").read_text(encoding="utf-8"))["wire_utf8"])
            payload = json.loads(json.loads((self.root / event["response_file"]).read_text(encoding="utf-8"))["wire_utf8"])
            validate_response(event["stage"], body, payload)
        accounting(events)
        if self.call_id not in {event["call_id"] for event in events}:
            if self.call_id != expected[len(events)]:
                raise RuntimeError("Cannot skip a question or stage")
            accounting(events, self.stage)
        return events

    def handle_request(self, request):
        self._check_files()
        if (self.root / "pending.json").exists():
            raise RuntimeError("Pending attempt; audit before restoration or sending")
        body = json.loads(request.content)
        if wire.digest(body) != wire.digest(self.approved_body):
            raise ValueError("Payload differs from the authorized question/candidate projection")
        events = self._history()
        existing = next((row for row in events if row["call_id"] == self.call_id), None)
        if existing is not None:
            # ponytail: exclusive ledger lock, not another replay database; <=480 records.
            lock = self.root / "transport.lock"
            with lock.open("x", encoding="utf-8"):
                pass
            try:
                self._check_files()
                events = self._history()
                if ((self.root / "pending.json").exists() or wire.digest(body) != existing["request_sha256"]
                        or request.method != "POST" or str(request.url) != ENDPOINTS[self.stage] + wire.STAGES[self.stage][1]):
                    raise RuntimeError("Changed restoration request")
                saved = json.loads((self.root / existing["response_file"]).read_text(encoding="utf-8"))["wire_utf8"]
                return httpx.Response(200, content=saved.encode(), headers={"content-type": "application/json"})
            finally:
                lock.unlink()
        response = super().handle_request(request)
        try:
            self._history()  # Validate semantic envelope and reservation before any next call.
        except Exception:
            wire.write_json(self.root / "pending.json", {"call_id": self.call_id, "status": "audit_required"})
            raise
        return response


def make_client(plan, root, cid, key, approved_body, *, delegate, allow_network=False):
    """Only the reviewed runner enables networking after explicit scoped approval.

    allow_network is an execution guard, never evidence of actual human consent.
    """
    live = (delegate is None and allow_network is True and plan["identity"]["planned_mode"] == "live"
            and plan["identity"]["reference_tokenizer"] is not None)
    if (not live and not isinstance(delegate, httpx.MockTransport)) or not isinstance(key, str) or not key.strip():
        raise ValueError("Explicit key and offline MockTransport required unless scoped live mode is explicitly enabled")
    identity = plan["identity"]
    if (not plan["contract"]["authorization_reference"].endswith(":" + wire.digest(identity))
            or list(plan["contract"]["calls"].items()) != list(_schedule(identity["questions"]).items())
            or plan["contract"]["max_total_request_bytes"] != 173015040
            or plan["contract"]["stages"] != {stage: {"base_url": url,
                "max_request_bytes": 32768 if stage == "embedding" else 524288} for stage, url in ENDPOINTS.items()}
            or identity["sources"] != {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in _SOURCE_FILES}
            or identity["python"] != sys.version
            or any(importlib.metadata.version(name) != value for name, value in identity["dependencies"].items())):
        raise RuntimeError("Phase-1 plan/source/dependency identity changed")
    stage = plan["contract"]["calls"][cid]
    wire.validate_body(stage, approved_body)
    if identity["reference_tokenizer"] is not None:
        estimate = inputs.estimate_input(stage, approved_body, identity["reference_tokenizer"])
        if not estimate["within_reference_profile"]:
            raise ValueError("Input exceeds local reference planning profile; not proof of provider rejection")
    transport = Phase1Transport(plan["contract"], root, cid,
                                httpx.HTTPTransport(retries=0, trust_env=False) if live else delegate)
    transport.approved_body = json.loads(json.dumps(approved_body))
    return OpenAI(api_key=key, base_url=ENDPOINTS[stage], organization="", project="", max_retries=0,
                  timeout=wire.STAGES[stage][2], http_client=httpx.Client(transport=transport,
                  trust_env=False, follow_redirects=False, timeout=wire.STAGES[stage][2]))


def observe_case(plan, index, root, keys, delegates, *, allow_network=False):
    """Run existing Hybrid once; observe Dense/BM25/fusion/full rerank and DR arm.

    Returns raw rankings without reading qrels. Replaying reconstructs them from
    saved provider responses and the same independently verified index snapshot.
    """
    if (type(index) is not int or not 0 <= index < 160
            or (hybrid.CANDIDATE_K, hybrid.RRF_K, hybrid.CONTEXT_K) != (100, 60, 14)):
        raise ValueError("Invalid case index or changed runtime retrieval profile")
    if not _SLOT.acquire(blocking=False):
        raise RuntimeError("Dedicated serial evaluation only")
    try:
        cid, question = plan["identity"]["questions"][index]
        observed = {}
        original_load, original_collection = hybrid.load_visible_chunks, hybrid.get_collection
        original_fuse, original_rerank = hybrid.fuse_rrf, hybrid.rerank_candidates

        def visible(*args, **kwargs):
            rows = original_load(*args, **kwargs)
            if kwargs.get("embedding_ids") is None:
                observed["corpus"] = rows
            return rows

        def embedding(texts):
            if texts != [question] or "embedding" in observed:
                raise RuntimeError("Query projection or embedding call count changed")
            body = {"model": "text-embedding-v4", "input": texts, "dimensions": 1024, "encoding_format": "float"}
            with make_client(plan, root, cid + ":E", keys["embedding"], body,
                             delegate=delegates["embedding"], allow_network=allow_network) as client:
                result = client.embeddings.create(**body)
            observed["embedding"] = [result.data[0].embedding]
            return observed["embedding"]

        def query(**kwargs):
            if kwargs["query_embeddings"] != observed["embedding"] or kwargs["n_results"] != 100:
                raise RuntimeError("Shared Dense100 profile changed")
            result = original_collection().query(**kwargs)
            observed["dense"] = list(result["ids"][0])
            if len(observed["dense"]) != 100:
                raise RuntimeError("Missing full Dense100")
            return result

        def fuse(rankings, **kwargs):
            result = original_fuse(rankings, **kwargs)
            if rankings[0] != observed["dense"] or len(result) != 100:
                raise RuntimeError("Missing or changed full fused ranking")
            observed.update(bm25=list(rankings[1]), fused=list(result))
            return result

        def rerank(query, candidates, **kwargs):
            if query != question or [row.chunk_id for row in candidates] != observed["fused"]:
                raise RuntimeError("Hybrid candidate identity changed")
            corpus = observed["corpus"]
            dense = [RerankCandidate(key, corpus[key].document_id, corpus[key].content) for key in observed["dense"]]
            pools = {"DR": dense, "H": candidates}
            for arm in (("DR", "H") if index % 2 == 0 else ("H", "DR")):
                pool = pools[arm]
                expected = {row.chunk_id: corpus[row.chunk_id] for row in pool}
                if original_load(TENANT, CATEGORY, embedding_ids=list(expected)) != expected:
                    raise RuntimeError("Candidates no longer visible before rerank")
                body = {"model": "qwen3-rerank", "query": question, "documents": [row.content for row in pool],
                        "top_n": len(pool), "instruct": DEFAULT_RERANK_INSTRUCTION}
                with make_client(plan, root, cid + ":" + arm, keys["rerank"], body,
                                 delegate=delegates["rerank"], allow_network=allow_network) as client:
                    observed[arm] = original_rerank(question, pool, client=client)
                if original_load(TENANT, CATEGORY, embedding_ids=list(expected)) != expected:
                    raise RuntimeError("Candidates changed after rerank")
            return observed["H"]

        with ExitStack() as stack:
            for name, value in {"load_visible_chunks": visible, "embed_texts": embedding,
                                "get_collection": lambda: SimpleNamespace(query=query), "fuse_rrf": fuse,
                                "rerank_candidates": rerank}.items():
                stack.enter_context(patch.object(hybrid, name, value))
            # Existing search only needs with_options/context-manager on the unused H client;
            # the observer supplies each independently bounded arm client above.
            stack.enter_context(patch.object(hybrid, "get_rerank_client", lambda: SimpleNamespace(with_options=lambda **kw: ExitStack())))
            top14 = hybrid.search_hybrid(question, 14, TENANT, CATEGORY)
        if [row.chunk_id for row in top14] != [row.chunk_id for row in observed["H"].results[:14]]:
            wire.write_json(Path(root) / "pending.json", {"call_id": cid, "status": "top14_audit_required"})
            raise RuntimeError("Runtime Top14 differs from observed full ranking")
        rankings = {"D": [observed["corpus"][key] for key in observed["dense"]],
                    "DR": observed["DR"].results, "H": observed["H"].results}
        return {"question_id": cid, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "chunk_rankings": {arm: [row.chunk_id for row in rows] for arm, rows in rankings.items()},
                "document_rankings": {arm: collapse_chunk_results_to_document_ranking(rows) for arm, rows in rankings.items()},
                "bm25_chunk_ids": observed["bm25"], "fused_chunk_ids": observed["fused"],
                "top14_chunk_ids": [row.chunk_id for row in top14]}
    finally:
        _SLOT.release()


def save_case(plan, index, root, keys, delegates, *, allow_network=False):
    """Exclusive private case artifact, bound to plan and three raw wire responses.

    Existing artifacts are verified by zero-send reconstruction, never trusted
    as permission to skip missing wire stages. Caller still verifies the snapshot.
    """
    if type(index) is not int or not 0 <= index < 160:
        raise ValueError("Invalid case index")
    root = Path(root)
    identity = {"plan_sha256": wire.digest(plan)}
    if not root.exists():
        root.mkdir()
        wire.write_json(root / "identity.json", identity)
    if (root.is_symlink() or any(path.is_symlink() for path in root.rglob("*"))
            or json.loads((root / "identity.json").read_text(encoding="utf-8")) != identity
            or any(path.name not in {"identity.json", "cases", "wire", "run.lock", "halt.json"} for path in root.iterdir())):
        raise RuntimeError("Case store identity/files changed")
    if (root / "halt.json").exists():
        raise RuntimeError("Case store halted; audit before any progress")
    lock = root / "run.lock"
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        directory = root / "cases"
        directory.mkdir(exist_ok=True)
        filenames = {path.name for path in directory.iterdir()}
        if filenames != {f"{i:03d}.json" for i in range(len(filenames))}:
            raise RuntimeError("Case files are not a contiguous prefix")
        path = directory / f"{index:03d}.json"
        if not path.exists() and index != len(filenames):
            raise RuntimeError("Cannot skip saving a case")
        history = root / "wire/attempts.jsonl"
        events = wire._read_jsonl(history) if history.exists() else []
        receipts = {event["call_id"]: {key: event[key] for key in ("call_id", "request_sha256", "response_sha256")}
                    for event in events if event.get("status") == "complete"}
        for i in range(len(filenames)):
            artifact = json.loads((directory / f"{i:03d}.json").read_text(encoding="utf-8"))
            question_id = plan["identity"]["questions"][i][0]
            call_ids = [key for key in plan["contract"]["calls"] if key.startswith(question_id + ":")]
            if (set(artifact) != {"plan_sha256", "index", "result", "result_sha256", "calls"}
                    or artifact["plan_sha256"] != identity["plan_sha256"] or artifact["index"] != i
                    or artifact["result_sha256"] != wire.digest(artifact["result"])
                    or not set(call_ids) <= receipts.keys()
                    or artifact["calls"] != [receipts[key] for key in call_ids]):
                raise RuntimeError("Case artifact integrity/wire receipts changed")
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        cid = plan["identity"]["questions"][index][0]
        ids = [key for key in plan["contract"]["calls"] if key.startswith(cid + ":")]
        if saved is not None:
            if (saved.get("plan_sha256") != identity["plan_sha256"] or saved.get("index") != index
                    or not set(ids) <= {event["call_id"] for event in events if event.get("status") == "complete"}):
                raise RuntimeError("Saved case lacks its complete wire history")
        result = observe_case(plan, index, root / "wire", keys, delegates, allow_network=allow_network)
        events = wire._read_jsonl(root / "wire/attempts.jsonl")
        calls = [{"call_id": event["call_id"], "request_sha256": event["request_sha256"],
                  "response_sha256": event["response_sha256"]} for event in events if event["call_id"] in ids]
        if [event["call_id"] for event in calls] != ids:
            raise RuntimeError("Incomplete case wire receipts")
        record = identity | {"index": index, "result": result, "result_sha256": wire.digest(result), "calls": calls}
        if saved is not None and saved != record:
            wire.write_json(root / "halt.json", {"case_index": index, "reason": "artifact_reconstruction_mismatch"})
            raise RuntimeError("Case artifact differs from snapshot/wire reconstruction")
        if saved is None:
            wire.write_json(path, record)
        return record
    finally:
        lock.unlink()
