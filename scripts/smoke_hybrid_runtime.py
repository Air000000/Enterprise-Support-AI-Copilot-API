"""Isolated ASGI/SQL/Chroma smoke; default is a zero-call plan, not a live server."""

import argparse
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from unittest.mock import patch
from urllib.parse import urlsplit
from uuid import uuid4

import httpx


DOCS = (
    ("vpn.md", "it", "support", "VPN error 809: reconnect the VPN client and check the network firewall settings."),
    ("printer.md", "it", "support", "Printer paper jam: switch off the printer and remove the jammed paper from the tray."),
    ("leave.md", "hr", "support", "Annual leave: submit a leave request in the HR portal and obtain manager approval."),
    ("foreign.md", "it", "support_other", "FOREIGN_ONLY VPN error 809: this synthetic guide belongs to another tenant."),
)
VPN_QUERY = "How do I resolve VPN error 809?"
HR_QUERY = "How do I request annual leave?"
QUERIES = {VPN_QUERY, HR_QUERY}
LIMITS = {"embedding": 8, "rerank": 3}
MAX_BODY_BYTES = 4096
MAX_TOTAL_BYTES = 16384


class BoundedTransport(httpx.BaseTransport):
    """Bound actual wire attempts, including failures; never log credentials or bodies."""

    def __init__(self, kind, base_url, records, delegate, *, ask=False):
        self.kind, self.base_url, self.records, self.delegate = kind, base_url, records, delegate
        self.ask = ask

    def handle_request(self, request):
        from scripts import hybrid_ask_cases as cases
        docs, queries, limits = (cases.DOCS, cases.QUERIES, cases.LIMITS) if self.ask else (DOCS, QUERIES, LIMITS)
        max_body, max_total = (cases.MAX_BODY_BYTES, cases.MAX_TOTAL_BYTES) if self.ask else (MAX_BODY_BYTES, MAX_TOTAL_BYTES)
        path = {"embedding": "/embeddings", "rerank": "/reranks", "classifier": "/chat/completions", "generation": "/chat/completions"}[self.kind]
        expected = self.base_url.rstrip("/") + path
        body = json.loads(request.content)
        if request.method != "POST" or str(request.url) != expected:
            raise RuntimeError("Unapproved model endpoint")
        model = {"embedding": "text-embedding-v4", "rerank": "qwen3-rerank", "classifier": "qwen3.5-plus-2026-04-20", "generation": "qwen3.5-plus"}[self.kind]
        if self.kind not in limits or body.get("model") != model:
            raise RuntimeError("Unapproved model")
        if self.kind == "embedding":
            if not set(body) <= {"model", "input", "encoding_format"}:
                raise RuntimeError("Unapproved embedding fields")
            allowed = queries | {f"{Path(name).stem}\n\n{text}" for name, _, _, text in docs}
            if not body.get("input") or not set(body["input"]) <= allowed:
                raise RuntimeError("Unapproved embedding text")
        elif self.kind == "rerank":
            if not set(body) <= {"model", "query", "documents", "top_n", "instruct"}:
                raise RuntimeError("Unapproved rerank fields")
            allowed = cases.allowed_contents(body.get("query")) if self.ask and body.get("query") in queries else {text for _, _, user, text in docs if user == "support"}
            if body.get("query") not in queries or not body.get("documents") or not set(body["documents"]) <= allowed:
                raise RuntimeError("Unapproved rerank text")
        else:
            cases.validate_chat_body(self.kind, body)
        size = len(request.content)
        if (sum(r["kind"] == self.kind for r in self.records) >= limits[self.kind]
                or size > max_body or sum(r["bytes"] for r in self.records) + size > max_total):
            raise RuntimeError("Smoke call/size budget exhausted")
        row = {"kind": self.kind, "bytes": size, "status": None, "total_tokens": None}
        self.records.append(row)  # Count an attempted send even if its response is lost.
        started = time.perf_counter()
        try:
            response = self.delegate.handle_request(request)
            response.read()
            row["status"] = response.status_code
            try:
                row["total_tokens"] = response.json().get("usage", {}).get("total_tokens")
                if self.ask and self.kind in {"classifier", "generation"} and response.status_code == 200:
                    row["completion_text"] = response.json()["choices"][0]["message"]["content"]
            except (ValueError, AttributeError, KeyError, IndexError, TypeError):
                pass
            return response
        finally:
            row["latency_ms"] = round((time.perf_counter() - started) * 1000)

    def close(self):
        self.delegate.close()


def run_flow(client):
    def request(method, path, expected=200, **kwargs):
        response = client.request(method, path, **kwargs)
        if response.status_code != expected:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}")
        return response.json()

    headers = {}
    for username in ("support", "support_other"):
        token = request("POST", "/auth/token", json={"username": username, "password": "support"})["access_token"]
        headers[username] = {"Authorization": f"Bearer {token}"}
    ids = {}
    for name, category, user, text in DOCS:
        doc = request("POST", "/documents/upload", 201, headers=headers[user],
                      data={"category": category}, files={"file": (name, text.encode(), "text/markdown")})
        ids[name] = doc["id"]
        indexed = request("POST", f"/documents/{doc['id']}/index", headers=headers[user])
        if indexed["status"] != "indexed" or indexed["chunk_count"] != 1:
            raise RuntimeError("Synthetic document did not index as one chunk")
    own = headers["support"]
    if client.get(f"/documents/{ids['foreign.md']}", headers=own).status_code != 404:
        raise RuntimeError("Cross-tenant document access allowed")

    def search(query, category, mode=None):
        body = {"query": query, "category": category, "tenant_id": "tenant_other"}
        if mode:
            body["retrieval_mode"] = mode
        data = request("POST", "/rag/search", headers=own, json=body)
        if data["retrieval_mode"] != (mode or "dense"):
            raise RuntimeError("Wrong retrieval mode")
        if any(r["tenant_id"] != "tenant_demo" or r["category"] != category for r in data["results"]):
            raise RuntimeError("Search scope violation")
        return data

    dense = search(VPN_QUERY, "it")
    hybrid = search(VPN_QUERY, "it", "hybrid")
    if dense["top_k"] != 3 or {r["document_id"] for r in dense["results"]} != {ids['vpn.md'], ids['printer.md']}:
        raise RuntimeError("Default Dense regression")
    if hybrid["top_k"] != 14 or {r["document_id"] for r in hybrid["results"]} != {ids['vpn.md'], ids['printer.md']}:
        raise RuntimeError("Hybrid candidate scope mismatch")
    if hybrid["results"][0]["document_id"] != ids["vpn.md"]:
        raise RuntimeError("VPN synthetic relevance check failed")
    hr = search(HR_QUERY, "hr", "hybrid")
    if [r["document_id"] for r in hr["results"]] != [ids["leave.md"]]:
        raise RuntimeError("Category filtering failed")
    request("DELETE", f"/documents/{ids['vpn.md']}", headers=own)
    after_delete = search(VPN_QUERY, "it", "hybrid")
    if [r["document_id"] for r in after_delete["results"]] != [ids["printer.md"]]:
        raise RuntimeError("Deleted document still searchable")
    request("DELETE", f"/documents/{ids['printer.md']}", headers=own)
    if search(VPN_QUERY, "it", "hybrid")["results"]:
        raise RuntimeError("Empty tenant/category scope was not empty")
    logs = request("GET", "/agent-ops/retrieval-logs", headers=own)
    hybrid_logs = [r for r in logs if isinstance(json.loads(r["scores_json"]), dict)]
    if len(logs) != 5 or len(hybrid_logs) != 4 or any(r["top_distance"] is not None for r in hybrid_logs):
        raise RuntimeError("Missing or incorrectly typed retrieval audit")
    return {"status": "PASS", "checks": ["upload/index", "JWT tenant scope", "body tenant ignored",
            "default Dense", "Hybrid VPN ranking", "category", "delete", "empty corpus", "typed audit"]}


def execute(env_file, *, ask=False):
    from dotenv import dotenv_values
    from openai import OpenAI
    import chromadb
    from fastapi.testclient import TestClient

    if "database" in sys.modules:
        raise RuntimeError("Execute in a fresh process to isolate the database")
    config = dotenv_values(env_file)
    embed_url = config.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    rerank_url = config.get("DASHSCOPE_RERANK_BASE_URL", "")
    rerank_host = urlsplit(rerank_url).hostname or ""
    if (embed_url.rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1"
            or not rerank_host.startswith("ws-") or not rerank_host.endswith(".ap-southeast-1.maas.aliyuncs.com")
            or urlsplit(rerank_url).scheme != "https" or urlsplit(rerank_url).path.rstrip("/") != "/compatible-api/v1"
            or not config.get("DASHSCOPE_API_KEY") or not config.get("DASHSCOPE_RERANK_API_KEY")):
        raise RuntimeError("Missing credentials or endpoints outside the approved Beijing/Singapore scope")
    phase_url = config.get("DASHSCOPE_PHASE_B_BASE_URL", "")
    parsed = urlsplit(phase_url)
    if ask and (not config.get("DASHSCOPE_PHASE_B_API_KEY") or parsed.scheme != "https"
            or parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment
            or not ((parsed.hostname or "") == "dashscope-intl.aliyuncs.com" or (parsed.hostname or "").endswith(".ap-southeast-1.maas.aliyuncs.com"))
            or parsed.path.rstrip("/") != "/compatible-mode/v1"):
        raise RuntimeError("Dedicated Singapore classifier configuration required")
    records, progress = [], []
    with tempfile.TemporaryDirectory(prefix="hybrid_smoke_") as temp, ExitStack() as stack:
        os.environ.update(DATABASE_URL="sqlite:///" + str(Path(temp) / "smoke.db"), SQL_ECHO="false",
            DOCUMENT_STORAGE_ROOT=str(Path(temp) / "documents"), AUTH_DEMO_USERS_JSON="",
            JWT_SECRET=secrets.token_hex(32), RAG_HYBRID_SEARCH_ENABLED="true", ANONYMIZED_TELEMETRY="False",
            EMBEDDING_MODEL="text-embedding-v4", PYTHON_DOTENV_DISABLED="1",
            DASHSCOPE_API_KEY=config["DASHSCOPE_API_KEY"], DASHSCOPE_BASE_URL=embed_url)
        if ask:
            os.environ.update(RAG_HYBRID_ASK_ENABLED="true", DASHSCOPE_PHASE_B_API_KEY=config["DASHSCOPE_PHASE_B_API_KEY"],
                DASHSCOPE_PHASE_B_BASE_URL=phase_url, CHAT_MODEL="qwen3.5-plus", DASHSCOPE_MODEL="qwen3.5-plus")
        import database
        import main
        import rag_runtime.build_rag_index as embeddings
        import rag_runtime.hybrid_search as hybrid
        import rag_runtime.query_chroma as dense
        import services.document_service as documents
        database.create_db_and_tables()
        stack.callback(database.engine.dispose)
        stack.callback(main.llm_client.close)
        for name in ("httpx", "httpx2", "openai", "bm25s"):
            logging.getLogger(name).setLevel(logging.WARNING)
        chroma = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
        collection = chroma.create_collection("smoke_" + uuid4().hex, embedding_function=None)
        stack.callback(chroma.delete_collection, collection.name)

        def model_client(kind, url, key):
            transport = BoundedTransport(kind, url, records, httpx.HTTPTransport(retries=0), ask=ask)
            return OpenAI(api_key=key, base_url=url, max_retries=0, timeout=30,
                http_client=httpx.Client(transport=transport, trust_env=False, follow_redirects=False))

        embedding_client = stack.enter_context(model_client("embedding", embed_url, config["DASHSCOPE_API_KEY"]))
        stack.enter_context(patch.object(embeddings, "get_client", lambda: embedding_client))
        stack.enter_context(patch.object(hybrid, "get_rerank_client", lambda: model_client(
            "rerank", rerank_url, config["DASHSCOPE_RERANK_API_KEY"])))
        if ask:
            import rag_runtime.hybrid_ask as hybrid_ask
            stack.enter_context(patch.object(hybrid_ask, "get_classifier_client", lambda: model_client(
                "classifier", phase_url, config["DASHSCOPE_PHASE_B_API_KEY"])))
            stack.enter_context(patch.object(hybrid_ask, "get_llm_client", lambda: model_client(
                "generation", embed_url, config["DASHSCOPE_API_KEY"])))
        for module, attr in ((dense, "get_collection"), (hybrid, "get_collection"), (documents, "get_document_chroma_collection")):
            stack.enter_context(patch.object(module, attr, lambda **_: collection))
        try:
            with TestClient(main.app) as client:
                if ask:
                    from scripts.hybrid_ask_cases import run_flow as run_ask_flow
                    result = run_ask_flow(client, records, progress)
                else:
                    result = run_flow(client)
            result["calls"] = records
            return result
        finally:
            print(json.dumps({"provider_attempts": records, "cases": progress}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Real paid calls; requires prior human authorization, not granted by this flag")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--ask", action="store_true", help="Use the separately authorized 7-document, 8-case ask acceptance plan")
    args = parser.parse_args()
    if not args.execute:
        from scripts import hybrid_ask_cases as cases
        print(json.dumps({"mode": "PLAN_ONLY", "limits": cases.LIMITS if args.ask else LIMITS,
                          "max_total_request_bytes": cases.MAX_TOTAL_BYTES if args.ask else MAX_TOTAL_BYTES,
                          "documents": cases.DOCS if args.ask else DOCS,
                          "queries": sorted(cases.QUERIES if args.ask else QUERIES)}, ensure_ascii=False, indent=2))
    elif args.env_file is None:
        parser.error("--execute requires --env-file")
    else:
        try:
            print(json.dumps(execute(args.env_file, ask=args.ask), ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"Smoke failed ({type(exc).__name__}); no automatic rerun. Provider bodies are not printed.", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
