"""Live container-loopback checks; disposable offline volumes ONLY, no model calls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import httpx

STATE = Path("/app/data/runtime_acceptance_state.json")
TEXT = "Fictional CedarDesk local persistence canary. No model or ticket action."
COLLECTION = "runtime_acceptance_canary"


def run(seed=False):
    if not __debug__:
        raise RuntimeError("Acceptance checks require assertions enabled; do not use -O")
    # ponytail: use Docker's network-none isolation, not a new mock-provider stack.
    assert Path("/.dockerenv").exists() and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}
    expected = {"DATABASE_URL": "sqlite:////app/data/acceptance.db", "SQL_ECHO": "false",
        "DOCUMENT_STORAGE_ROOT": "/app/storage/documents", "PYTHON_DOTENV_DISABLED": "1",
        "DASHSCOPE_API_KEY": "offline-acceptance-not-a-key", "DASHSCOPE_BASE_URL": "http://127.0.0.1:9/v1",
        "DASHSCOPE_PHASE_B_API_KEY": "offline-acceptance-not-a-key",
        "DASHSCOPE_PHASE_B_BASE_URL": "http://127.0.0.1:9/v1", "ANONYMIZED_TELEMETRY": "False"}
    assert all(os.getenv(key) == value for key, value in expected.items()), "Disposable environment required"
    with httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=5) as api:
        for _ in range(40):
            try:
                if api.get("/health").status_code == 200:
                    break
            except httpx.RequestError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("Live server did not become healthy")

        def request(method, path, status=200, **kwargs):
            response = api.request(method, path, **kwargs)
            assert response.status_code == status, f"{method} {path}: HTTP {response.status_code}"
            return response.json()

        headers = {}
        for user, password in (("support", "support"), ("support_other", "support"), ("employee", "employee")):
            token = request("POST", "/auth/token", json={"username": user, "password": password})["access_token"]
            headers[user] = {"Authorization": "Bearer " + token}
        request("GET", "/documents", 401)
        request("GET", "/documents", 403, headers=headers["employee"])

        from services.document_service import get_document_chroma_collection
        collection = get_document_chroma_collection(collection_name=COLLECTION)
        if seed:
            assert not STATE.exists() and collection.count() == 0, "Refusing to reseed existing state"
            state = {"documents": []}
            for user in ("support", "support_other"):
                assert request("GET", "/documents", headers=headers[user])["total"] == 0
                doc = request("POST", "/documents/upload", 201, headers=headers[user], data={"category": "it"},
                              files={"file": ("runtime_canary.md", TEXT.encode(), "text/markdown")})
                state["documents"].append({"user": user, "document": doc})
            collection.add(ids=["synthetic-persistence-canary"], embeddings=[[1.0, 0.0]], documents=[TEXT])
        else:
            state = json.loads(STATE.read_text(encoding="utf-8"))

        for item in state["documents"]:
            user, original = item["user"], item["document"]
            doc = request("GET", "/documents/" + original["id"], headers=headers[user])
            assert doc == original and doc["status"] == "uploaded"  # Deliberately no embedding/index call.
            path = Path(doc["source_path"]).resolve()
            assert path.is_relative_to(Path("/app/storage/documents"))
            assert path.read_text(encoding="utf-8") == TEXT
            assert hashlib.sha256(path.read_bytes()).hexdigest() == doc["checksum"]
            other = "support_other" if user == "support" else "support"
            request("GET", "/documents/" + doc["id"], 404, headers=headers[other])
        vector = collection.get(ids=["synthetic-persistence-canary"], include=["documents", "embeddings"])
        assert vector["ids"] == ["synthetic-persistence-canary"] and vector["documents"] == [TEXT]
        assert vector["embeddings"].tolist() == [[1.0, 0.0]]

        enabled = os.getenv("RAG_HYBRID_SEARCH_ENABLED") == "true"
        search = request("POST", "/rag/search", 200 if enabled else 503, headers=headers["support"],
                         json={"query": "Fictional empty scope", "category": "other", "retrieval_mode": "hybrid"})
        if enabled:
            assert search["results"] == [] and search["retrieval_mode"] == "hybrid"
        ask_enabled = enabled and os.getenv("RAG_HYBRID_ASK_ENABLED") == "true"
        answer = request("POST", "/rag/ask", 200 if ask_enabled else 503, headers=headers["support"],
            json={"question": "Fictional empty scope", "category": "other", "retrieval_mode": "hybrid",
                  "refusal_policy": "v5_3_experimental"})
        if ask_enabled:
            assert answer["retrieval_status"] == "no_context" and answer["sources"] == []
        logs = request("GET", "/agent-ops/retrieval-logs", headers=headers["support"])
        ids = {row["id"] for row in logs}
        assert len(ids) >= 2 and all(row["tenant_id"] == "tenant_demo" for row in logs)
        if seed:
            state["initial_audit_ids"] = sorted(ids)
            STATE.write_text(json.dumps(state), encoding="utf-8")
        else:
            assert set(state["initial_audit_ids"]) <= ids

    with sqlite3.connect("/app/data/acceptance.db") as db:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        assert {row[0] for row in db.execute("select version_num from alembic_version")} == set(
            ScriptDirectory.from_config(Config("alembic.ini")).get_heads())
        assert db.execute("pragma integrity_check").fetchone() == ("ok",)
        assert db.execute("select count(*) from documents").fetchone() == (2,)
        for table in ("ticket", "agent_runs", "approval_requests", "tool_calls"):
            assert db.execute(f"select count(*) from {table}").fetchone() == (0,)
    return {"status": "PASS", "phase": "seed" if seed else "verify", "network": "none",
        "hybrid_search_enabled": enabled, "hybrid_ask_enabled": ask_enabled,
        "documents": 2, "vector_canaries": 1, "audit_records": len(ids), "provider_calls": 0,
        "checks": ["live loopback HTTP", "migration head/integrity", "JWT/role/tenant scope",
                   "SQL/document file checksums", "Chroma canary (not retrieval quality)",
                   "Hybrid empty-scope/disabled behavior", "audit persistence", "zero tickets/agent actions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true", help="Seed once into brand-new disposable volumes")
    print(json.dumps(run(parser.parse_args().seed)))
