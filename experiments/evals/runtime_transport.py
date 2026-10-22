"""Evaluation-only wire limits/checkpoints. No CLI, credentials or dataset loader.

This is not authorization, a paid runner, or a CNY invoice cap. A future frozen
protocol must authorize payloads, prices and budget before attaching these clients
to an isolated serial evaluation process. Never patch the daily service.
"""

from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from experiments.evals.refusal_phase_b_runner import _append_jsonl, _read_jsonl
from experiments.evals.refusal_v5_paired_runner import write_json

STAGES = {
    "embedding": ("text-embedding-v4", "/embeddings", 30, 0),
    "rerank": ("qwen3-rerank", "/reranks", 15, 0),
    "classifier": ("qwen3.5-plus-2026-04-20", "/chat/completions", 30, 2048),
    "generation": ("qwen3.5-plus", "/chat/completions", 30, 1024),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_contract(contract):
    if (set(contract) != {"authorization_reference", "stages", "calls", "max_total_request_bytes"}
            or not isinstance(contract["authorization_reference"], str) or not contract["authorization_reference"].strip()
            or not isinstance(contract["stages"], dict) or not contract["stages"]
            or not set(contract["stages"]) <= set(STAGES)
            or not isinstance(contract["calls"], dict) or not contract["calls"]
            or any(not isinstance(cid, str) or not cid.strip() or len(cid) > 128
                   or stage not in contract["stages"] for cid, stage in contract["calls"].items())
            or type(contract["max_total_request_bytes"]) is not int or contract["max_total_request_bytes"] <= 0):
        raise ValueError("Invalid evaluation transport contract")
    for stage, spec in contract["stages"].items():
        if set(spec) != {"base_url", "max_request_bytes"}:
            raise ValueError("Invalid stage contract")
        url = urlsplit(spec["base_url"])
        if (url.scheme != "https" or not url.hostname or url.port is not None
                or any((url.username, url.password, url.query, url.fragment))
                or not url.path.endswith("/v1") or spec["base_url"].endswith("/")
                or type(spec["max_request_bytes"]) is not int or not 1 <= spec["max_request_bytes"] <= 2 * 1024 * 1024):
            raise ValueError("Invalid pinned stage URL/byte limit")


def validate_body(stage, body):
    model, _, _, output = STAGES[stage]
    if not isinstance(body, dict) or body.get("model") != model:
        raise ValueError("Unapproved model")
    if stage == "embedding":
        if (set(body) != {"model", "input", "encoding_format", "dimensions"}
                or not isinstance(body["input"], list) or len(body["input"]) != 1
                or not isinstance(body["input"][0], str) or not body["input"][0].strip()
                or body["dimensions"] != 1024 or body["encoding_format"] != "float"):
            raise ValueError("Only one explicit 1024-dimensional query embedding is allowed")
    elif stage == "rerank":
        if (set(body) != {"model", "query", "documents", "top_n", "instruct"}
                or not isinstance(body["query"], str) or not body["query"].strip()
                or not isinstance(body["documents"], list) or not 1 <= len(body["documents"]) <= 100
                or any(not isinstance(text, str) or not text.strip() for text in body["documents"])
                or type(body["top_n"]) is not int or body["top_n"] != len(body["documents"])
                or not isinstance(body["instruct"], str)):
            raise ValueError("Invalid bounded rerank request")
    else:
        allowed = {"model", "messages", "temperature", "max_tokens", "enable_thinking"}
        if stage == "classifier":
            allowed.add("response_format")
        if (set(body) != allowed or type(body["max_tokens"]) is not int or body["max_tokens"] != output
                or type(body["temperature"]) not in (int, float)
                or body["enable_thinking"] is not False or body["temperature"] != (0 if stage == "classifier" else 0.2)
                or not isinstance(body["messages"], list) or len(body["messages"]) != 2
                or any(set(message) != {"role", "content"} or message["role"] != role
                       or not isinstance(message["content"], str) or not message["content"].strip()
                       for message, role in zip(body["messages"], ("system", "user"), strict=True))
                or (stage == "classifier" and body["response_format"] != {"type": "json_object"})):
            raise ValueError("Invalid bounded chat request")


def usage_record(stage, payload):
    usage = payload["usage"]
    total = usage["total_tokens"]
    prompt = total if stage == "rerank" else usage["prompt_tokens"]
    completion = usage.get("completion_tokens", 0) if stage in {"embedding", "rerank"} else usage["completion_tokens"]
    if (not all(type(value) is int for value in (prompt, completion, total))
            or prompt <= 0 or completion < 0 or total != prompt + completion or completion > STAGES[stage][3]):
        raise ValueError("Missing, inconsistent or out-of-bound provider usage")
    finish = payload["choices"][0]["finish_reason"] if stage in {"classifier", "generation"} else None
    if stage in {"classifier", "generation"} and finish != "stop":
        raise ValueError("Incomplete chat response")
    return dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total, finish_reason=finish)


class RuntimeEvalTransport(httpx.BaseTransport):
    def __init__(self, contract, root, call_id, delegate):
        validate_contract(contract)
        self.contract = json.loads(json.dumps(contract))
        self.root, self.call_id, self.delegate = Path(root), call_id, delegate
        self.stage = self.contract["calls"][call_id]
        self.identity = {
            "contract_sha256": digest(self.contract),
            "stage_profile": {key: list(value) for key, value in STAGES.items()},
            "source_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
                __file__, write_json.__code__.co_filename, _append_jsonl.__code__.co_filename)},
            "dependencies": {name: importlib.metadata.version(name) for name in ("openai", "httpx")},
        }
        if not self.root.exists():
            self.root.mkdir()  # Parent must exist; never reuse an unidentified directory.
            write_json(self.root / "identity.json", self.identity)
        self._check_files()

    def _check_files(self):
        if (self.root.is_symlink() or any(p.is_symlink() for p in self.root.rglob("*"))
                or json.loads((self.root / "identity.json").read_text(encoding="utf-8")) != self.identity):
            raise RuntimeError("Evaluation checkpoint identity/links changed")
        if any(p.name not in {"identity.json", "attempts.jsonl", "pending.json", "transport.lock", "requests", "responses", "failure.json"}
               for p in self.root.iterdir()):
            raise RuntimeError("Unexpected evaluation checkpoint file")

    def _history(self):
        # ponytail: rescan bounded local history per send, not another ledger database.
        path = self.root / "attempts.jsonl"
        events = _read_jsonl(path) if path.exists() else []
        seen = set()
        for index, event in enumerate(events):
            if set(event) != {"call_id", "stage", "request_sha256", "request_bytes", "prompt_tokens", "completion_tokens",
                              "total_tokens", "finish_reason", "status", "response_sha256", "response_file", "latency_ms"}:
                raise RuntimeError("Invalid transport history fields")
            cid, stage = event["call_id"], event["stage"]
            if (cid in seen or self.contract["calls"].get(cid) != stage or event["status"] != "complete"
                    or event["response_file"] != f"responses/{index}.json"
                    or type(event["request_bytes"]) is not int
                    or not 0 < event["request_bytes"] <= self.contract["stages"][stage]["max_request_bytes"]
                    or type(event["latency_ms"]) is not int or event["latency_ms"] < 0):
                raise RuntimeError("Unexpected transport history")
            seen.add(cid)
            wire = json.loads((self.root / "requests" / f"{index}.json").read_text(encoding="utf-8"))["wire_utf8"]
            body = json.loads(wire)
            validate_body(stage, body)
            saved = json.loads((self.root / event["response_file"]).read_text(encoding="utf-8"))
            payload = json.loads(saved["wire_utf8"])
            accounting = usage_record(stage, payload)
            if (digest(body) != event["request_sha256"] or len(wire.encode()) != event["request_bytes"]
                    or digest(payload) != event["response_sha256"]
                    or any(event[key] != value for key, value in accounting.items())):
                raise RuntimeError("Request/response/accounting checkpoint changed")
        expected = {f"{index}.json" for index in range(len(events))}
        if any({p.name for p in (self.root / name).iterdir()} != expected
               for name in ("requests", "responses") if (self.root / name).exists()):
            raise RuntimeError("Orphan request/response file: audit before continuing")
        if sum(event["request_bytes"] for event in events) > self.contract["max_total_request_bytes"]:
            raise RuntimeError("Transport history exceeds approved byte budget")
        return events

    def handle_request(self, request):
        self._check_files()
        if (self.root / "pending.json").exists():
            raise RuntimeError("Pending/unknown attempt: audit before any further send")
        lock = self.root / "transport.lock"
        with lock.open("x", encoding="utf-8"):
            pass
        response = None
        try:
            if (self.root / "pending.json").exists():
                raise RuntimeError("Pending/unknown attempt: audit before any further send")
            events_path = self.root / "attempts.jsonl"
            events = self._history()
            if any(event["call_id"] == self.call_id for event in events):
                raise RuntimeError("Completed call must be restored, never sent twice")
            if any(event["call_id"] not in self.contract["calls"] or event["status"] != "complete" for event in events):
                raise RuntimeError("Unexpected or unfinished transport history")
            _, path, timeout, _ = STAGES[self.stage]
            spec = self.contract["stages"][self.stage]
            if request.method != "POST" or str(request.url) != spec["base_url"] + path:
                raise ValueError("Unapproved endpoint/method")
            timeouts = request.extensions.get("timeout", {})
            if set(timeouts) != {"connect", "read", "write", "pool"} or any(
                type(value) not in (int, float) or not 0 < value <= timeout for value in timeouts.values()
            ):
                raise ValueError("Unbounded HTTP phase timeout")
            body = json.loads(request.content)
            validate_body(self.stage, body)
            credential = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not credential or credential in json.dumps(body, ensure_ascii=False):
                raise ValueError("Missing credentials or credential text in model payload")
            size = len(request.content)
            if (size > spec["max_request_bytes"] or sum(event["request_bytes"] for event in events) + size
                    > self.contract["max_total_request_bytes"]):
                raise RuntimeError("Evaluation request byte budget exhausted")
            row = dict(call_id=self.call_id, stage=self.stage, request_sha256=digest(body), request_bytes=size)
            pending = self.root / "pending.json"
            write_json(pending, row)  # Durable reservation before a possible paid send.
            request_dir = self.root / "requests"
            request_dir.mkdir(exist_ok=True)
            write_json(request_dir / f"{len(events)}.json", {"wire_utf8": request.content.decode("utf-8")})
            started = time.monotonic()
            response = self.delegate.handle_request(request)
            response.read()
            if response.status_code != 200 or credential.encode() in response.content:
                write_json(self.root / "failure.json", dict(http_status=response.status_code,
                    response_sha256=hashlib.sha256(response.content).hexdigest(), response_bytes=len(response.content)))
                response.close()
                raise RuntimeError("Provider HTTP failure/credential echo: pending retained, no body/headers logged")
            directory = self.root / "responses"
            directory.mkdir(exist_ok=True)
            response_path = directory / f"{len(events)}.json"
            write_json(response_path, {"wire_utf8": response.content.decode("utf-8")})
            payload = response.json()
            accounting = usage_record(self.stage, payload)
            _append_jsonl(events_path, row | accounting | dict(status="complete", response_sha256=digest(payload),
                          response_file=f"responses/{len(events)}.json", latency_ms=round((time.monotonic()-started)*1000)))
            pending.unlink()  # Any prior crash/error leaves pending and prevents automatic resubmission.
            return response
        finally:
            if response is not None:
                response.close()
            lock.unlink()

    def close(self):
        self.delegate.close()


def make_client(contract, root, call_id, api_key, *, delegate=None):
    """Caller supplies credentials/contract after authorization; import alone sends nothing."""
    validate_contract(contract)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Explicit credentials required; no SDK environment fallback")
    if delegate is not None and not isinstance(delegate, httpx.MockTransport):
        raise ValueError("Only offline MockTransport injection is accepted")
    stage = contract["calls"][call_id]
    transport = RuntimeEvalTransport(contract, root, call_id,
        delegate if delegate is not None else httpx.HTTPTransport(retries=0, trust_env=False))
    client = OpenAI(api_key=api_key, base_url=contract["stages"][stage]["base_url"], organization="", project="",
        max_retries=0, timeout=STAGES[stage][2], http_client=httpx.Client(
            transport=transport, trust_env=False, follow_redirects=False, timeout=STAGES[stage][2]))
    if stage == "embedding":
        client.embeddings.create = partial(client.embeddings.create, dimensions=1024)
    elif stage == "generation":
        # Dense has no output/thinking ceiling; apply only to this evaluation client.
        # Hybrid's with_options copy already supplies these same explicit request fields.
        client.chat.completions.create = partial(client.chat.completions.create,
            max_tokens=1024, extra_body={"enable_thinking": False})
    return client
