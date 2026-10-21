"""Mocked execution checks, never real calls or model-quality evidence."""

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import httpx
import pytest

from experiments.evals import refusal_v5_4_synthetic_runner as run
from tests.test_refusal_v5_2_1_runner import Client

URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@pytest.fixture(autouse=True)
def fixed_offline_clock(monkeypatch):
    monkeypatch.setattr(run, "datetime", SimpleNamespace(now=lambda _: datetime(2026, 10, 21, tzinfo=timezone.utc)))


def prepared(tmp_path, monkeypatch):
    data = tmp_path / "probe"
    for key, value in {"DATA": data, "INPUTS": data / "inputs.json", "CONTRACT": data / "contract.json"}.items():
        monkeypatch.setattr(run, key, value)
    env = tmp_path / "fake.env"
    env.write_text(f"DASHSCOPE_PHASE_B_API_KEY=offline-not-a-key\nDASHSCOPE_PHASE_B_BASE_URL={URL}\n")
    contract = run.prepare_contract(env, "OFFLINE_TEST_ONLY_NOT_HUMAN_CONSENT", authorization_confirmed=True)
    return contract, run.load_probe()[1], env


def test_consent_and_plan_drift_block_before_credentials_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(run.reuse, "resolve_credentials", lambda _: pytest.fail("credentials read"))
    monkeypatch.setattr(run, "DATA", tmp_path / "absent")
    for confirmation in (False, None, 1, "continue"):
        with pytest.raises(RuntimeError, match="authorization"):
            run.prepare_contract("not-read.env", "not-consent", authorization_confirmed=confirmation)
        with pytest.raises(RuntimeError, match="authorization"):
            run.run_paid(client=Client(), authorization_confirmed=confirmation)
    monkeypatch.setattr(run, "PLAN_SHA256", "changed")
    with pytest.raises(RuntimeError, match="plan/protocol"):
        run.prepare_contract("not-read.env", "offline", authorization_confirmed=True)
    assert not run.DATA.exists()


def test_default_cli_is_zero_call_and_paid_mode_without_consent_stops(monkeypatch, capsys):
    monkeypatch.setattr(run.reuse, "resolve_credentials", lambda _: pytest.fail("credentials read"))
    monkeypatch.setattr("sys.argv", ["runner"])
    run.main()
    assert json.loads(capsys.readouterr().out)["provider_calls"] == 0
    monkeypatch.setattr("sys.argv", ["runner", "run"])
    monkeypatch.setattr(run, "load_probe", lambda: pytest.fail("execution state read before consent"))
    with pytest.raises(SystemExit) as stopped:
        run.main()
    assert stopped.value.code == 1


def test_explicit_credentials_and_retirement_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "DATA", tmp_path / "absent")
    with pytest.raises(RuntimeError, match="credential file"):
        run.prepare_contract(None, "offline", authorization_confirmed=True)
    with pytest.raises(RuntimeError, match="credential file"):
        run.bounded_client({}, None)
    monkeypatch.setattr(run, "datetime", SimpleNamespace(now=lambda _: datetime(2027, 1, 19, tzinfo=timezone.utc)))
    with pytest.raises(RuntimeError, match="retirement"):
        run.run_paid(client=Client(), authorization_confirmed=True)


def test_complete_mock_run_preserves_raw_before_parse_and_keeps_old_engines_isolated(tmp_path, monkeypatch):
    contract, rows, _ = prepared(tmp_path, monkeypatch)
    root, client = tmp_path / "run", Client()
    original = run.engine.finish_record
    def finish(raw, row):
        staged = run.paired.shared._read_json(tmp_path / "run_raw_response.json")
        assert staged == raw and "result" not in staged
        return original(raw, row)
    monkeypatch.setattr(run.engine, "finish_record", finish)
    events = run.run_paid(root=root, client=client, authorization_confirmed=True)
    assert len(events) == client.calls == 24
    assert all(e["result"] is not None for e in events)  # Includes deliberately wrong semantics.
    assert run.read_state(root, contract, rows)[1] == 24
    assert set(run.control.BUILDERS) == {"v5_1", "v5_2"}
    assert run.control.paired.load_state is run.paired.load_state
    assert "expected" not in run.INPUTS.read_text()
    assert not (root / "runner.lock").exists() and not (tmp_path / "run_raw_response.json").exists()


@pytest.mark.parametrize("failure", ("schema", "usage", "network", "nonstop"))
def test_first_failure_is_saved_and_blocks_all_further_calls(tmp_path, monkeypatch, failure):
    prepared(tmp_path, monkeypatch)
    client, root = Client(bad=failure == "schema", unknown=failure == "usage"), tmp_path / "failed"
    create = client.create
    def respond(**kwargs):
        if failure == "network":
            client.calls += 1
            raise httpx.ReadTimeout("PRIVATE_ERROR_DO_NOT_PERSIST")
        response = create(**kwargs)
        if failure == "nonstop":
            response.choices[0].finish_reason = "length"
        return response
    client.create = respond
    for resume in (False, True):
        with pytest.raises(RuntimeError):
            run.run_paid(root=root, client=client, authorization_confirmed=True, resume_after_audit=resume)
    assert client.calls == 1
    rows = run.paired.shared._read_jsonl(root / "v5_3_1/attempts.jsonl")
    assert len(rows) == 1 and "PRIVATE_ERROR" not in json.dumps(rows)
    if failure in {"schema", "nonstop"}:
        assert run.audit(root=root)[1] == 1
    else:
        with pytest.raises(RuntimeError):
            run.audit(root=root)
    assert client.calls == 1


def test_interruption_requires_audit_and_recovers_staged_response_without_resend(tmp_path, monkeypatch):
    contract, rows, _ = prepared(tmp_path, monkeypatch)
    root, client = tmp_path / "recover", Client()
    run.run_paid(root=root, client=client, authorization_confirmed=True, max_new_calls=1)
    with pytest.raises(RuntimeError, match="existing state"):
        run.run_paid(root=root, client=client, authorization_confirmed=True)
    raw = run.engine.request_record(rows[0], "v5_4", 1, contract, client)
    run.paired.write_json(root / "pending_request.json", {k: raw[k] for k in ("call_index", "arm", "question_id")})
    run.paired.write_json(tmp_path / "recover_raw_response.json", raw)
    assert run.audit(root=root)[1] == 2 and client.calls == 2
    events = run.run_paid(root=root, client=client, authorization_confirmed=True, resume_after_audit=True)
    assert len(events) == client.calls == 24
    run.run_paid(root=root, client=client, authorization_confirmed=True, resume_after_audit=True)
    assert client.calls == 24


def test_pending_lock_budget_caps_and_input_tampering_block_without_send(tmp_path, monkeypatch):
    contract, rows, _ = prepared(tmp_path, monkeypatch)
    client, root = Client(), tmp_path / "pending"
    run.audit(root=root)
    run.paired.write_json(root / "pending_request.json", {"call_index": 0, "arm": "v5_3_1", "question_id": rows[0]["question_id"]})
    with pytest.raises(RuntimeError, match="unknown charge"):
        run.run_paid(root=root, client=client, authorization_confirmed=True, resume_after_audit=True)
    (root / "runner.lock").touch()
    with pytest.raises(FileExistsError):
        run.audit(root=root)
    changed = contract | {"pricing_snapshot": contract["pricing_snapshot"] | {"hard_cost_cap_cny": 0}}
    with pytest.raises(RuntimeError, match="spend cap"):
        run.run_paid(root=tmp_path / "budget", client=client, authorization_confirmed=True,
            probe_loader=lambda: (changed, rows))
    for cap in (-1, True, 25):
        with pytest.raises(ValueError):
            run.run_paid(client=client, authorization_confirmed=True, max_new_calls=cap)
    with pytest.raises(RuntimeError, match="bounded client"):
        run.run_paid(authorization_confirmed=True)
    run.INPUTS.write_text("[]")
    with pytest.raises(RuntimeError, match="inputs changed"):
        run.load_probe()
    assert client.calls == 0


def test_wire_guard_pins_body_order_endpoint_timeout_and_24_call_cap(tmp_path, monkeypatch):
    contract, _, env = prepared(tmp_path, monkeypatch)
    sent = []
    def respond(request):
        assert request.extensions["timeout"]["read"] == 30
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})
    def transport(**kwargs):
        assert kwargs == {"retries": 0}
        return httpx.MockTransport(respond)
    monkeypatch.setattr(run.httpx, "HTTPTransport", transport)
    calls = run.prep.planned_calls()
    with run.bounded_client(contract, env) as client:
        assert client.max_retries == 0 and not client._client.follow_redirects
        first = calls[0][2]
        for url, body in ((URL + "/embeddings", first), (URL + "/chat/completions", calls[1][2]),
                (URL + "/chat/completions", first | {"expected": "SECRET"}),
                (URL + "/chat/completions", first | {"model": "other"})):
            with pytest.raises(RuntimeError, match="unapproved"):
                client._client.post(url, json=body)
        assert not sent
        duplicate = json.dumps(first).replace('"model": "', '"model": "other", "model": "', 1)
        with pytest.raises(ValueError, match="duplicate JSON field"):
            client._client.post(URL + "/chat/completions", content=duplicate, headers={"Content-Type": "application/json"})
        assert not sent
        for _, _, body in calls:
            parameters = dict(body)
            thinking = parameters.pop("enable_thinking")
            client.chat.completions.create(**parameters, extra_body={"enable_thinking": thinking})
        with pytest.raises(RuntimeError, match="cap"):
            client._client.post(URL + "/chat/completions", json=first)
    assert sent == [body for _, _, body in calls]
    with pytest.raises(RuntimeError, match="endpoint changed"):
        run.bounded_client(contract | {"endpoint_sha256": "changed"}, env)
