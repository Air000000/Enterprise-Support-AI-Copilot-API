"""Fictional SDK/transport checks, including interrupted-response recovery."""

import json
import shutil

import httpx
import pytest

from experiments.evals import hybrid_contract_probe as probe


def setup(root):
    value = probe.prepare(root)
    return probe.wire.digest(value)


def delegates(*, all_supported=False, bad_gate=False, bad_answer=False, generation_length=False, provider_failure=False):
    calls = []
    by_question = {c["question"]: c for c in probe.fixtures.inputs()}
    expected = {name: (positions, target) for name, _, positions, target, _ in probe.fixtures.FIXTURES}

    def respond(request):
        body = json.loads(request.content)
        stage = "classifier" if "response_format" in body else "generation"
        payload = json.loads(body["messages"][1]["content"])
        case = by_question[payload["question"]]
        calls.append((stage, body))
        assert not any(k in json.dumps(payload) for k in ("rubric", "SUFFICIENT", "expected"))
        if provider_failure:
            return httpx.Response(400, json={"error": {"message": "PRIVATE_PROVIDER_ERROR"}})
        if stage == "classifier":
            positions, target = expected[case["id"]]
            supported = all_supported or target == "SUFFICIENT"
            item = dict(status="SUPPORTED" if supported else "MISSING", reason="Fictional test response.",
                evidence=[dict(source_id=f"Source {i}", segment_ids=[1]) for i in positions] if supported else [])
            content = dict(requirements=[dict(requirement="Answer the fictional question", **item)],
                coverage_checks=dict(requested_outcomes=item, applicability=item,
                    version_and_time=dict(status="NOT_REQUIRED", evidence=[], reason="Synthetic fixture only.")),
                unresolved_ambiguities=[])
            if bad_gate:
                del content["unresolved_ambiguities"]
            content = json.dumps(content)
        else:
            source = "Source 99" if bad_answer else payload["allowed_citation_ids"][0]
            content = f"Fictional test answer. [{source}]"
        truncated = stage == "generation" and generation_length
        output = 1024 if truncated else 20
        return httpx.Response(200, json={"id": "synthetic", "object": "chat.completion", "model": body["model"],
            "choices": [dict(index=0, finish_reason="length" if truncated else "stop", message=dict(role="assistant", content=content))],
            "usage": dict(prompt_tokens=100, completion_tokens=output, total_tokens=100 + output)})
    return {stage: httpx.MockTransport(respond) for stage in probe.ENDPOINTS}, calls


def run(root, sha, transport, **options):
    return probe.execute(root, {s: "OFFLINE_TEST_CREDENTIAL" for s in probe.ENDPOINTS}, sha,
        "OFFLINE_TEST_ONLY_NOT_PAID_CONSENT", confirmed=True, delegates=transport, **options)


def test_plan_is_fixed_and_excludes_expectations_from_requests():
    value = probe.plan()
    assert value == probe.plan()
    manifest = probe.read(probe.Path(probe.__file__).with_name("reports") / "hybrid_contract_probe/plan.json")
    assert manifest["sources"] == value["sources"]
    assert manifest["cases_sha256"] == value["cases_sha256"]
    assert manifest["classifier_request_sha256"] == value["classifier_request_sha256"]
    assert len(value["cases"]) == 12 and all(len(c["sources"]) == 14 for c in value["cases"])
    assert value["max_total_calls"] == 48
    assert probe.qa.Decimal(value["planning_full_reservation_cny"]) < probe.qa.Decimal(value["accounting_cap_cny"])
    assert value["order"][:4] == [["single", probe.POLICIES[0]], ["single", probe.POLICIES[1]],
        ["late_source", probe.POLICIES[1]], ["late_source", probe.POLICIES[0]]]
    assert all(set(c) == {"id", "question", "sources"} for c in value["cases"])
    from rag_runtime import hybrid_ask as h
    for case in value["cases"]:
        for policy in probe.POLICIES:
            request = probe.body(case, policy, "classifier")
            assert request["messages"] == h.POLICY_BUILDERS[policy](probe.runtime_input(case))
            assert probe.wire.digest(request) == value["classifier_request_sha256"][f"{case['id']}:{policy}"]


def test_frozen_synthetic_run_and_resume_never_resend(tmp_path):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates()
    first = run(root, sha, transport, max_cases=1)
    assert first["outcomes"] == 1 and len(calls) == 2
    with pytest.raises(ValueError, match="All24"):
        probe.report(root, sha)
    final = run(root, sha, transport)
    assert final["outcomes"] == 24 and len(calls) == 40  #24 classifications +16 actual admissions.
    assert run(root, sha, transport) == final and len(calls) == 40
    report = probe.report(root, sha)
    assert report["completed_provider_calls"] == 40 and report["quality_review_pending"] is True
    assert all(a["expected_gate_matches"] == 12 and a["cases"] == 12 for a in report["arms"].values())
    assert len(calls) == 40


def test_generation_depends_on_actual_gate_not_expected_labels(tmp_path):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates(all_supported=True)
    result = run(root, sha, transport)
    assert result["completed_provider_calls"] == len(calls) == 48
    assert sum(stage == "generation" for stage, _ in calls) == 24


@pytest.mark.parametrize("kind", ["gate", "citation", "length"])
def test_known_validation_failures_retained_without_retry(tmp_path, kind):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates(bad_gate=kind == "gate", bad_answer=kind == "citation", generation_length=kind == "length")
    result = run(root, sha, transport)
    assert result["outcomes"] == 24
    assert len(calls) == (24 if kind == "gate" else 40)
    report = probe.report(root, sha)
    assert sum(a["validation_failures"] for a in report["arms"].values()) == (24 if kind == "gate" else 16)
    assert not any(root.rglob("pending.json"))


def test_receipts_survive_interrupt_before_outcome_write(tmp_path, monkeypatch):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates()
    keep = probe.reuse.keep

    def interrupt(path, value):
        if path.name == "result.json":
            raise OSError("synthetic interruption after response receipts")
        return keep(path, value)
    with monkeypatch.context() as scoped:
        scoped.setattr(probe.reuse, "keep", interrupt)
        with pytest.raises(OSError):
            run(root, sha, transport, max_cases=1)
    assert len(calls) == 2 and not (root / "run.lock").exists()
    assert run(root, sha, transport, max_cases=1)["outcomes"] == 1
    assert len(calls) == 2


def test_provider_failure_blocks_every_later_send_and_keeps_pending(tmp_path):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates(provider_failure=True)
    with pytest.raises(RuntimeError, match="audit required"):
        run(root, sha, transport)
    assert len(calls) == 1 and len(list(root.rglob("pending.json"))) == 1
    with pytest.raises(RuntimeError, match="Pending/unknown"):
        run(root, sha, transport)
    assert len(calls) == 1
    assert "PRIVATE_PROVIDER_ERROR" not in "".join(p.read_text(encoding="utf-8") for p in root.rglob("*.json"))


@pytest.mark.parametrize("tamper", ["plan", "result", "receipt", "extra_case", "missing_receipts", "deleted_generation"])
def test_changed_checkpoint_is_not_silently_adopted(tmp_path, tamper):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates()
    run(root, sha, transport, max_cases=1)
    if tamper == "extra_case":
        (root / "cases/unapproved").mkdir()
    elif tamper == "missing_receipts":
        shutil.rmtree(next(root.glob("cases/*/wire")))
    elif tamper == "deleted_generation":
        folder = next(root.glob("cases/*/wire"))
        path = folder / "attempts.jsonl"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        (folder / "requests/1.json").unlink()
        (folder / "responses/1.json").unlink()
    else:
        path = (root / "plan.json" if tamper == "plan" else next(root.glob("cases/*/result.json"))
                if tamper == "result" else next(root.glob("cases/*/wire/responses/*.json")))
        value = probe.read(path)
        if tamper == "receipt":
            value["wire_utf8"] += " "
            # Canonical JSON hashes ignore whitespace, so change a real response field.
            body = json.loads(value["wire_utf8"])
            body["choices"][0]["message"]["content"] = "Changed receipt"
            value["wire_utf8"] = json.dumps(body)
        else:
            value["changed"] = True
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)):
        run(root, sha, transport)
    assert len(calls) == 2


def test_refuses_missing_consent_hash_and_changed_billing_before_send(tmp_path, monkeypatch):
    root = tmp_path / "probe"
    sha = setup(root)
    transport, calls = delegates()
    with pytest.raises(ValueError, match="human"):
        probe.execute(root, {}, sha, "", delegates=transport)
    with pytest.raises(ValueError, match="identity"):
        run(root, "wrong-hash", transport)
    with monkeypatch.context() as scoped:
        scoped.setattr(probe, "CAP", "0.001")
        with pytest.raises(ValueError, match="cap"):
            run(root, sha, transport)
    assert calls == []
