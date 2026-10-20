import json

import pytest

from experiments.evals import refusal_v5_3_1_draft_review as amendment
from tests.test_refusal_v5_2_annotation import Client, setup_run


def setup_amendment(tmp_path, monkeypatch):
    run = amendment.run
    contract, short, annotation = setup_run()
    rows = [{**short[0], "question_id": f"TRAIN_Q{i:03}"} for i in range(50)]
    contract.update(max_calls=50, input_byte_bounds=[10000] * 50, inputs_sha256="fake-input")
    original = tmp_path / "original"
    original.mkdir()
    run.annotation._engine.run(contract, rows[:3], original, client=Client())
    events = run.shared._read_jsonl(original / "attempts.jsonl")
    for event in events:
        event.update(raw_response="{}", result=None)
    (original / "attempts.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    run.paired.write_json(original / "stop.json", {
        "annotation_contract_sha256": run.canonical(contract),
        "journal_sha256": run.shared._sha256(original / "attempts.jsonl"),
        "citation_recoveries_sha256": run.shared._sha256(original / "identity.json")})
    (original / "citation_recoveries.jsonl").write_bytes((original / "identity.json").read_bytes())
    for name, path in (("ROOT", tmp_path / "drafts"), ("CONTRACT", tmp_path / "amendment.json"),
            ("STOP", original / "stop.json")):
        monkeypatch.setattr(amendment, name, path)
    for name, path in (("DATA", tmp_path), ("ANNOTATION_ROOT", original), ("ROOT", tmp_path / "paired"),
            ("FREEZE", tmp_path / "freeze.json"), ("CONTRACT", tmp_path / "classification.json")):
        monkeypatch.setattr(run, name, path)
    monkeypatch.setattr(run, "load_probe", lambda stage: (contract, rows))
    monkeypatch.setattr(run, "read_plan", lambda: {"provenance": "synthetic test"})
    return rows, annotation, (original / "attempts.jsonl").read_bytes()


def test_migration_preserves_charged_raws_invalid_drafts_advance_without_retry(tmp_path, monkeypatch):
    rows, _, raw = setup_amendment(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="approval"):
        amendment.prepare()
    assert amendment.prepare(authorization_confirmed=True)["completed"] == 3
    assert (amendment.run.ANNOTATION_ROOT / "attempts.jsonl").read_bytes() == raw
    client = Client(raw="{}")
    state = amendment.run_drafts(client=client)
    assert state["completed"] == state["calls"] == 50 and client.calls == 47
    assert amendment.run_drafts(client=client)["calls"] == 50 and client.calls == 47
    events = amendment.run.shared._read_jsonl(amendment.ROOT / "attempts.jsonl")
    assert [e["question_id"] for e in events] == [r["question_id"] for r in rows]
    assert all(e["result"] == {"question_id": e["question_id"], "schema_valid": False,
        "strict_annotation": None, "reviewed_label": False} for e in events)
    assert state["estimated_cost_cny"] == pytest.approx(sum(e["estimated_cost_cny"] for e in events))
    with pytest.raises(RuntimeError, match="already exists"):
        amendment.prepare(authorization_confirmed=True)
    with pytest.raises(ValueError):
        amendment.run_drafts(max_new_calls=48)
    with pytest.raises(RuntimeError, match="bounded client"):
        amendment.run_drafts()


def test_unknown_charge_pending_lock_budget_and_migration_tamper_block_sends(tmp_path, monkeypatch):
    setup_amendment(tmp_path, monkeypatch)
    amendment.prepare(authorization_confirmed=True)
    root, run = amendment.ROOT, amendment.run
    good = Client()
    run.paired.write_json(root / "pending.json", {"call_index": 3, "question_id": "TRAIN_Q003"})
    with pytest.raises(RuntimeError, match="outcome unknown"):
        amendment.run_drafts(client=good)
    (root / "pending.json").unlink()
    (root / "runner.lock").touch()
    with pytest.raises(FileExistsError):
        amendment.run_drafts(client=good)
    (root / "runner.lock").unlink()
    assert good.calls == 0
    bad = Client(fail=True)
    with pytest.raises(RuntimeError, match="failure saved"):
        amendment.run_drafts(client=bad)
    with pytest.raises(RuntimeError, match="provider exception"):
        amendment.run_drafts(client=bad)
    assert bad.calls == 1 and "SECRET" not in (root / "attempts.jsonl").read_text()
    (run.ANNOTATION_ROOT / "attempts.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="original charged stop"):
        amendment.run_drafts(client=good)
    assert good.calls == 0


def test_review_requires_all_drafts_and_strict_evidence_before_prediction(tmp_path, monkeypatch):
    rows, annotation, _ = setup_amendment(tmp_path, monkeypatch)
    amendment.prepare(authorization_confirmed=True)
    path = tmp_path / "review.jsonl"
    with pytest.raises(RuntimeError, match="all50"):
        amendment.freeze_review(path)
    amendment.run_drafts(client=Client(raw="{}"))
    review = [{"question_id": row["question_id"], **annotation} for row in rows]
    path.write_text("".join(json.dumps(r) + "\n" for r in review[:-1]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="complete ordered"):
        amendment.freeze_review(path)
    invalid = review[:1] + [{**r, "annotation": {**r["annotation"], "notes": "Source 1:L999"}} for r in review[1:]]
    path.write_text("".join(json.dumps(r) + "\n" for r in invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="line reference"):
        amendment.freeze_review(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in review), encoding="utf-8")
    amendment.run.ROOT.mkdir()
    with pytest.raises(RuntimeError, match="classifier exists"):
        amendment.freeze_review(path)
    amendment.run.ROOT.rmdir()
    freeze = amendment.freeze_review(path)
    assert freeze["cases"] == freeze["provider_calls"] == freeze["original_schema_failures"] == 50
    assert freeze["counts"] == {"SUFFICIENT": 50, "INSUFFICIENT": 0, "AMBIGUOUS": 0}
    assert not freeze["class_support_at_least_five_each"] and not freeze["classifier_predictions_created"]
    with pytest.raises(RuntimeError, match="overwrite"):
        amendment.freeze_review(path)
    with pytest.raises(ValueError, match="provider"):
        amendment.draft_outcome(None, rows[0], None)
    assert amendment.draft_outcome("{}", rows[0], "length")["strict_annotation"] is None


def test_published_complete_result_keeps_failures_and_does_not_promote():
    run = amendment.run
    old = run.shared._read_json(run.ANNOTATION_CONTRACT)
    contract = run.shared._read_json(amendment.CONTRACT)
    freeze = run.shared._read_json(run.FREEZE)
    paid = run.shared._read_json(run.CONTRACT)
    result = run.shared._read_json(run.RESULT)
    assert {k: v for k, v in contract.items() if k not in {"run", "draft_amendment"}} == {
        k: v for k, v in old.items() if k != "run"}
    assert contract["draft_amendment"]["old_contract_sha256"] == run.canonical(old)
    assert freeze["amendment_contract_sha256"] == run.canonical(contract)
    assert paid["freeze_sha256"] == run.canonical(freeze)
    assert result["contract_sha256"] == run.canonical(paid)
    assert result["targets_sha256"] == paid["targets_sha256"] == freeze["targets_sha256"]
    assert freeze["provider_calls"] == 50 and result["provider_calls"] == 100
    assert freeze["counts"] == result["primary"]["target_counts"] == {
        "SUFFICIENT": 39, "INSUFFICIENT": 11, "AMBIGUOUS": 0}
    assert freeze["original_schema_failures"] == 23 and freeze["changes_from_raw_draft"]["target_class"] == 2
    assert freeze["estimated_annotation_cost_cny"] + result["estimated_classification_cost_cny"] == pytest.approx(4.6080765)
    for arm in result["primary"]["arms"].values():
        assert sum(arm["confusion"].values()) + arm["binary_failures"] == 50
        assert arm["valid_correct_binary"] == 43 and arm["binary_failures"] == 2
        assert arm["binary_agreement_including_failures"] == .86
    assert result["common_valid_binary"]["cases"] == 46
    assert result["paired_binary_transitions"] == {"both_agree": 40, "baseline_only": 1,
        "candidate_only": 3, "neither_agrees": 2}
    assert not freeze["classifier_predictions_created"] and not result["dev_opened"]
    assert not result["promotion_allowed"] and result["formal_pass_gate"] is None
    for value in (contract, freeze, paid, result):
        assert not any(key in json.dumps(value) for key in ('"question":', '"raw_response":', '"annotation":', '"api_key":'))
