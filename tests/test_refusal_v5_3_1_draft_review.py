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
