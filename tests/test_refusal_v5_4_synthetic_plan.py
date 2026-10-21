"""Fixed synthetic plan and label-free payload checks; zero real provider calls."""

import copy
import hashlib
import json

from experiments.evals import refusal_v5_4_synthetic_plan as prep


PLAN_SHA256 = "803eca9901216e7436066a0412721770aaa1bca4d3c3ae3ca0a476fe734a8a8d"


def test_frozen_plan_and_full_paired_payloads_without_network_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline plan attempted credentials or provider access")

    monkeypatch.setattr("dotenv.dotenv_values", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    result, calls = prep.plan(), prep.planned_calls()
    assert result["plan_sha256"] == PLAN_SHA256
    assert result["target_counts"] == {"SUFFICIENT": 6, "INSUFFICIENT": 6}
    assert len(calls) == result["max_calls"] == 24 and result["cases"] == 12
    assert result["provider_calls"] == 0 and not result["runner_ready"] and not result["promotion_allowed"]
    assert 0 < result["planning_cost_cny"] < result["proposed_accounting_cap_cny"] == 2
    assert max(result["input_byte_plus_framing_bounds"]) + 2048 < 256_000
    assert result["request_body_sha256"] == [prep.canonical(body) for _, _, body in calls]
    cases = json.loads(prep.CASES_PATH.read_text(encoding="utf-8"))
    for index, case in enumerate(cases):
        first, second = calls[index * 2:index * 2 + 2]
        assert first[0] == second[0] == case["id"] and first[1] != second[1]
        assert first[1] == ("v5_3_1" if index % 2 == 0 else "v5_4")
        assert first[2]["messages"][1:] == second[2]["messages"][1:]
        payload = json.loads(first[2]["messages"][1]["content"])
        assert set(payload) == {"question", "sources"} and payload["question"] == case["question"]
        assert len(payload["sources"]) == len(case["sources"])
        for actual, original in zip(payload["sources"], case["sources"]):
            assert actual["source_id"] == original["source_id"]
            assert "".join(s["content"] for s in actual["segments"]) == original["content"]
        assert case["expected"]["review"].strip()
        assert set(case["expected"]["coverage_statuses"]) == {"requested_outcomes", "applicability", "version_and_time"}
        for _, _, body in (first, second):
            assert body["max_tokens"] == 2048 and body["enable_thinking"] is False
            assert body["response_format"] == {"type": "json_object"}


def test_changed_local_rubrics_change_freeze_but_never_model_payload(tmp_path, monkeypatch):
    before = prep.planned_calls()
    cases = json.loads(prep.CASES_PATH.read_text(encoding="utf-8"))
    modified = copy.deepcopy(cases)
    for case in modified:
        case["expected"] = {"decision": "LOCAL_SECRET_LABEL", "review": "LOCAL_SECRET_RATIONALE"}
        case["family"] = "LOCAL_SECRET_METADATA"
    path = tmp_path / "fictional.json"
    path.write_text(json.dumps(modified), encoding="utf-8")
    monkeypatch.setattr(prep, "CASES_PATH", path)
    assert prep.planned_calls() == before
    assert prep.plan()["plan_sha256"] != PLAN_SHA256
    assert "LOCAL_SECRET" not in json.dumps(prep.planned_calls())


def test_protocol_binds_same_frozen_plan():
    text = prep.CASES_PATH.with_name("v5_4_synthetic_protocol.md").read_text(encoding="utf-8")
    assert PLAN_SHA256 in text and "FROZEN_PLAN_NOT_AUTHORIZED" in text
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "ae01a09561b97c27162367b2ba681cb5d410b7821d5f6d1e52013f5bd4423e24"
