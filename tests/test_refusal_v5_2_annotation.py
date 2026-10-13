import json
from types import SimpleNamespace

import pytest

from experiments.evals import refusal_v5_2_annotation as a


def setup_run():
    rows = [{"question_id": f"TRAIN_Q{i:03}", "question": "Question", "sources": [
        {"source_id": "Source 1", "content": "Supported fact.\nLimit."}
    ]} for i in range(3)]
    contract = {"model": a.MODEL, "max_calls": 5, "max_output_tokens": 4096,
                "hard_accounting_cap_cny": 8, "input_byte_bounds": [10000] * 3,
                "input_cny_per_million": 3.75, "output_cny_per_million": 22.5}
    annotation = {"annotation": {"target_class": "SUFFICIENT", "supporting_source_ids": ["Source 1"],
                                  "notes": "Requirement directly supported at Source 1:L1."}}
    return contract, rows, annotation


class Client:
    def __init__(self, raw=None, fail=False):
        self.calls = 0
        self.raw = raw or json.dumps(setup_run()[2])
        self.fail = fail
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls += 1
        assert kwargs["model"] == a.MODEL
        assert '"question_id"' not in json.dumps(kwargs["messages"])
        assert '"annotation"' not in kwargs["messages"][1]["content"]
        if self.fail:
            raise ConnectionError("SECRET SHOULD NOT BE LOGGED")
        return SimpleNamespace(id="request", usage=SimpleNamespace(prompt_tokens=100,
            completion_tokens=100, total_tokens=200), choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=self.raw))])


def test_resume_skips_successes_and_frozen_identity(tmp_path):
    contract, rows, _ = setup_run()
    client = Client()
    assert a.run(contract, rows, tmp_path, client=client, max_new_calls=1)["completed"] == 1
    assert a.run(contract, rows, tmp_path, client=client)["completed"] == 3
    assert client.calls == 3
    assert a.run(contract, rows, tmp_path, client=client)["completed"] == 3
    assert client.calls == 3
    with pytest.raises(RuntimeError, match="identity"):
        a.run(contract | {"model": "changed"}, rows, tmp_path, client=client)


@pytest.mark.parametrize("raw,unknown", [("{}", False), (None, True)])
def test_failures_saved_and_never_silently_retried(tmp_path, raw, unknown):
    contract, rows, _ = setup_run()
    client = Client(raw=raw, fail=unknown)
    with pytest.raises(RuntimeError, match="failure saved"):
        a.run(contract, rows, tmp_path, client=client)
    with pytest.raises(RuntimeError, match="unknown charge" if unknown else "resume-after-audit"):
        a.run(contract, rows, tmp_path, client=client)
    assert client.calls == 1
    assert "SECRET" not in (tmp_path / "attempts.jsonl").read_text()
    if not unknown:
        good = Client()
        assert a.run(contract, rows, tmp_path, client=good, resume_after_audit=True)["completed"] == 3
        assert good.calls == 3


def test_unfinished_request_blocks_and_saved_response_recovers(tmp_path):
    contract, rows, _ = setup_run()
    client = Client()
    a.run(contract, rows, tmp_path, client=client, max_new_calls=1)
    journal = tmp_path / "attempts.jsonl"
    event = json.loads(journal.read_text())
    # Simulate crash after raw response saved but before final append.
    journal.unlink()
    a.write_json(tmp_path / "pending.json", {k: event[k] for k in ("call_index", "question_id")})
    with pytest.raises(RuntimeError, match="outcome unknown"):
        a.run(contract, rows, tmp_path, client=client)
    assert client.calls == 1
    a.write_json(tmp_path / "response.json", {k: v for k, v in event.items() if k != "result"})
    assert a.run(contract, rows, tmp_path, client=client)["completed"] == 3
    assert client.calls == 3


def test_budget_and_bad_line_references_stop(tmp_path):
    contract, rows, value = setup_run()
    client = Client()
    with pytest.raises(RuntimeError, match="cap reached"):
        a.run(contract | {"hard_accounting_cap_cny": 0}, rows, tmp_path, client=client)
    assert client.calls == 0
    value["annotation"]["notes"] = "Source 1:L999"
    with pytest.raises(ValueError, match="line reference"):
        a.parse(json.dumps(value), rows[0], "stop")
    value["annotation"]["notes"] = "General assertion without evidence"
    with pytest.raises(ValueError, match="valid evidence"):
        a.parse(json.dumps(value), rows[0], "stop")
