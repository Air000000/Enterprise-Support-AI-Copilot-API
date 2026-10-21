"""PR104 thin execution adapter; default is zero-call preflight, not authorization."""

import argparse
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import httpx
from openai import OpenAI

from experiments.evals import refusal_v5_4_synthetic_plan as prep
from experiments.evals import refusal_v5_2_control_runner as control
from experiments.evals import refusal_v5_3_1_runner as reuse

PLAN_SHA256 = "803eca9901216e7436066a0412721770aaa1bca4d3c3ae3ca0a476fe734a8a8d"
PROTOCOL_SHA256 = "ae01a09561b97c27162367b2ba681cb5d410b7821d5f6d1e52013f5bd4423e24"
DATA = Path("data/refusal_v5_4_synthetic_probe")
CONTRACT = DATA / "contract.json"
INPUTS = DATA / "inputs.json"
ROOT = DATA / "run_v1"
paired = control.paired


def checked_plan():
    value = prep.plan()
    protocol = prep.CASES_PATH.with_name("v5_4_synthetic_protocol.md").read_text(encoding="utf-8")
    if value["plan_sha256"] != PLAN_SHA256 or hashlib.sha256(protocol.encode()).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("frozen PR104 plan/protocol changed")
    return value


def rows_from_plan():
    # ponytail: project the approved wire payload, never copy expected/rubric fields.
    return [{"question_id": name, "question": payload["question"], "sources": [
        {"source_id": source["source_id"], "content": "".join(part["content"] for part in source["segments"])}
        for source in payload["sources"]]}
        for name, _, body in prep.planned_calls()[::2]
        for payload in [json.loads(body["messages"][1]["content"])]]


def execution_contract(endpoint_sha256, authorization_reference):
    plan = checked_plan()
    if not isinstance(authorization_reference, str) or not authorization_reference.strip():
        raise RuntimeError("explicit human authorization reference required")
    bounds = {arm: [plan["input_byte_plus_framing_bounds"][i] for i, (_, a) in
        enumerate(plan["call_order"]) if a == arm] for arm in prep.BUILDERS}
    modules = (control, paired, paired.shared, reuse)
    return {"run": plan["run"], "plan_sha256": PLAN_SHA256,
        "authorization_reference": authorization_reference, "endpoint_sha256": endpoint_sha256,
        "adapter_source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "engine_source_sha256": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_text(
            encoding="utf-8").encode()).hexdigest() for m in modules},
        "dependencies": {name: importlib.metadata.version(name) for name in ("openai", "httpx")},
        "inputs_sha256": prep.canonical(rows_from_plan()), "request_body_sha256": plan["request_body_sha256"],
        "classifier": {"model": plan["model"]}, "max_calls_total": 24,
        "arms": {arm: {"max_calls": 12, "max_output_tokens": 2048,
            "request_input_byte_bounds": bounds[arm]} for arm in prep.BUILDERS},
        "pricing_snapshot": {"input_cny_per_1m_tokens": 3, "output_cny_per_1m_tokens": 18,
            "hard_cost_cap_cny": 2},
        "controls": {"failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 1},
        "timeout_seconds": 30, "promotion_allowed": False}


def require_authorization(confirmed):
    if confirmed is not True:
        raise RuntimeError("explicit PR104 paid authorization required")
    if datetime.now(timezone.utc).date() >= date(2027, 1, 19):
        raise RuntimeError("frozen model retirement date reached; no automatic replacement")


def prepare_contract(env_file, authorization_reference, *, authorization_confirmed=False):
    require_authorization(authorization_confirmed)
    checked_plan()
    if DATA.exists():
        raise RuntimeError("existing execution namespace: audit, never overwrite")
    if env_file is None or not Path(env_file).is_file():
        raise RuntimeError("explicit existing local credential file required")
    _, url = reuse.resolve_credentials(env_file)
    contract = execution_contract(hashlib.sha256(url.encode()).hexdigest(), authorization_reference)
    DATA.mkdir()
    paired.write_json(INPUTS, rows_from_plan())
    paired.write_json(CONTRACT, contract)
    return contract


def load_probe():
    contract = paired.shared._read_json(CONTRACT)
    if contract != execution_contract(contract["endpoint_sha256"], contract["authorization_reference"]):
        raise RuntimeError("execution contract changed")
    rows = json.loads(INPUTS.read_text(encoding="utf-8"))
    if prep.canonical(rows) != contract["inputs_sha256"] or rows != rows_from_plan():
        raise RuntimeError("execution inputs changed")
    return contract, rows


def parse_result(arm, raw, row, finish):
    if arm not in prep.BUILDERS or finish != "stop":
        raise ValueError("invalid arm or incomplete classifier response")
    return prep.candidate.validate_requirement_coverage_response(raw, paired.shared._classifier_input_from_row(row))


class ResponseFailure(RuntimeError):
    """A failed response is durable; audit may inspect it, execution must stop."""


def stop_on_failure(*args, **kwargs):
    state = paired.load_state(*args, **kwargs)
    if any(event["result"] is None for event in state[0]):
        raise ResponseFailure("failed response durably saved; no further calls")
    return state


# ponytail: isolate existing engines; only tighten failure stopping, not the old experiments.
engine = reuse.isolate(control, "v54_fictional")
engine.BUILDERS, engine.parse_result = prep.BUILDERS, parse_result
engine.paired = reuse.isolate(paired, "v54_fictional_state")
engine.paired.load_state = stop_on_failure


def read_state(root, contract, rows):
    return paired.load_state(root, rows, contract, builders=prep.BUILDERS,
        result_parser=parse_result, contract_sha256=control.controls.prep.prior.canonical_sha256(contract))


def audit(*, root=ROOT, probe_loader=load_probe):
    contract, rows = probe_loader()
    try:
        engine.run_paid(root=root, client=None, max_new_calls=0, probe_loader=lambda: (contract, rows))
    except ResponseFailure:
        pass  # Inspect accounted failures without allowing another request.
    return read_state(root, contract, rows)


def run_paid(*, client=None, root=ROOT, max_new_calls=24, authorization_confirmed=False,
             resume_after_audit=False, probe_loader=load_probe):
    require_authorization(authorization_confirmed)
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 24:
        raise ValueError("call cap must be 0..24")
    if client is None:
        raise RuntimeError("explicit bounded client required; no fallback credentials")
    root = Path(root)
    if (root.exists() or root.with_name(root.name + "_raw_response.json").exists()) and not resume_after_audit:
        raise RuntimeError("existing state: new authorization and explicit resume-after-audit required")
    return engine.run_paid(root=root, client=client, max_new_calls=max_new_calls, probe_loader=probe_loader)


def bounded_client(contract, env_file, *, start_index=0):
    if env_file is None or not Path(env_file).is_file():
        raise RuntimeError("explicit existing local credential file required")
    key, url = reuse.resolve_credentials(env_file)
    if hashlib.sha256(url.encode()).hexdigest() != contract["endpoint_sha256"]:
        raise RuntimeError("frozen Singapore endpoint changed")
    if type(start_index) is not int or not 0 <= start_index <= 24:
        raise ValueError("invalid starting call index")
    index = start_index

    def guard(request):
        nonlocal index
        if (index >= 24 or request.method != "POST" or str(request.url) != url + "/chat/completions"
                or prep.canonical(json.loads(request.content, object_pairs_hook=paired.structured._unique_object))
                != contract["request_body_sha256"][index]):
            raise RuntimeError("unapproved body, order, endpoint or exhausted call cap")
        index += 1

    return OpenAI(api_key=key, base_url=url, max_retries=0, timeout=30,
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=30,
            transport=httpx.HTTPTransport(retries=0), event_hooks={"request": [guard]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="preflight", choices=("preflight", "prepare", "run", "audit"))
    parser.add_argument("--confirm-authorization", action="store_true", help="Records prior human consent; not permission itself")
    parser.add_argument("--authorization-reference")
    parser.add_argument("--env-file")
    parser.add_argument("--resume-after-audit", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            value = checked_plan()
            print(json.dumps({"plan_sha256": value["plan_sha256"], "provider_calls": 0, "authorization_required": True}))
        elif args.mode == "prepare":
            value = prepare_contract(args.env_file, args.authorization_reference,
                authorization_confirmed=args.confirm_authorization)
            print(json.dumps({"contract_sha256": prep.canonical(value), "provider_calls": 0}))
        elif args.mode == "audit":
            events, completed, counts, spent = audit()
            print(json.dumps({"recorded_calls": len(events), "completed": completed, "estimated_cost_cny": spent,
                "arms": counts, "provider_calls_this_command": 0, "unattempted": 24 - len(events),
                "promotion_allowed": False}))
        else:
            require_authorization(args.confirm_authorization)
            contract, rows = load_probe()
            existing = ROOT.exists() or ROOT.with_name(ROOT.name + "_raw_response.json").exists()
            if existing and not args.resume_after_audit:
                raise RuntimeError("existing state requires explicit resume-after-audit")
            start = audit()[1] if existing else 0
            with bounded_client(contract, args.env_file, start_index=start) as client:
                events = run_paid(client=client, authorization_confirmed=True,
                    resume_after_audit=args.resume_after_audit)
            print(json.dumps({"recorded_calls": len(events), "quality_review_pending": True, "promotion_allowed": False}))
    except Exception as error:
        print(f"Stopped ({type(error).__name__}); audit saved state, no automatic retry.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
