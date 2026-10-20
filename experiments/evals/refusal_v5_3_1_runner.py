"""Authorized prospective50 adapter; durable engines reused, no serving changes."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

from experiments.evals import refusal_v5_3_1_preparation as prep
from experiments.evals import refusal_v5_2_control_runner as control
from experiments.evals import refusal_natural_diagnostic_runner as metrics

shared, canonical, paired = prep.shared, prep.canonical, control.paired
DATA = prep.DATA
ROOT = DATA / "paired_run_v1"
ANNOTATION_ROOT = DATA / "blind_annotation"
REPORT = prep.REPORT.parent
ANNOTATION_CONTRACT = REPORT / "v5_3_1_prospective_annotation_contract.json"
CONTRACT = REPORT / "v5_3_1_prospective_run_contract.json"
FREEZE = REPORT / "v5_3_1_prospective_annotation_freeze.json"
RESULT = REPORT / "v5_3_1_prospective_result.json"
PLAN_SHA256 = "daac4754916020b880e6d6d660e99b0d806df8561cff3d67e29e9eb7a786de49"
AUTHORIZATION = "User explicitly selected both PR99 stages: frozen50 TRAIN/full Top14 to Singapore; <=50 qwen3.6-plus-2026-04-02 annotations/4096tokens/60s/CNY10; assistant review and complete label freeze before <=100 qwen3.5-plus-2026-04-20 classifications/2048tokens/30s/CNY15; total<=150/CNY25 accounting, not invoice guarantee. No paid retries, DEV, retrieval, answers, tickets, deployment, recharge or promotion."


def isolate(module, name):
    # ponytail: isolate three existing adapters; do not fork their checkpoint algorithms.
    spec = importlib.util.spec_from_file_location("_prospective_v531_" + name, module.__file__)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


annotation = isolate(prep.readiness, "annotation")
engine = isolate(control, "paired")
metric_adapter = isolate(metrics, "metrics")
metric_adapter.prep = SimpleNamespace(BUILDERS=prep.BUILDERS)
annotation.DATA, annotation.ROOT = DATA, ANNOTATION_ROOT
annotation.CONTRACT, annotation.FREEZE = ANNOTATION_CONTRACT, FREEZE
engine.BUILDERS = prep.BUILDERS


def read_plan():
    value = shared._read_json(prep.REPORT)
    if value["plan_canonical_sha256"] != PLAN_SHA256 or canonical({k: v for k, v in value.items() if k != "plan_canonical_sha256"}) != PLAN_SHA256:
        raise RuntimeError("frozen plan changed")
    paths = {Path("experiments/evals") / name: digest for name, digest in value["source_hashes"].items()}
    paths[prep.PROTOCOL] = value["protocol_normalized_sha256"]
    if any(hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() != digest for path, digest in paths.items()):
        raise RuntimeError("frozen source/protocol changed")
    return value


def load_rows():
    plan = read_plan()
    _, rows = prep.readiness.load_inputs()
    return plan, rows


def body(stage, row, arm=None):
    messages = prep.prior.annotation.messages(row) if stage == "annotation" else prep.BUILDERS[arm](shared._classifier_input_from_row(row))
    return {"model": "qwen3.6-plus-2026-04-02" if stage == "annotation" else "qwen3.5-plus-2026-04-20",
        "temperature": 0, "enable_thinking": False, "response_format": {"type": "json_object"},
        "max_tokens": 4096 if stage == "annotation" else 2048, "messages": messages}


def runtime_bindings():
    modules = (control, metrics, prep.readiness, prep.prior.annotation)
    return {"source_hashes": {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_text(encoding="utf-8").encode()).hexdigest() for m in modules},
        "runner_source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("openai", "httpx")}}


def execution_contract(stage, plan, rows, endpoint_sha256):
    arms = plan["preflight"]["arms"]
    value = {"run": "prospective_v5_3_1_" + stage + "_v1", "stage": stage, "cases": 50,
        "plan_sha256": PLAN_SHA256, "inputs_sha256": plan["inputs_sha256"], "case_order": plan["case_order"],
        "endpoint_sha256": endpoint_sha256,
        "authorization": AUTHORIZATION, "bindings": runtime_bindings(),
        "request_body_sha256": [canonical(body(stage, row, arm)) for row, arm in
            ([(row, None) for row in rows] if stage == "annotation" else paired.schedule(rows, tuple(prep.BUILDERS)))],
        "timeout_seconds": 60 if stage == "annotation" else 30, "sdk_retries": 0,
        "controls": {"dev_opened": False, "promotion_allowed": False, "unknown_charge_blocks_resume": True,
            "failed_pair_is_terminal": True, "max_failed_attempts_per_arm": 50, "sdk_retries": 0}}
    if stage == "annotation":
        value.update(model="qwen3.6-plus-2026-04-02", max_calls=50, max_output_tokens=4096,
            hard_accounting_cap_cny=10, input_cny_per_million=3.75, output_cny_per_million=22.5,
            input_byte_bounds=arms["annotation"]["input_byte_plus_framing_bounds"])
    else:
        freeze = shared._read_json(FREEZE)  # Metadata only; no target rows in the paid loader.
        if freeze["cases"] != 50 or freeze["inputs_sha256"] != plan["inputs_sha256"] or freeze["classifier_predictions_created"]:
            raise RuntimeError("complete preprediction annotation freeze required")
        value.update(max_calls_total=100, classifier={"model": "qwen3.5-plus-2026-04-20"},
            targets_sha256=freeze["targets_sha256"], target_counts=freeze["counts"], freeze_sha256=canonical(freeze),
            arms={arm: {"max_calls": 50, "max_output_tokens": 2048,
                "request_input_byte_bounds": arms[arm]["input_byte_plus_framing_bounds"]} for arm in prep.BUILDERS},
            pricing_snapshot={"input_cny_per_1m_tokens": 3, "output_cny_per_1m_tokens": 18, "hard_cost_cap_cny": 15})
    return value


def prepare_contract(stage, *, authorization_confirmed=False, env_file=None):
    if authorization_confirmed is not True:
        raise RuntimeError("explicit two-stage authorization required")
    if env_file is None:
        raise RuntimeError("explicit credential file required for endpoint binding")
    _, url = resolve_credentials(env_file)
    plan, rows = load_rows()
    path = ANNOTATION_CONTRACT if stage == "annotation" else CONTRACT
    if stage == "annotation":
        if plan != prep.build_preparation() or path.exists():
            raise RuntimeError("existing state or changed preparation")
        DATA.mkdir()  # New namespace only; never overwrite old private inputs.
        shutil.copyfile(plan["inputs_path"], DATA / "inputs.jsonl")
    elif shared._sha256(DATA / "targets.jsonl") != shared._read_json(FREEZE)["targets_sha256"]:
        raise RuntimeError("target freeze changed before execution contract")
    value = execution_contract(stage, plan, rows, hashlib.sha256(url.encode()).hexdigest())
    prep.prior.selection.prior._write_json(path, value)
    return value


def load_probe(stage):
    plan, rows = load_rows()
    contract = shared._read_json(ANNOTATION_CONTRACT if stage == "annotation" else CONTRACT)
    if contract != execution_contract(stage, plan, rows, contract["endpoint_sha256"]):
        raise RuntimeError("execution contract changed")
    if shared._sha256(DATA / "inputs.jsonl") != plan["inputs_sha256"]:
        raise RuntimeError("local input copy changed")
    return contract, rows


def resolve_credentials(env_file):
    from dotenv import dotenv_values
    config = dotenv_values(env_file)
    key, url = config.get("DASHSCOPE_PHASE_B_API_KEY"), (config.get("DASHSCOPE_PHASE_B_BASE_URL") or "").rstrip("/")
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if not key or not (parsed.scheme == "https" and (host == "dashscope-intl.aliyuncs.com" or host.endswith(".ap-southeast-1.maas.aliyuncs.com"))
            and parsed.path == "/compatible-mode/v1" and parsed.port is None
            and not any((parsed.username, parsed.password, parsed.query, parsed.fragment))):
        raise RuntimeError("dedicated Singapore chat credentials required")
    return key, url


def bounded_client(stage, contract, env_file):
    key, url = resolve_credentials(env_file)
    if hashlib.sha256(url.encode()).hexdigest() != contract["endpoint_sha256"]:
        raise RuntimeError("frozen endpoint changed")
    allowed = set(contract["request_body_sha256"])
    def guard(request):
        if request.method != "POST" or str(request.url) != url + "/chat/completions" or canonical(json.loads(request.content)) not in allowed:
            raise RuntimeError("unapproved request blocked before send")
    return OpenAI(api_key=key, base_url=url, max_retries=0, timeout=contract["timeout_seconds"],
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=contract["timeout_seconds"],
            transport=httpx.HTTPTransport(retries=0), event_hooks={"request": [guard]}))


def parse_result(arm, raw, row, finish_reason):
    if arm not in prep.BUILDERS or finish_reason != "stop":
        raise ValueError("invalid arm or incomplete response")
    return prep.candidate.validate_requirement_coverage_response(raw, shared._classifier_input_from_row(row))


engine.parse_result = parse_result


def run_annotation(*, root=ANNOTATION_ROOT, client=None, max_new_calls=50, probe_loader=None):
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 50:
        raise ValueError("annotation call cap must be 0..50")
    if max_new_calls and client is None:
        raise RuntimeError("explicit bounded client required")
    contract, rows = (probe_loader or (lambda: load_probe("annotation")))()
    return annotation._engine.run(contract, rows, root, client=client, max_new_calls=max_new_calls)


annotation.run_annotation = run_annotation
annotation.load_probe = lambda: load_probe("annotation")


def export_draft():
    _, _, _, draft = annotation.accounted_draft(root=ANNOTATION_ROOT)
    prep.prior.selection.prior._write_jsonl(DATA / "blind_draft_annotations.jsonl", draft)
    return len(draft)


def freeze_review(path):
    return annotation.freeze_review(path, root=ANNOTATION_ROOT)


def run_classification(*, root=ROOT, client=None, max_new_calls=100, probe_loader=None):
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 100:
        raise ValueError("classification call cap must be 0..100")
    if max_new_calls and client is None:
        raise RuntimeError("explicit bounded client required")
    return engine.run_paid(root=root, client=client, max_new_calls=max_new_calls,
        probe_loader=probe_loader or (lambda: load_probe("classification")))


def evaluate():
    contract, rows = load_probe("classification")
    events, completed, _, spent = paired.load_state(ROOT, rows, contract, builders=prep.BUILDERS,
        result_parser=parse_result, contract_sha256=canonical(contract))
    if completed != 100 or (ROOT / "runner.lock").exists():
        raise RuntimeError("all100 terminal outcomes and stopped runner required")
    targets = DATA / "targets.jsonl"
    if shared._sha256(targets) != contract["targets_sha256"]:
        raise RuntimeError("frozen targets changed")
    labels = shared._read_jsonl(targets)  # Only after the complete paid loop.
    if [t["question_id"] for t in labels] != contract["case_order"]:
        raise RuntimeError("target order changed")
    outcomes = {(e["question_id"], e["arm"]): e["result"] for e in events}
    transitions = [t | {a: outcomes[t["question_id"], a]["decision"] if outcomes[t["question_id"], a] else None for a in prep.BUILDERS} for t in labels]
    primary = metric_adapter.population_metrics(transitions)
    if primary["target_counts"] != contract["target_counts"]:
        raise RuntimeError("target class counts changed")
    common = [t for t in transitions if t["target_class"] != "AMBIGUOUS" and all(t[a] is not None for a in prep.BUILDERS)]
    result = {"status": "PROSPECTIVE_TRAIN_AI_PROXY_COMPLETE_NO_PROMOTION", "provider_calls": len(events),
        "estimated_classification_cost_cny": spent, "primary": primary,
        "common_valid_binary": metric_adapter.population_metrics(common),
        "paired_binary_transitions": {name: sum((t["v5_3"] == t["target_class"], t["v5_3_1"] == t["target_class"]) == state for t in common)
            for name, state in (("both_agree", (True, True)), ("baseline_only", (True, False)), ("candidate_only", (False, True)), ("neither_agrees", (False, False)))},
        "usage": {a: {"calls": sum(e["arm"] == a for e in events),
            "prompt_tokens": sum(e["prompt_tokens"] for e in events if e["arm"] == a),
            "completion_tokens": sum(e["completion_tokens"] for e in events if e["arm"] == a),
            "latency_p50_ms": shared._percentile([e["latency_ms"] for e in events if e["arm"] == a], .5)} for a in prep.BUILDERS},
        "contract_sha256": canonical(contract), "targets_sha256": contract["targets_sha256"],
        "journal_sha256": {a: shared._sha256(ROOT / a / "attempts.jsonl") for a in prep.BUILDERS},
        "class_support_at_least_five_each": all(primary["target_counts"][c] >= 5 for c in ("SUFFICIENT", "INSUFFICIENT")),
        "formal_pass_gate": None, "dev_opened": False, "promotion_allowed": False,
        "provenance": read_plan()["provenance"]}
    if RESULT.exists() or (ROOT / "summary.json").exists():
        raise RuntimeError("refusing to overwrite evaluation")
    paired.write_json(ROOT / "summary.json", {"aggregate": result, "private_transitions": transitions})
    prep.prior.selection.prior._write_json(RESULT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare-annotation", "annotate", "export-draft", "freeze-review", "prepare-classification", "classify", "evaluate"))
    parser.add_argument("--confirm-authorization", action="store_true", help="Records prior human approval; not permission itself")
    parser.add_argument("--env-file")
    parser.add_argument("--review-file")
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()
    try:
        if args.mode.startswith("prepare-"):
            value = prepare_contract(args.mode.removeprefix("prepare-"), authorization_confirmed=args.confirm_authorization, env_file=args.env_file)
            print(json.dumps({"stage": value["stage"], "contract_sha256": canonical(value)}))
        elif args.mode in ("annotate", "classify"):
            if not args.env_file:
                raise RuntimeError("explicit local credential file required")
            stage = "annotation" if args.mode == "annotate" else "classification"
            contract, _ = load_probe(stage)
            with bounded_client(stage, contract, args.env_file) as client:
                run = run_annotation if stage == "annotation" else run_classification
                value = run(client=client, max_new_calls=args.max_new_calls if args.max_new_calls is not None else (50 if stage == "annotation" else 100))
            print(json.dumps(value if isinstance(value, dict) else {"terminal_outcomes": len(value)}))
        elif args.mode == "export-draft":
            print(json.dumps({"exported": export_draft()}))
        elif args.mode == "freeze-review":
            if not args.review_file:
                raise RuntimeError("explicit review file required")
            print(json.dumps(freeze_review(args.review_file)))
        else:
            print(json.dumps(evaluate()))
    except Exception as error:
        print(f"Stopped ({type(error).__name__}); audit saved state, no automatic retry.")
        raise SystemExit(1) from None
