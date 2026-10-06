"""New TRAIN-only v4/v5.1 diagnostic; preflight is offline, paid use needs authorization."""

from pathlib import Path

from experiments.evals import refusal_citation_selection as citation
from experiments.evals import refusal_v5_paired_runner as paired

CONTRACT_PATH = paired.REPORT_DIR / "v5_1_train_run_contract.json"
CONTRACT_SHA256 = "76210c6cf96c9bd57986dccc6847170ec178be25bc4512fece2e2bda1ad6d8fb"
OUTPUT_DIR = Path("data/refusal_v5_train/paired_run_v5_1")
BUILDERS = {"v4": paired.build_classifier_messages_v4_dev,
            "v5_1": citation.build_classifier_messages_v5_1_dev}


def load_probe(inputs_path=None):
    return paired.load_probe(
        inputs_path, contract_path=CONTRACT_PATH, contract_sha256=CONTRACT_SHA256,
        builders=BUILDERS,
        prompts={"v4": paired.CLASSIFIER_SYSTEM_PROMPT_V4_DEV,
                 "v5_1": citation.CLASSIFIER_SYSTEM_PROMPT_V5_1_DEV},
        validators={"v5_validator_normalized_source_sha256": paired.structured,
                    "citation_selector_normalized_source_sha256": citation})


def parse_result(arm, raw, row, finish_reason):
    if arm == "v4":
        return paired.parse_result(arm, raw, row, finish_reason)
    if arm != "v5_1" or finish_reason != "stop":
        raise ValueError("invalid arm or response did not finish normally")
    return citation.validate_citation_selection_response(
        raw, paired.shared._classifier_input_from_row(row))


def options():
    # ponytail: bind the second real protocol to the existing client/journal, no new framework.
    return {"probe_loader": load_probe, "builders": BUILDERS,
            "result_parser": parse_result, "contract_sha256": CONTRACT_SHA256}


def run_paid(*, output_dir=OUTPUT_DIR, **kwargs):
    return paired.run_paid(output_dir=output_dir, **options(), **kwargs)


def evaluate(*, output_dir=OUTPUT_DIR, **kwargs):
    return paired.evaluate(output_dir=output_dir, **options(), **kwargs)


if __name__ == "__main__":
    paired.main(output_dir=OUTPUT_DIR, **options())
