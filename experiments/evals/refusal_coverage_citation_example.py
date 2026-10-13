"""Offline v5.2.1: corrected audit examples, unchanged v5.2 validator/payload."""

import json

from experiments.evals import refusal_requirement_coverage as prior
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput

# ponytail: retain the frozen policy; replace only its three empty audit placeholders.
_EMPTY_AUDIT = '"evidence": [], "reason": "audit'
_CITED_AUDIT = '"evidence": [{"source_id": "Source N", "segment_ids": [1, 2]}], "reason": "audit'
if prior.CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV.count(_EMPTY_AUDIT) != 3:
    raise RuntimeError("frozen v5.2 audit template changed")

_SUPPORTED_AUDIT = {
    "status": "SUPPORTED", "evidence": [{"source_id": "Source 1", "segment_ids": [1]}],
    "reason": "The displayed policy explicitly covers Atlas release 14.",
}
_CONFLICTING_AUDIT = {
    "status": "CONFLICTING", "evidence": [
        {"source_id": "Source 1", "segment_ids": [1]},
        {"source_id": "Source 2", "segment_ids": [1]},
    ], "reason": "Equally applicable current policies disagree; neither takes precedence.",
}

CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV = prior.CLASSIFIER_SYSTEM_PROMPT_V5_2_DEV.replace(
    _EMPTY_AUDIT, _CITED_AUDIT
) + """

The schema above is a template: select one status and use only actual input IDs.
Every SUPPORTED or CONFLICTING coverage check must contain its own evidence
references, even when the same lines were cited in requirements. Reuse a reference
only when it supports that particular audit; never copy unrelated references.
For MISSING/CONDITIONAL, evidence may be empty when support is absent.
For version_and_time=NOT_REQUIRED, evidence MUST be empty. Never invent citations.

Fictional illustration ONLY, not evidence for the actual input: Source 1, segment 1
says 'Atlas release 14 current cache timeout is 30 seconds.' If asked for the
release 14 timeout, a supported applicability audit is:
""" + json.dumps(_SUPPORTED_AUDIT) + """
If Source 2, segment 1 instead says 'Atlas release 14 current cache timeout is
90 seconds.', with equal authority and no precedence, a conflicting outcome audit is:
""" + json.dumps(_CONFLICTING_AUDIT)

validate_requirement_coverage_response = prior.validate_requirement_coverage_response


def build_classifier_messages_v5_2_1_dev(classifier_input: ClassifierInput) -> list[dict[str, str]]:
    messages = prior.build_classifier_messages_v5_2_dev(classifier_input)
    messages[0]["content"] = CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV
    return messages
