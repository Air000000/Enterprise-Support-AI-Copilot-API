"""Offline v5.3 policy candidate; requirement relevance remains a model judgment."""

from experiments.evals import refusal_coverage_citation_example as prior
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput

CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV = prior.CLASSIFIER_SYSTEM_PROMPT_V5_2_1_DEV + """

Question-scoped requirement and applicability policy:
Derive material requirements from the requested claims, including every requested
outcome and genuinely necessary dependency, condition, version and time constraint.
If removing an item still permits a complete, properly qualified answer to the
question, do not invent that item as an extra requirement. Sources supply evidence,
not additional user demands. A scoped method/example is not a universal guarantee.

Use all context, but do not require every retrieved platform, optional alternative
or unrelated symptom to apply. A complete, applicable evidence path need not reconcile
irrelevant alternatives. Explain scope exclusions in existing reasons; do not ignore
real conflicting evidence for the same requested claim in the same applicable scope.
Different minimum fix levels or different symptoms are not automatically conflicts;
nor may apparent differences be discarded without a visible scope/compatibility basis.

Distinguish possible causes/diagnostic checks from an established incident cause or
remedy. Shared product, update, vulnerability or error names alone are not a causal
bridge. For a specific incident, explain in the applicability reason which relevant
mechanism and prerequisites match facts in the question/context, and which are unknown.
A causal attribution in the question is a claim to check, not independent confirmation
of its mechanism.
If a necessary bridge is absent, mark MISSING or CONDITIONAL, not SUPPORTED from topic
overlap. Do not invent missing patch, configuration, classpath or environment facts.
General explanations/checks may be supported when that is what the question requests.
An incident with a visibly matching mechanism and established relevant prerequisites
may also be supported: neither exact exception wording nor exhaustive elimination of
every hypothetical cause is automatically a requirement.

Keep the same response schema, mandatory audits and citation rules. Do not erase a
real gap, infer NOT_REQUIRED to discard explicit constraints, or fabricate references.
"""

validate_requirement_coverage_response = prior.validate_requirement_coverage_response


def build_classifier_messages_v5_3_dev(classifier_input: ClassifierInput) -> list[dict[str, str]]:
    # ponytail: policy-only revision; retain full context, line numbering and strict gate.
    messages = prior.build_classifier_messages_v5_2_1_dev(classifier_input)
    messages[0]["content"] = CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV
    return messages
