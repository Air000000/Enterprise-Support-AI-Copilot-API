"""Offline v5.3.1 candidate; prompt clarification, NOT verified semantic repair."""

from experiments.evals import refusal_question_scoped_coverage as prior
from experiments.evals.refusal_evidence_sufficiency import ClassifierInput

# ponytail: clarify the existing audit; retain payload, statuses and strict locator gate.
CLASSIFIER_SYSTEM_PROMPT_V5_3_1_DEV = prior.CLASSIFIER_SYSTEM_PROMPT_V5_3_DEV + """

Version/time audit clarification:
version_and_time reports whether the question's material version/time constraints
are satisfied by the evidence, NOT whether a source states its own scope clearly.
A fixed, explicitly requested version is a material constraint even without words
such as current or latest. Audit it; do not mark it NOT_REQUIRED for being static.

SUPPORTED requires evidence covering the requested version/time, directly or via
a visible compatibility/applicability bridge. A source that explicitly excludes
the requested version/time establishes a boundary, NOT support for that request:
mark MISSING if coverage is absent. Unresolved material conditions are CONDITIONAL;
incompatible equally applicable evidence is CONFLICTING. Do not invent conflicts
between irrelevant versions or demand exact version-string repetition when visible
compatibility evidence covers the requested scope.

Before returning, compare this audit with the version/time claims in requirements
and applicability reasons. Do not label version/time SUPPORTED while explaining
that the requested version/time is not covered. An unrelated feature/procedure gap
does not by itself make the version/time constraint unsupported. NOT_REQUIRED is
only for a question imposing no material version/time requirement.
"""

validate_requirement_coverage_response = prior.validate_requirement_coverage_response


def build_classifier_messages_v5_3_1_dev(classifier_input: ClassifierInput) -> list[dict[str, str]]:
    messages = prior.build_classifier_messages_v5_3_dev(classifier_input)
    messages[0]["content"] = CLASSIFIER_SYSTEM_PROMPT_V5_3_1_DEV
    return messages
