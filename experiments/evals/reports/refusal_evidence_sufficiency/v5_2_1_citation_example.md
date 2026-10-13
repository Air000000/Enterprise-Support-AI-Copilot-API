# Offline v5.2.1 audit citation example fix

Status: **prompt candidate only; zero provider calls and no production change**.

The [frozen controlled run](v5_2_control_result.md) had two v5.2 format failures:
SUPPORTED/CONFLICTING coverage audits omitted their references. Empty audit arrays
in the old schema example are a plausible contributor, not a proven cause.

`experiments/evals/refusal_coverage_citation_example.py` replaces those three empty
schema placeholders with citation-shaped placeholders and supplies two concrete,
fictional supported/conflicting audit illustrations. Instructions require actual
input references relevant to each audit; NOT_REQUIRED still requires an empty
array and is restricted to version/time. No new response fields or automatic
reference copying/repair are introduced.

The original v5.2 policy text, full numbered user payload and validator are reused.
Frozen v5.1/v5.2 modules, contracts, journals, failures and scores are unchanged.
Tests check example validity, identical user payload and unchanged rejection of
missing/invalid citations. These are mechanical regressions, not model accuracy
or evidence that format reliability, semantic coverage or refusal quality improved.
False SUPPORTED/NOT_REQUIRED judgments remain undetectable by this locator gate.

The 14 now-observed controlled fixtures are development regression cases, not an
unseen benchmark. Any future model comparison needs a separate registered run,
new checkpoint and scoped authorization; the completed 28-call allowance is not
reused. Natural TRAIN labels (including only four insufficient cases), DEV and
serving remain untouched. No new runner or paid entrypoint is added here.
