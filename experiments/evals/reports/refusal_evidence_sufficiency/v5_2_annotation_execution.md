# v5.2 blind annotation execution

This is annotation only. No v5.1/v5.2 classifier execution, DEV access, runtime
change or deployment approval is included. The 30-case preparation and original
annotation contract remain immutable. User authorization covers <=32 calls and
an accounting cap of CNY8; recharge and additional attempts are not authorized.

Annotator: Singapore `qwen3.6-plus-2026-04-02`, temperature 0, thinking disabled,
JSON response, 4096 output-token cap. This is a different snapshot but the same
Qwen family as the prospective classifier: not an independent judge or human gold.
Questions and full ordered Top14 are sent without IDs, prior labels, predictions,
retrieval scores or document metadata. Body text is untrusted data.

## Checkpoint and recovery

Ignored local directory: `data/refusal_v5_2_train/blind_annotation/`.
The runner takes an exclusive local lock; a stale lock requires confirming that
the original process has stopped before removal. It writes a pending marker before
the request, then fsyncs the raw response/usage **before** parsing, and fsyncs each
attempt. Ordinary interruption resumes without repeating successes; a saved raw
response can be replayed without another call. A request with no known response or
unknown usage blocks paid resume until provider/account reconciliation. Recharging
alone does not resolve unknown-charge evidence. No automatic SDK retries or recharge.

The original v1 ledger includes a paid retry after a reference-format rejection.
This remains counted. Amendment v1.1 permits separately audited citation recoveries:
an additional local JSONL record pins the original response hash, retains its class
and supporting-source IDs, and supplies verified original line references. It never
rewrites the raw response/attempt. Provider failures and unknown usage cannot be
recovered this way. Later semantic label review is separate and happens before any
classifier prediction; both the draft and final reviewed targets remain preserved.
Original contract/identity remain, with a separate amended contract/identity.

```powershell
python -m experiments.evals.refusal_v5_2_annotation --run-paid --env-file 'path\to\.env'
```

This requires the existing Singapore-scoped credentials. The Beijing embedding
key is not reused. After a known charged schema failure, `--resume-after-audit`
permits only a budgeted retry; it does not bypass unknown-charge, fingerprint,
spend or call limits. Citation recoveries save repeated model calls instead.

## Evidence review and freeze

Each question, draft rationale and cited original evidence is reviewed by the
current Codex assistant, with targeted full-context checks. This is not full-source
expert adjudication, and this reviewer is not independent. Review catches scope
inflation, mismatched bulletin passages, conditional compatibility and ambiguous
title/body requirements. Do not choose labels to make either classifier win.

Local `reviewed_annotations.jsonl` retains per-case evidence and disagreements;
local `blind_draft_annotations.jsonl` preserves model targets. Freeze local
`targets.jsonl` and public counts/hashes only after all 30 cases are reviewed.
AMBIGUOUS remains separate, never silently forced into a binary class. Category
coverage is descriptive and overlapping. Fewer than five examples in either
binary class makes the earlier paired decision rule inconclusive; do not resample.

Public artifacts exclude questions, source bodies, target rows and rationales.
Cost uses official international rates ($0.5/$3 per million input/output tokens),
fixed 7.5 CNY/USD accounting and no discounts, checked 2026-10-13
([pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)).
Byte/framing reservation is a planning proxy, not a tokenizer or invoice guarantee.

## Frozen annotation outcome

All 30 cases annotated and evidence-reviewed before classifier predictions.
31 calls, including one paid format retry; estimated CNY0.7080075, not an invoice.
Ten other format-rejected responses were recovered from their original evidence
without a paid repeat. The original raw ledger retains all eleven format failures.
No provider, quota or unknown-usage failures occurred. Final AI-proxy counts:
21 SUFFICIENT, 4 INSUFFICIENT, 5 AMBIGUOUS; six class changes during pre-prediction
review. Original draft counts: 22 SUFFICIENT, 8 INSUFFICIENT.

The insufficient class falls below the preregistered five-case minimum. This
sample cannot satisfy the prospective paired decision rule even if descriptive
agreement improves. No classifier scores exist and no production gain is claimed.
Keep all cases; do not repair the shortfall by relabeling or replacing samples.
Category coverage: steps 10, applicability 23, version/time 13, conditions 14,
explicit conflict 0 (overlapping review tags, not a validated category benchmark).
Selected TRAIN_Q383 and TRAIN_Q417 are near-duplicate scenarios; the family proxy
did not guarantee semantic independence. Report this correlation, not 30 independent
trials. Any additional sampling or paid classification needs a separately scoped
contract and authorization. Public freeze records hashes/counts only.
