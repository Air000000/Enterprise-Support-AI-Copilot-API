# v5.3 / v5.3.1 prospective TRAIN preparation

Preparation only: zero model calls, no labels, predictions, paid runner or execution
contract. User approval to prepare is NOT permission to send data or spend money.
Keep the old v5.2.1/v5.3 protocol, manifest and authorization gates unchanged.

## Frozen scope

Reuse the exact 50 question IDs, order, questions and full ordered Top14 bodies in
`prospective_train_preparation.json`, verified through its existing read-only
loader. No new selection, retrieval, reranking, truncation, balancing or replacement.
Do not open DEV, gold answers or qrels. Check the recorded local annotation, target,
checkpoint and contract paths before assuming the batch has no predictions/labels;
existing state requires an audit, not deletion or a new paid run. Absence in this
checkout does not establish absence on other machines/checkouts.

Freeze the unchanged `v5_3` and `v5_3_1` builders and shared strict validator before
annotation. Only the system prompt differs; user messages and source order are
identical. Historical old30, fictional controls and old model predictions are not
comparison arms or pooled scores. This replaces neither old experiment nor serving.

## Two stages, only after fresh authorization

1. At most 50 Singapore `qwen3.6-plus-2026-04-02` blind annotation calls, temperature
   0, thinking off, JSON object, max4096 output tokens, no paid retries. Reuse the
   existing annotation prompt. Current assistant reviews all drafts and evidence,
   records review depth/changes and preserves raw responses. Do not ask the user to
   perform specialist labeling. Do not auto-map parser failures to AMBIGUOUS.
   Recover only provable citation-format issues from saved responses, preserving
   original class/source IDs and an audit; no model retry or raw overwrite.
2. Freeze all50 reviewed annotations, target hashes and actual class/ambiguity
   counts before predictions. If complete defensible labels cannot be frozen, stop.
   Then at most 100 fresh Singapore `qwen3.5-plus-2026-04-20` classification calls,
   50 per arm, max2048 output tokens, temperature0, thinking off, JSON object.
   Fixed question-ID order: even index v5.3 then v5.3.1, odd index reversed. Send
   question/full sources only, never labels, reviewer reasons or retrieval metadata.

Use a new `data/refusal_v5_3_1_prospective_train` checkpoint namespace and existing
durable machinery, not a new client/framework. Future execution must bind this
preparation, inputs, sources, prompts, complete request parameters, dependencies,
endpoint and frozen targets. Exclusive lock; pending before POST; durable raw/usage
before parsing; saved-response recovery without repeat calls. Unknown charge,
network or provider error stops execution; do not auto-resume uncertain sends.
Classifier schema/finish failures remain terminal null outcomes, not refusals or
successes. No post-result output repair, relabeling, prompt tuning or paid reruns.
No answers, judge, embedding, rerank, tickets, DEV, deployment or default switches.

## Metrics and interpretation

Report all50 per-arm outcomes/failures and actual target counts first. All binary
targets, including failed predictions, remain the primary agreement denominator;
report confusion and per-class recalls. Separately report common-valid binary
denominators/classes and paired fixes/regressions. Ambiguous targets are descriptive,
not correct decisions. Report all50 admission coverage, admitted-binary agreement,
ambiguous admissions, and possible agreement bounds for null failures. Undefined
denominators are null. Include every charged outcome in tokens/accounting cost and
latency. Do not compare only valid successes or replace the concurrent baseline
with a historical score. Inspect version-audit contradictions descriptively; there
are no frozen per-facet targets, so do not invent a version-facet accuracy metric.

This is AI-proxy, fresh-to-recorded-refusal-selection TRAIN development evidence,
NOT independent expert gold, untouched data or enterprise answer accuracy. All TRAIN
has prior retrieval/analysis exposure; family exclusion is only a visible proxy and
same-family Qwen annotation plus assistant review remains coupled. Report whether
each binary class has at least5 examples; never resample/relabel to achieve it.
There is no formal PASS/promotion gate, significance claim or automatic adoption.

## Pricing and next gate

[Official Singapore / International pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
rechecked2026-10-20: <=256K input band, USD per million input/output tokens:
annotation0.5/3, classification0.4/2.4. Accounting FX7.5 is fixed for comparison,
not a live FX quote. Ignore discounts/free quota. UTF-8 bytes+512 framing and full
output limits are planning proxies, not certified token counts or invoice caps.
The generated manifest binds all request hashes and per-case planning bounds.

Proposed separate accounting limits: annotation CNY10, classification CNY15,
total CNY25, at most150 calls; no stage transfer, recharge, retries or automatic
continuation. Actual charges can differ or be unknown. Model availability, region,
credentials and execution accounting must be checked before requesting a new
explicit data/model/call/budget authorization and adding a thin paid adapter.
The preparation CLI never executes either stage. Merge/continue is not paid consent.
