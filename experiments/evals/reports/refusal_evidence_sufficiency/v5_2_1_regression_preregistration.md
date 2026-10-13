# v5.2 / v5.2.1 development regression preflight

Status: **prepared, not run or authorized; no paid entrypoint**.

Compare original v5.2 against the prompt-only v5.2.1 citation-example correction
on the same 14 fictional cases (seven correlated pairs). Use a fresh contemporaneous
v5.2 arm, not historical predictions as its replacement. No fixture changes, new
labels, natural TRAIN pooling, DEV access or serving integration.

Freeze both prompts, requests, source dependencies, case order and prior result
hashes in `v5_2_1_regression_preparation.json`. Both arms use Singapore
`qwen3.5-plus-2026-04-20`, temperature 0, thinking off, JSON output, 2048-token
output cap. Alternate arm order per case. Up to 28 calls total, no retries;
proposed fixed-FX accounting budget CNY3. Pricing checked 2026-10-13 against
[Alibaba Cloud pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing):
International input/output $0.4/$2.4 per million tokens up to 256K input.
UTF-8 byte/framing planning bounds and 7.5 CNY/USD are not certified token limits,
spot exchange rates or an invoice guarantee. Exact planning estimate is in JSON.

Future execution requires separate scoped authorization and an execution contract
that additionally binds the runner. A distinct ignored checkpoint must persist
pending markers, raw responses/usage before parsing and terminal journal entries;
successful calls never repeat, schema failures never retry, unknown charge blocks
resume. A local exclusive lock prevents concurrent dispatch. Do not reuse or edit
PR70 journals/contracts/results. Targets stay outside requests and the paid loop.

Report all-14 failures per arm, common-valid target agreement, paired transitions
(null failures remain explicit), valid false admissions/refusals, tokens/cost and
latency. Fewer failures versus concurrent v5.2 with no new valid semantic errors
is only a format-fix development signal; equal failure counts are inconclusive.
One run cannot prove causality or establish real-data refusal accuracy. These are
already-observed regression fixtures, not independent/unseen validation or expert
gold. Natural insufficient-class count remains four. No production promotion.
