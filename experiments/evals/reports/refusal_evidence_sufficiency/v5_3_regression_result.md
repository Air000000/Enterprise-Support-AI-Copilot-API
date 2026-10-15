# Observed TRAIN v5.2.1 / v5.3 development regression

Decision: **small proxy-agreement gain, no independent acceptance or promotion**.
v5.3 corrects two proxy disagreements and introduces one, net one more agreement.
The same insufficient case is still admitted. The incident-applicability hypothesis
has no demonstrated benefit here; broader semantic reliability is not established.

## Execution and evidence boundary

User explicitly authorized execution under merged PR77. Follow the frozen
[protocol](v5_3_regression_protocol.md): all30 original TRAIN questions/full ordered
Top14; unchanged 21 SUFFICIENT, 4 INSUFFICIENT and 5 AMBIGUOUS AI-proxy labels.
These are Qwen drafts plus current-assistant evidence review, not independent
expert gold. Prior predictions and targeted evidence informed v5.3; this is observed
development regression, not unseen validation or representative enterprise accuracy.
The original minimum-five-per-class rule remains unmet and unchanged.

Exactly60 calls, 30 per fresh concurrent arm; Singapore
`qwen3.5-plus-2026-04-20`, temperature0, thinking off, JSON, max2048 output tokens,
alternating arm order. Two initial calls were followed by the remaining58 from the
same checkpoint, with no repeated calls. No retries, recharge, provider/usage/schema
failures, new retrieval, annotations, relabeling, DEV access or serving changes.
Requests omit labels, review reasons, IDs, retrieval metadata and secrets. Targets
were opened for scoring only after all60 terminal outcomes and a stopped runner.

## Counts before rates

| Outcome relative to frozen AI proxy | v5.2.1 | v5.3 |
| --- | ---: | ---: |
| All-case failures | 0/30 | 0/30 |
| Binary target agreement, failures included | 22/25 (88%) | 23/25 (92%) |
| Sufficient targets admitted | 19/21 | 20/21 |
| Sufficient targets refused | 2/21 | 1/21 |
| Insufficient targets admitted | 1/4 | 1/4 |
| Insufficient targets refused | 3/4 | 3/4 |
| All-case admissions | 23/30 | 24/30 |
| Admitted-binary proxy agreement | 19/20 | 20/21 |
| Ambiguous admissions / refusals | 3/2 | 3/2 |

All25 binary pairs are common-valid: both agree21, candidate only2, baseline only1,
neither1. Two sufficient-case refusals become admissions and one new sufficient-case
refusal appears; every insufficient and ambiguous decision is unchanged. Five
ambiguous cases never count as binary successes. Null-failure agreement bounds
coincide at88%/92% because there are no failures. These are proxy agreements, not
answer-accuracy measurements; no answers were generated.

Dropping either known near-duplicate member separately gives baseline21/24 versus
candidate22/24, still net one. Both members agree in both arms. This is limited
correlation sensitivity, not a complete family partition or IID significance.
One outcome changes the insufficient-class rate by1/4.

The contemporaneous baseline scores22/25, versus21/25 historically: one old refusal
does not recur despite the same frozen inputs/prompt/temperature0. The historical
score is not the comparison arm. Do not claim a two-case gain by comparing92% to
the old84%, or treat temperature0 as a determinism guarantee.

## Targeted diagnosis, no post-result label repair

One corrected decision stops refusing an exact-product search instruction because
of other search-engine documents. However, its reasoning still extrapolates from
another product using an unestablished shared-technology assumption. Agreement with
the proxy does not validate that reasoning or prove every cited source applies.
The other correction retains an explicit locale/OS-build compatibility caveat
instead of treating that documented caveat as insufficient evidence to answer.

The new disagreement concerns a bulletin-location request with a stated desire for
details. The candidate adds a requirement to reproduce technical vulnerability
details, whereas the frozen proxy interprets the visible exact bulletin/CVE pointer
as sufficient for the location request. This is scope interpretation relative to an
AI proxy, not independent expert proof that the model or label is definitively wrong.
No label, requirement, citation or outcome is edited after seeing predictions.

The persistent insufficient admission still connects a classloading explanation
from another application and package-isolation guidance to the actual reported
incident without establishing patch/configuration/classpath applicability. Valid
source references and an explicit applicability policy did not prevent a falsely
SUPPORTED judgment relative to the proxy. Do not claim the causal-bridge issue fixed.

## Cost, latency and reproducibility

Total estimated fixed-FX accounting cost **CNY1.917294**, below CNY8; not an invoice.
Use the frozen official International pricing/FX in the protocol, not an observed
discount or live exchange rate.

| Usage | v5.2.1 | v5.3 |
| --- | ---: | ---: |
| Input/output tokens | 178,030 / 21,794 | 189,550 / 23,459 |
| Estimated cost | CNY0.926382 | CNY0.990912 |
| Median provider latency | 11.81 s | 12.28 s |

Candidate cost is about1.070x and median latency1.040x in this run. This small
observed gain plus unchanged false admissions does not justify replacing the
experimental baseline or online policy. Do not automatically add another prompt,
retry these cases, broaden the dataset or open DEV under this exhausted authorization.

The thin runner reuses the existing isolated durable execution engine and population
metrics. Contract/source/protocol/request/input/target fingerprints are pinned;
raw response/usage is durable before parsing. Private raw journals and per-case
targets/transitions stay ignored; public output is aggregate fingerprints/counts.
Completed replay adds zero calls; no pending, lock or staged response remains.
Historical experiment and annotation hashes remain unchanged.
