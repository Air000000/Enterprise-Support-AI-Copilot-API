# G1 Document-Local Evidence Sufficiency — Final Decision

## Status

**Official preregistered decision: NO_GO.**

G1 materially improved aggregate evidence sufficiency over the E1 reference on the frozen 30-case experiment, but it triggered one preregistered catastrophic regression (`E1=COMPLETE` and `G1=INSUFFICIENT`). The gate is therefore not relaxed post hoc, and G1 does **not** replace E1 as the current reference context policy.

This is not evidence that the document-local design is broadly worse than E1. It is a bounded deployment/evaluation decision: aggregate gains were strong, while one predeclared safety gate failed.

## Experiment scope

The authoritative preregistration is:

- `evidence_sufficiency_30case_preregistration_v1_1.json`
- experiment id: `techqa_g1_blinded_evidence_sufficiency_30case_v1`
- sample size: 30 answerable `TRAIN_*` cases
- deterministic sample seed: `techqa-g1-evidence-sufficiency-v1`
- historical G0 12-case pilot excluded
- no resampling or outcome-based case replacement
- conditional Stage2 scope: the formal relevant TechQA document had to appear among the first five unique documents in the historical E0 ranking

The experiment therefore evaluates **context evidence sufficiency after a bounded candidate-document admission condition**. It does not establish an all-TechQA retrieval uplift or a production accuracy improvement.

## Compared methods

### E1 reference

```text
historical E0 Dense Top100
        ↓
one shared qwen3-rerank
        ↓
Top3 rerank anchors
+ historical Dense rank1 rescue
        ↓
for each unique anchor document:
anchor + up to 3 forward siblings
        ↓
deduplicate
        ↓
max 16 context chunks
```

### G1 candidate

```text
historical E0 ranking
        ↓
first 5 unique candidate documents
        ↓
full frozen document-local chunk expansion
        ↓
one merged qwen3-rerank
        ↓
Top16 evidence chunks
        ↓
max 16 context chunks
```

Frozen reranker contract:

- model: `qwen3-rerank`
- instruction: `Rank the candidate passages by relevance to resolving the technical support query.`
- no query rewrite
- no method-specific prompt
- no per-document reranking
- no second-pass rerank
- no embedding, generation, or judge calls in the paid 30-case retrieval execution

## Evidence review contract

Gold-answer material claims were frozen before any blinded context review. The final claim freeze contained:

- 30 cases
- 65 material claims
- 195 claim-context judgments across G0 / E1 / G1

Claim support labels:

- `EXPLICIT_SUPPORT`
- `COMPOSABLE_SUPPORT`
- `INFERENCE_ONLY`
- `ABSENT`

Only `EXPLICIT_SUPPORT` and `COMPOSABLE_SUPPORT` count as covered.

Per-context sufficiency:

- `COMPLETE`: every material claim covered
- `PARTIAL`: at least one but not all claims covered
- `INSUFFICIENT`: zero material claims covered
- `UNRESOLVED`: evidence support cannot be judged reliably

The primary aggregate coverage metric is the macro mean of per-case claim coverage, not a micro-weighted total over claims.

Method identity remained blinded until all claim-support and sufficiency judgments were frozen.

## Formal result

| Metric | E1 | G1 | G1 - E1 |
| --- | ---: | ---: | ---: |
| COMPLETE | 24 / 30 | **28 / 30** | +4 cases |
| PARTIAL | 1 / 30 | 1 / 30 | 0 |
| INSUFFICIENT | 5 / 30 | **1 / 30** | -4 cases |
| Macro claim coverage | 0.825000 | **0.955556** | **+0.130556** |

Pairwise preregistered ordinal movement (`COMPLETE=2`, `PARTIAL=1`, `INSUFFICIENT=0`):

- G1 wins: **6**
- ties: **22**
- losses: **2**

Win cases:

- `TRAIN_Q458`
- `TRAIN_Q118`
- `TRAIN_Q227`
- `TRAIN_Q415`
- `TRAIN_Q589`
- `TRAIN_Q091`

Loss cases:

- `TRAIN_Q346`
- `TRAIN_Q492`

The preregistered catastrophic regression definition was:

```text
E1 = COMPLETE
and
G1 = INSUFFICIENT
```

Observed catastrophic regressions:

- count: **1**
- case: `TRAIN_Q346`

Therefore the formal gate is:

```text
G1_complete_count >= E1_complete_count       PASS
G1_insufficient_count <= E1_insufficient     PASS
G1_wins >= G1_losses                          PASS
catastrophic_regression_count == 0            FAIL
-------------------------------------------------
OFFICIAL DECISION                             NO_GO
```

## Claim-level transition audit

A post-unblinding forensic pass examined all 65 frozen claims without changing the rubric or judgments.

| Transition | Count |
| --- | ---: |
| E1 covered -> G1 covered | 57 |
| E1 covered -> G1 not covered | 2 |
| E1 not covered -> G1 covered | 6 |
| E1 not covered -> G1 not covered | 0 |

Net claim movement:

- lost claims: 2
- gained claims: 6
- net gain: **+4**

Gain attribution:

- merged-rerank gain: 3 claims
- document-local expansion gain: 2 claims
- multi-chunk evidence gain: 1 claim

The gains are therefore mixed rather than attributable to one single mechanism.

## Regression forensics

### `TRAIN_Q346`: candidate-document admission miss

Frozen claim:

> This defect is resolved in IBM Rational DOORS Version 9.4.0.1.

E1 covered the claim explicitly. The decisive evidence came from:

- document: `swg1PM50525.txt`
- chunk: `swg1PM50525.txt_chunk_0`

The decisive chunk was deep in the historical E0 chunk ranking (position 72, one-based), but E1's global reranker promoted it into the Top3 anchor set. G1 instead admitted only the first five unique documents from the historical E0 order before its merged document-local rerank. `swg1PM50525.txt` was not in that first-five document set, so the decisive chunk never entered G1's 27-chunk candidate pool.

Root cause:

```text
CANDIDATE_DOCUMENT_MISS
```

The important diagnostic lesson is narrower than "Top5 is too small":

> a weak first-stage document-order gate can permanently remove a tail chunk that a stronger global reranker would have rescued.

The formal relevant-document eligibility rule and the generation evidence needed for a material claim are not guaranteed to be the same thing. This case therefore also reinforces the existing project distinction:

```text
document-level qrel hit != answer-bearing evidence hit
```

### `TRAIN_Q492`: continuity miss

E1 was `COMPLETE` (3/3 claims); G1 was `PARTIAL` (2/3).

For the missing claim, E1 retained evidence from:

- `swg21972012.txt_chunk_1`
- forward sibling `swg21972012.txt_chunk_2`

G1 admitted the document and included both chunks in its 34-chunk document-local candidate pool, but the final Top16 retained chunk 1 while omitting the decisive sibling chunk 2. The exact merged-rerank rank of chunk 2 was not persisted; the preserved evidence establishes only that it was outside the final selected Top16.

Root cause:

```text
CONTINUITY_MISS
```

## Are these systemic G1 defects?

No repeated mechanism was found in the remaining frozen claim transitions:

- candidate-document admission miss repeated elsewhere: **no**
- continuity miss repeated elsewhere: **no**
- candidate-chunk construction miss: 0
- assembly miss after Top16 selection: 0
- insufficient provenance: 0

The two regressions therefore have different failure boundaries and were classified as **isolated regressions**, not evidence of one shared systemic reranker defect.

This matters for the next decision: changing `first5 -> first10`, forcing a sibling rule, or adding case-specific rescue logic after inspecting Q346/Q492 would be high-risk TRAIN overfitting.

## Final engineering decision

**Keep E1 as the current reference. Keep G1 as an evaluated experimental candidate. Do not patch G1 on the already-inspected 30-case TRAIN set.**

Supported conclusions:

1. G1 materially improves aggregate evidence sufficiency on the frozen conditional 30-case experiment.
2. G1 increases COMPLETE contexts from 24/30 to 28/30 and macro claim coverage from 0.825 to 0.9556.
3. The preregistered replacement gate still fails because one E1-complete case becomes G1-insufficient.
4. The two observed regressions have distinct, non-repeating mechanisms in this sample.
5. The current evidence does not justify a Q346/Q492-targeted patch.

Not supported:

- claiming G1 replaced E1;
- claiming a production generation-quality uplift;
- claiming a full-TechQA retrieval improvement from this conditional Stage2 experiment;
- relaxing the catastrophic-regression gate after seeing results;
- tuning document count or continuity rules on these already-inspected cases and reporting that as fresh validation.

## Next hypothesis boundary

A future experiment may test a cleaner candidate-admission architecture on a **fresh preregistered held-out sample**, for example:

```text
wide chunk recall
        ↓
strong global rerank
        ↓
rerank-informed document admission
        ↓
document-local expansion
        ↓
merged evidence selection
```

A continuity safeguard should be treated as a separate causal variable unless independent evidence shows the two mechanisms should be coupled.

The previous 30 G1 cases are now diagnostic/development evidence for any such next design and must not be presented as fresh validation of a patch derived from Q346 or Q492.

## Artifact lineage

Repository-frozen preregistration:

- `experiments/evals/reports/g1_document_local/evidence_sufficiency_30case_preregistration_v1_1.json`
- SHA256 recorded before paid execution: `05442933333b6b32c7988b8012feb6df33a97afd19494fc145dfa34d5cf1c898`

Key local execution/review artifacts were kept outside Git; their recorded SHA256 digests are preserved here for audit lineage:

- frozen claim decomposition v2: `cffe8033311df4c5482addbd53c48ce7d30c9469ccfe7da190ef086f0865b01f`
- frozen blinded aggregation: `73ec0d79a32dd37667d51c3067778051c0ab4af04fea7e1c08bd50046dc82f7b`
- blinded final manifest: `33f569f91dc9dad53d8346cd4caca21ba5efebf1e0772cf2ef0ea27614c1d157`
- unblinded results: `6084ba65d24c19853403cb54067139e2a0a0176b4bbf885f6e7fa72905e4f2a1`
- unblinded summary: `a005af6f3873ba2269b11e8c13572bd6e3ed1ece389c5c490cb7e772a3dba9a3`
- 65-claim transition table: `555b21c0ee081431fcc4aaa6f9a179ed7de96f0f1ffed5cfc0d599033c05ea07`
- claim-transition forensics summary: `79f8cd6019430c2c58a2d8a8232435e2757ebf6525aea0cfe60c1529bd7e7f92`

Raw provider packets, reviewer packets, and scratch execution artifacts are intentionally not reconstructed from these digests or committed as if they were repository-native results. This report records the frozen decision and audit lineage; it does not fabricate missing raw artifacts.
