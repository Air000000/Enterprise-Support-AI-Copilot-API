# G2 Rerank-Informed Admission Post-Hoc Forensic Analysis

## 1. Frozen confirmatory result

The frozen Task 8 result is unchanged and remains the confirmatory record:

| Metric | G1 | G2-A |
| --- | ---: | ---: |
| COMPLETE | 19 | 19 |
| PARTIAL | 2 | 1 |
| INSUFFICIENT | 9 | 10 |
| UNRESOLVED | 0 | 0 |
| Macro claim coverage | 0.666667 | 0.650000 |

Pairwise result is **2 wins / 25 ties / 3 losses**, including **2 catastrophic regressions**. The final decision is **NO_GO**. The Task 8 result artifact SHA-256 is `8e6e51ca0bc2200987e473058c77fa58293e62be45e5964f94b79eddfe904145`; the canonical SHA-256 is `94c45247dd7399f882ca1cd8bad1c9b5702d7420183b35fbc547531877a886e9`.

This report does not rewrite that result. It is a post-hoc diagnostic over already-frozen local artifacts and is not preregistered confirmatory evidence. No rerank, embedding, generation, or judge calls were made during this analysis.

## 2. Method and causal boundary

The sealed A/B mapping was used to unblind the frozen outcomes. The five movements were then derived mechanically by comparing the frozen G1 and G2 outcome arrays:

- G1 INSUFFICIENT -> G2 COMPLETE: `TRAIN_Q287`, `TRAIN_Q578`
- G1 COMPLETE -> G2 INSUFFICIENT: `TRAIN_Q090`, `TRAIN_Q500`
- G1 PARTIAL -> G2 INSUFFICIENT: `TRAIN_Q367`

G2 changes only admission order. G1 takes the first five unique documents in Dense Top100; G2 takes the first five unique documents in the shared global rerank over the same Dense Top100. Document-local expansion, the 500-chunk pool cap, merged rerank, and Top16 selection are otherwise frozen.

## 3. Five movement cases

The `Dense relevant positions` and `global relevant positions` columns are 1-based chunk positions. `G1 docs` and `G2 docs` are the exact first-five admitted document IDs reconstructed from the frozen Dense chunk order and paid `shared_global` rerank, respectively. Pool sizes and document IDs are reconstructed from the complete paid `g1_merged` and `g2_merged` permutations. The following invariants all passed: `dense_preflight_equals_E0`, `g1_pool_docs_equals_g1_admitted`, `g2_pool_docs_equals_g2_admitted`, `g1_top16_docs_within_g1_admitted`, and `g2_top16_docs_within_g2_admitted`.

| Case | Gold document | Dense relevant positions | Global relevant positions | G1 docs / pool | G2 docs / pool | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| `TRAIN_Q287` (`U023`) | `swg27047895.txt` | 20, 58 | 1, 8 | no; `swg1IO23927.txt`, `swg1IO25547.txt`, `swg1IO25431.txt`, `swg1IO23565.txt`, `swg1IO23514.txt` / 21 | yes; `swg27047895.txt`, `swg21599801.txt`, `swg21959318.txt`, `swg21903282.txt`, `swg21968904.txt` / 35 | INSUFFICIENT -> COMPLETE |
| `TRAIN_Q578` (`U059`) | `swg21612222.txt` | 14, 67 | 1, 11 | no; `swg21504129.txt`, `swg21260903.txt`, `swg22004447.txt`, `swg22006446.txt`, `swg1IT12043.txt` / 27 | yes; `swg21612222.txt`, `swg22006446.txt`, `swg21260903.txt`, `swg21902484.txt`, `swg1IT09552.txt` / 27 | INSUFFICIENT -> COMPLETE |
| `TRAIN_Q090` (`U006`) | `swg24044840.txt` | 2, 11 | 11, 14 | yes; `swg24041994.txt`, `swg24044840.txt`, `swg21692150.txt`, `swg21974112.txt`, `swg21687624.txt` / 102 | no; `swg21692150.txt`, `swg21687624.txt`, `swg21616976.txt`, `swg1IZ51168.txt`, `swg21965628.txt` / 16 | COMPLETE -> INSUFFICIENT |
| `TRAIN_Q500` (`U010`) | `swg21631488.txt` | 3 | 8 | yes; `swg27024104.txt`, `swg21631488.txt`, `swg21594791.txt`, `swg27023623.txt`, `swg21686987.txt` / 16 | no; `swg21594791.txt`, `swg27023623.txt`, `swg21608749.txt`, `swg21607198.txt`, `swg27023851.txt` / 22 | COMPLETE -> INSUFFICIENT |
| `TRAIN_Q367` (`U055`) | `swg21968549.txt` | 15 | 9 | no; `swg21903444.txt`, `swg21973739.txt`, `swg21976161.txt`, `swg21503678.txt`, `swg21968904.txt` / 25 | no; `swg21503322.txt`, `swg21976161.txt`, `swg21973739.txt`, `swg21503678.txt`, `swg21632650.txt` / 25 | PARTIAL -> INSUFFICIENT |

For `TRAIN_Q287` and `TRAIN_Q578`, the relevant document is absent from G1's admitted set and present in G2's. The G2 merged Top16 starts with `swg27047895.txt_chunk_6` and `swg21612222.txt_chunk_1`, respectively; those are the rescued relevant chunks. For `TRAIN_Q090`, G1's Top16 contains `swg24044840.txt_chunk_13` and `swg24044840.txt_chunk_14`, while G2 cannot select either because the relevant document is excluded at admission. For `TRAIN_Q500`, the gold document is `swg21631488.txt`: its Dense rank is 3 and global rerank rank is 8. G1's Top16 includes `swg21631488.txt_chunk_0`, while G2's selected Top16 comes from the changed 22-chunk pool and contains no `swg21631488.txt` chunk.

`TRAIN_Q367` is the important control against over-attributing every movement to admission rescue or exclusion: both arms exclude the gold document. The first merged selections are still different: G1 starts with `swg21976161.txt_chunk_1`, `swg21973739.txt_chunk_12`, and `swg21503678.txt_chunk_2`; G2 starts with `swg21976161.txt_chunk_1`, `swg21503322.txt_chunk_1`, and `swg21973739.txt_chunk_12`.

Material claims and blinded-review outcomes were preserved from the frozen artifacts. `TRAIN_Q287` moved from 0/1 to 1/1 covered claims; `TRAIN_Q578` moved from 0/2 to 2/2; `TRAIN_Q090` moved from 1/1 to 0/1; `TRAIN_Q500` moved from 1/1 to 0/1; and `TRAIN_Q367` moved from 1/2 to 0/2.

## 4. First divergence and mechanism findings

| Case | First G1/G2 divergence | Mechanism classification |
| --- | --- | --- |
| `TRAIN_Q287` | Document admission | Relevant document rescued: global rerank moved its Dense-rank-20/58 chunks to global ranks 1/8, bringing it into G2's five-document budget. |
| `TRAIN_Q578` | Document admission | Relevant document rescued: its Dense chunks at 14/67 became global ranks 1/11, so G2 admitted the document and G1 did not. |
| `TRAIN_Q090` | Document admission | Relevant document displaced: despite Dense ranks 2/11, global admission selected five other documents and permanently removed the useful G1 document. |
| `TRAIN_Q500` | Document admission | Relevant document displaced: G1 admitted the gold document; global admission changed the five-document set, after which the gold document had no downstream opportunity. |
| `TRAIN_Q367` | Document-admission-set composition | Both arms excluded the gold document, but their admitted document sets differ. That first admission-set divergence changes candidate-pool composition and then downstream merged Top16 selection. |

The two wins are gold-document admission rescues. The two catastrophic regressions are gold-document displacements under the rerank-informed five-document budget. `TRAIN_Q367` also first diverges at admission-set composition: both arms miss the gold document, but their different admitted sets change the downstream candidate pool and selected evidence. There is not enough evidence here to claim independent merged-rerank instability.

## 5. Decision questions

**Q1.** Yes. Both G2 wins are cases where Dense admission excluded the relevant document and global rerank admission rescued it.

**Q2.** Yes. Both catastrophic regressions are cases where G1 admitted the relevant/useful document but the global-rerank five-document budget did not.

**Q3.** For `TRAIN_Q367`, both arms miss the gold document. The first divergence is document-admission-set composition; that changes the candidate pool and downstream selected evidence.

**Q4.** **A. Admission ranking instability is the dominant mechanism, with the fixed five-document budget acting as an amplifier.** Four of five movements directly involve gold-document rescue or displacement, and `TRAIN_Q367` also first diverges at admission-set composition. These cases do not establish independent merged-rerank instability or evidence continuity as the cause.

**Q5.** **MAYBE — mechanism is suggestive but insufficient; gather broader diagnostic evidence first.** The rescue/displacement pattern is strong enough to motivate diagnostics, but five post-hoc cases are not enough to justify designing or preregistering G3. No G3 is designed or implemented here.

## 6. What this does not establish

This was a fresh TRAIN development experiment, not frozen DEV validation. It does not establish generation accuracy uplift, production latency/cost improvement, or that global reranking is generally harmful. It only rejects this specific rerank-informed five-document admission policy. The forensic findings are post-hoc and must not be promoted to confirmatory evidence.

Task 7/8 execution deviations are recorded here for audit completeness: the reviewer child high-level file-read workflow stalled; explicit Python byte I/O was adopted before any judgments were observed; no judgment or rule changed after unblinding; Task 8 required one controlled mapping reread after final-attestation Python boolean literals used `false` instead of `False`; the actual correction was four identical `false -> False` repairs in final-attestation fields; and no analytical or gate logic changed.

## 7. Recommendation

Keep E1 reference behavior unchanged and record G1 and G2-A as experimental **NO_GO** branches. Before considering another experiment, gather broader offline diagnostics over the frozen traces: admission-miss frequency, relevant-document displacement frequency, candidate-pool overlap, and the rate at which merged Top16 selection changes after admission-set changes. Any future experiment would require a new preregistration and must not use this post-hoc table as confirmatory evidence.