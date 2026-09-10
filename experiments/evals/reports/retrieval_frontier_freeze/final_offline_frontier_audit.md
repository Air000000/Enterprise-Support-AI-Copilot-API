# Retrieval Frontier Freeze: Final Offline Frontier Audit

Date: 2026-09-10

This is a zero-provider, post-hoc counterfactual audit over the frozen G2 30-case TRAIN design/diagnostic set. It is exploratory only. The cases are not a future confirmation sample.

## 1. Confirmatory results

The frozen confirmatory lineage is unchanged:

- E1 remains the reference policy and its held-out retrieval result is unchanged.
- G1 remains formal `NO_GO`.
- G2-A remains formal `NO_GO`.
- This audit makes no new formal GO/NO_GO decision.

The existing G1 and G2-A reports remain the sources for those decisions. This report does not re-review their claims.

Provider calls during this audit: reranker=0, embedding=0, generation=0, judge=0.

## 2. Post-hoc diagnostics

The prior diagnostics remain bounded as post-hoc evidence:

- G1's failure mechanism is documented in `reports/g1_document_local/final_decision.md`.
- G2-A's five-case forensic is documented in `reports/g2_rerank_informed_admission/post_hoc_forensic_analysis.md`.
- The broader frozen 30-case diagnostic found 20 `BOTH`, 2 `G1_ONLY_DISPLACED`, 2 `G2_ONLY_RESCUED`, and 6 `NEITHER`.
- Admission overlap had mean 1.83 and median 2; all 30 admission sets changed.
- The five known movements were preserved: two rescues, two displacements, and one `NEITHER` loss.

These observations are diagnostic only. They are not a new benchmark result, generation uplift claim, production claim, or confirmation sample.

## 3. Counterfactual frontier result

The counterfactuals use the frozen Dense Top100, frozen shared-global rerank, frozen corpus and splitter, and uncapped document-local chunk expansion. Policies are Dense K=5..10, Global K=5..10, document-level RRF K=5..10 with fixed RRF constant 60, and Dense5 UNION Global5. The RRF tie-break is symmetric and deterministic: min rank, max rank, then lexical document ID.

All values below are design-set-only diagnostics. `raw chunks` means a merged-rerank workload proxy, not latency, token cost, or production cost.

| Point | Gold / 30 | Rescue | Displacement | Mean chunks | P50 | P95 | Max | Total raw chunks | >=450 | >=500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense5 | 22 | 0 | 0 | 30.5 | 27 | 49 | 102 | 916 | 0 | 0 |
| Dense6 | 22 | 0 | 0 | 36.2 | 32 | 55 | 121 | 1086 | 0 | 0 |
| Dense7 | 22 | 0 | 0 | 43.9 | 39 | 61 | 142 | 1316 | 0 | 0 |
| Dense8 | 22 | 0 | 0 | 49.4 | 45 | 74 | 161 | 1481 | 0 | 0 |
| Dense9 | 22 | 0 | 0 | 55.6 | 51 | 80 | 180 | 1667 | 0 | 0 |
| Dense10 | 22 | 0 | 0 | 61.6 | 56 | 85 | 198 | 1847 | 0 | 0 |
| Global5 | 22 | 2 | 2 | 29.6 | 27 | 43 | 102 | 889 | 0 | 0 |
| Global6 | 22 | 2 | 2 | 36.1 | 32 | 55 | 121 | 1082 | 0 | 0 |
| Global7 | 23 | 3 | 2 | 43.7 | 39 | 64 | 142 | 1312 | 0 | 0 |
| Global8 | 27 | 5 | 0 | 51.6 | 47 | 92 | 161 | 1548 | 0 | 0 |
| Global9 | 27 | 5 | 0 | 59.0 | 54 | 106 | 180 | 1770 | 0 | 0 |
| Global10 | 27 | 5 | 0 | 65.8 | 60 | 121 | 198 | 1975 | 0 | 0 |
| RRF5 | 24 | 2 | 0 | 31.9 | 28 | 45 | 102 | 956 | 0 | 0 |
| RRF6 | 25 | 3 | 0 | 37.6 | 34 | 60 | 121 | 1127 | 0 | 0 |
| RRF7 | 25 | 3 | 0 | 44.3 | 40 | 71 | 142 | 1329 | 0 | 0 |
| RRF8 | 25 | 3 | 0 | 49.7 | 46 | 77 | 161 | 1492 | 0 | 0 |
| RRF9 | 25 | 3 | 0 | 55.4 | 51 | 93 | 180 | 1662 | 0 | 0 |
| RRF10 | 25 | 3 | 0 | 61.0 | 56 | 100 | 198 | 1831 | 0 | 0 |
| Dense5 UNION Global5 | 24 | 2 | 0 | 50.2 | 46 | 94 | 204 | 1505 | 0 | 0 |

Relative to Dense5, the RRF exploratory points have total workload ratios of 1.04, 1.23, 1.45, 1.63, 1.81, and 2.00 for K=5..10. The Union diagnostic has ratio 1.64. These are workload-proxy ratios only.

### Pareto and near-cost findings

Using the specified exploratory non-dominated rule, the points are:

`Dense5`, `Global5`, `Global8`, `RRF5`, `RRF6`.

Representative approximately comparable workload pairs include:

- Dense5 vs Global5: 916 vs 889 chunks, 22 vs 22 gold, 0 vs 2 rescue, 0 vs 2 displacement.
- Dense5 vs RRF5: 916 vs 956 chunks, 22 vs 24 gold, 0 vs 2 rescue, 0 vs 0 displacement.
- Dense6 vs RRF6: 1086 vs 1127 chunks, 22 vs 25 gold, 0 vs 3 rescue, 0 vs 0 displacement.
- Dense7 vs RRF7: 1316 vs 1329 chunks, 22 vs 25 gold, 0 vs 3 rescue, 0 vs 0 displacement.
- Dense8 vs Global8: 1481 vs 1548 chunks, 22 vs 27 gold, 0 vs 5 rescue, 0 vs 0 displacement.
- Dense8 vs RRF8: 1481 vs 1492 chunks, 22 vs 25 gold, 0 vs 3 rescue, 0 vs 0 displacement.
- RRF8 vs Union5: 1492 vs 1505 chunks, 25 vs 24 gold, 3 vs 2 rescue, 0 vs 0 displacement.

The near-cost pattern is descriptive: RRF preserves Dense5's 22/30 baseline hits, adds rescues, and shows zero displacement in this design set. Global K=8..10 reaches 27/30, but does so by replacing the Dense admission direction rather than preserving both rankings. Union preserves complementarity but expands the document and chunk pool more than RRF5..8 at similar coverage.

No point is called optimal, best, validated, or production-ready.

## 4. Future hypothesis

**Q1. MAIN BOTTLENECK = BOTH.** The data show a fixed admission budget boundary and genuine Dense/Global complementarity: increasing K in one ranking does not recover the other ranking's direction, while Global and Dense each retain unique gold admissions in the frozen design set.

**Q2. POLICY FAMILY = RRF_K.** Among the allowed families, document-level RRF is the clearest exploratory design direction because RRF5/RRF6/RRF7 add 2/3/3 rescues with zero displacement at workloads close to Dense K points. This is a policy-family diagnostic, not a validated configuration.

**Q3. FUTURE DIRECTION = FUTURE_G3_CANDIDATE_IDENTIFIED.**

Future hypothesis: “document-level RRF admission may preserve complementary Dense/Global coverage at lower expanded-chunk workload than simply increasing a single-ranking cutoff.”

This does not design G3, freeze a K, preregister an experiment, or authorize a paid/DEV run. Any future G3 would require new authorization, new preregistration, and fresh TRAIN confirmation cases; these 30 cases remain design/diagnostic cases and cannot be reused as confirmation evidence.

## Scope and reproducibility

The complete structured result is in `frontier_metrics.json`. The reproduction gate passed for all 30 cases: reconstructed G1/G2 raw expanded pool sizes exactly matched the frozen full `g1_merged`/`g2_merged` permutations. The corpus was read from the local frozen TechQA cache matching the manifest revision and SHA256. No provider was called.

This final offline frontier audit closes the retrieval research line. No further retrieval optimization, context-policy tuning, G3 implementation, confirmation sampling, paid rerank, DEV run, or provider call is authorized by this artifact.