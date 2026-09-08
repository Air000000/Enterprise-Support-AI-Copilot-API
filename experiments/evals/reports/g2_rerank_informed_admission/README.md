# G2-A rerank-informed document admission

G1 remains the reference and evaluated predecessor. G2-A is an offline experiment,
not serving code. It changes only document-admission ranking: G1 admits the first
five unique documents in frozen E0 Dense Top100 order; G2-A admits the first five
in one shared global rerank of those same 100 chunks. Both arms retain the same
document expansion, 500-chunk pool cap, one merged document-local rerank,
Top16 evidence selection, and 16-chunk final context cap. Continuity, sibling
expansion, extra rerank passes, query rewriting, tuning, and serving changes remain
explicitly out of scope.

The previous 30 G1 cases are excluded from confirmation. The fresh 30 IDs,
94-ID exclusion-union hash, budgets, and GO/NO-GO gates are frozen in
[preregistration_v1.json](preregistration_v1.json). No resampling or threshold
tuning may follow inspection of preflight capacities or later outcomes.

## Offline preflight

`run_offline_admission_preflight` in `experiments/evals/eval_techqa_g2_admission.py`
accepts a mapping of frozen E0 traces (`question_id` to `G0RetrievedChunk` sequence),
a frozen corpus chunk loader, a mapping of precomputed shared `RerankResult`s,
and a mapping of precomputed `g1`/`g2` merged `RerankResult`s per question ID.
There is no provider callback or credential argument. The caller must use the
preregistered 30-case subset and a loader backed by the frozen local corpus;
the function deliberately does not load datasets or run retrieval itself.

The existing diagnostic executes once per arm, with budgets 5/500/16/16. G1's
admission replay is an in-memory reordering into Dense order, not a provider pass.
Both merged results are replayed in memory. Shared replay must preserve all 100
Dense chunk identities and document associations. Invalid loaded/replayed IDs,
duplicate replay IDs, capacity violations, and empty case sets fail the preflight.
Loaded duplicate chunks are deduplicated by the existing G1 helper.

The returned JSON-compatible summary can be persisted with `output_path=Path(...)`.
It records descriptive candidate/context counts only, alongside identity fields
needed for reproducibility. It contains no quality scores, GO decision, or tuning
recommendations. SHA256 covers the UTF-8 canonical JSON array sorted by question
ID, with sorted object keys, `ensure_ascii=False`, and separators `(',', ':')`.
Each case contains exactly `question_id`, `dense_chunk_ids`,
`g1_admitted_doc_ids`, `g2_admitted_doc_ids`, `g1_candidate_chunk_ids`, and
`g2_candidate_chunk_ids`; list ordering is preserved. Run twice with identical
frozen inputs and require equal fingerprints. Fake rerank order demonstrates
mechanical correctness only and is not evidence of G2 quality or real capacity
under a provider ranking.

Run local verification with no API credentials:

```powershell
conda run --no-capture-output -n enterprise-g1-verify python -m pytest tests/test_eval_techqa_g2_admission.py tests/test_generation_eval_runner.py -q
conda run --no-capture-output -n enterprise-g1-verify ruff check experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
conda run --no-capture-output -n enterprise-g1-verify python -m compileall -q experiments/evals/eval_techqa_g2_admission.py tests/test_eval_techqa_g2_admission.py
```

Tripwire tests fail on live Dense/Chroma queries, provider global rerank,
generation, judging, embeddings, or network connections. Paid execution is
authorized only after preflight passes; it is a separate later task governed by
the frozen preregistration. Task 5 does not execute paid providers.
