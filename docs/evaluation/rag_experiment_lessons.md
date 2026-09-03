# RAG Experiment Lessons

This document records only lessons that changed an engineering decision. Each entry keeps four parts: observation, mechanism, boundary, and decision.

## 1. Retrieval hit is not complete evidence

**Observation.** Q055 and Q572 hit the correct document, but the Top3 generation context missed the decisive sibling chunks. Adding document-local context moved correctness from `0.2 -> 0.6` and `0.2 -> 0.9`.

**Mechanism.** Retrieval ranking identifies promising locations; it does not guarantee evidence completeness across artificial chunk boundaries.

**Boundary.** This does not prove `forward +3` is a general strategy. Its direction and window size were shaped by known TRAIN failures.

**Decision.** Separate chunk retrieval from document-local evidence selection. Use adjacency only to restore local continuity after evidence is selected.

## 2. Learn mechanisms from benchmark failures, not case shapes

**Observation.** Two known failures placed decisive evidence after the retrieved anchor, making a forward window look attractive.

**Mechanism.** Chunk indices are implementation artifacts, not semantic coordinates. Evidence may appear before, after, or far from the first matching chunk.

**Boundary.** TRAIN is appropriate for discovering failure classes, but not for turning instance geometry into a production invariant.

**Decision.** Prefer rules that survive changes in chunk size, document structure, domain, and query type; validate strategy choices on data not used to invent them.

## 3. Relevance is not answerability

**Observation.** Several impossible TechQA questions still had low Dense Top1 distance, yet the available context did not contain sufficient answer evidence.

**Mechanism.** Semantic similarity answers "is there related text?"; answerability asks "does the retrieved evidence support the requested claim?" These are different variables.

**Boundary.** A single Dense or rerank threshold may be a useful signal, but it is not evidence sufficiency by definition.

**Decision.** Treat abstention/answerability as a separate evaluation and calibration problem rather than overloading retrieval similarity.

## 4. Metric implementation is not semantic behavior

**Observation.** The evaluator counted abstention only when generation exactly matched one refusal string. Q447 and Q485 expressed insufficient evidence with explanation but were recorded as non-abstentions and therefore hallucinations.

**Mechanism.** A convenient string predicate is only a proxy for semantic refusal behavior.

**Boundary.** The observed `hallucination_rate=1.0` cannot be read directly as a semantic 100% hallucination rate.

**Decision.** Audit metric definitions when case behavior and aggregate metrics disagree; do not optimize a proxy before validating that it represents the intended behavior.

## 5. Retrieval granularity and evaluation granularity can differ

**Observation.** TechQA retrieval operates on chunks while qrels identify relevant Technote documents. R4 already needed chunk-level diagnostics and document-level collapse.

**Mechanism.** Retrieval unit, grouping/parent unit, evaluation unit, and generation-context unit serve different purposes and need not be identical.

**Boundary.** A chunk belonging to the relevant document is not automatically a gold evidence chunk; TechQA provides no gold chunk annotation.

**Decision.** Keep formal retrieval metrics at document level, retain chunk-level diagnostics for coverage/diversity, and let generation consume multiple selected evidence chunks when useful.

## 6. More components or more context do not imply monotonic improvement

**Observation.** Hybrid retrieval produced both rescue cases and fusion-loss cases; document-aware expansion repaired Q055/Q572 while Q299 regressed because decisive evidence disappeared from the final context.

**Mechanism.** Ranking fusion and context assembly redistribute finite candidate/context budgets. Gains for one case can displace useful evidence for another.

**Boundary.** Aggregate improvement alone does not explain which mechanism improved or regressed.

**Decision.** Report aggregate metrics together with paired case movement and root-cause analysis before accepting a strategy.

## 7. Preserve expensive artifacts before assuming they are lost

**Observation.** A TechQA Chroma restart failure initially looked like a lost paid embedding index, but vectors were recoverable from HNSW storage plus the embeddings queue without provider calls.

**Mechanism.** Index graph corruption or reopen failure is not equivalent to vector-value loss.

**Boundary.** Salvage is only valid after integrity checks; this is not an argument to ignore real corruption.

**Decision.** Inspect metadata, persistent segments, queue/WAL, and binary recoverability before authorizing a paid rebuild.
