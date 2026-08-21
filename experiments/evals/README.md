# Evaluation

This directory contains the offline evaluation harness for Enterprise Support AI Copilot.

## Evaluation surfaces

The existing enterprise question sets remain regression/smoke tests. They are not the primary resume benchmark.

Phase 1 uses TechQA for two related but distinct purposes:

- Retrieval: the full 28,481-document Technote corpus with 610 answerable queries and deterministic qrels.
- Generation/abstention: NVIDIA TechQA-RAG-Eval with 910 QA records, including 610 answerable and 300 impossible questions.

TechQA is treated as a single-evidence-document Technical Support benchmark. It is not used to claim multi-document or multi-source Agentic RAG capability. Multi-source, conflict, completeness, and agentic stress testing is deferred to the later EnterpriseRAG-Bench phase.

## Split contract

Question IDs preserve original provenance:

- `TRAIN_*`: development, failure analysis, and parameter selection.
- `DEV_*`: frozen held-out comparison after E0/E1 configuration is fixed.

Do not inspect individual DEV failures to tune E1 before the held-out comparison.

## Retrieval contract

The online RAG retrieves chunks, while TechQA qrels are document-level.

For formal IR evaluation:

1. preserve the raw chunk ranking;
2. collapse chunks to unique document IDs;
3. retain a document at the rank where its first chunk appears;
4. evaluate the collapsed document ranking against qrels.

Each answerable TechQA retrieval query has exactly one relevant Technote. Therefore Document Recall@K and Hit@K are numerically equivalent on this benchmark; MRR additionally measures first relevant rank.

## Leakage rules

Do not:

- construct the retrieval corpus from NVIDIA gold contexts;
- use gold answers during retrieval;
- substitute gold contexts for actual retrieved context in end-to-end evaluation;
- hard-code question IDs to document IDs;
- tune E1 from individual DEV failures;
- remove hard cases to improve reported metrics.

## Reproducibility

Formal runs must record project commit SHA, dataset revisions, raw-file SHA256 values, loaded counts, split rules, model/config identities, latency, and dependency versions.

See `datasets/techqa/manifest.json` and `datasets/techqa/corpus_manifest.json` for the frozen Phase 1 data contract.
