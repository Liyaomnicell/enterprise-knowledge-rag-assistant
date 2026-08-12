# Retrieval Baseline v1

## Experiment Objective

Establish the initial retrieval quality baseline before introducing retrieval thresholds, alternative chunking strategies, vector databases, or reranking.

## Configuration

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Chunking strategy: fixed-size character chunking
- Chunk size: 500 characters
- Chunk overlap: 100 characters
- Similarity metric: cosine similarity
- Vector search: brute-force in-memory scan

## Evaluation Dataset

- Answerable questions: 15
- Unanswerable questions: 5

## Retrieval Metrics

| Metric | Result |
|---|---:|
| Hit@1 | 1.000 |
| Hit@3 | 1.000 |
| MRR | 1.000 |

## Unknown Query Analysis

- Minimum Top-1 score: 0.073
- Maximum Top-1 score: 0.376
- Average Top-1 score: 0.228

These values will be compared with relevant-query similarity scores before selecting a no-answer threshold.

## Hit@1 Failures

No Hit@1 failures were observed.
## Unknown Query Details

### unknown_001

Question: What is the company's vacation policy?

Top score: 0.240

Top retrieved document: `service_retry_policy.md`

### unknown_002

Question: How many paid sick days do employees receive?

Top score: 0.073

Top retrieved document: `release_checklist.md`

### unknown_003

Question: What is the recommended Kubernetes autoscaling policy?

Top score: 0.257

Top retrieved document: `service_retry_policy.md`

### unknown_004

Question: Which programming language should the frontend team use?

Top score: 0.193

Top retrieved document: `release_checklist.md`

### unknown_005

Question: What is the disaster recovery RTO for the production system?

Top score: 0.376

Top retrieved document: `service_retry_policy.md`

## Initial Interpretation

This experiment is a baseline, not a final quality result.

The next experiment should compare the score distribution of answerable and unanswerable queries and introduce a relevance threshold only if the distributions provide a defensible separation.

Future experiments will also compare fixed-size chunking against structure-aware paragraph chunking.