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

- Answerable questions: 21
- Unanswerable questions: 9

## Retrieval Metrics

| Metric | Result |
|---|---:|
| Hit@1 | 0.952 |
| Hit@3 | 1.000 |
| MRR | 0.976 |

## Unknown Query Analysis

- Minimum Top-1 score: 0.073
- Maximum Top-1 score: 0.722
- Average Top-1 score: 0.406

These values will be compared with relevant-query similarity scores before selecting a no-answer threshold.

## Hit@1 Failures

### retry_004

Question: A downstream service temporarily returns HTTP 503. Should the client attempt the operation again?

Expected: service_retry_policy.md

Retrieved ranking: api_timeout.md, service_retry_policy.md, cache_strategy.md

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

### hard_negative_001

Question: How many seconds should the service wait before declaring a downstream request permanently failed?

Top score: 0.563

Top retrieved document: `api_timeout.md`

### hard_negative_002

Question: What exact TTL value should be configured for the application cache?

Top score: 0.722

Top retrieved document: `cache_strategy.md`

### hard_negative_003

Question: How many retry attempts must every production service perform before giving up?

Top score: 0.611

Top retrieved document: `service_retry_policy.md`

### hard_negative_004

Question: Which database backup product should be used before performing a downgrade?

Top score: 0.621

Top retrieved document: `database_downgrade.md`

## Initial Interpretation

This experiment is a baseline, not a final quality result.

The next experiment should compare the score distribution of answerable and unanswerable queries and introduce a relevance threshold only if the distributions provide a defensible separation.

Future experiments will also compare fixed-size chunking against structure-aware paragraph chunking.