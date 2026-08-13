# Retrieval Threshold Analysis

## Objective

Analyze whether cosine similarity scores can distinguish answerable queries from queries that are not supported by the knowledge base.

## Positive Score Distribution

Positive scores are the highest similarity scores from the known ground-truth documents for answerable queries.

- Count: 21
- Minimum: 0.368
- Maximum: 0.793
- Mean: 0.581
- Median: 0.548
- Std Dev: 0.137

## Negative Score Distribution

Negative scores are the Top-1 retrieval scores for queries labeled as unanswerable.

- Count: 9
- Minimum: 0.073
- Maximum: 0.722
- Mean: 0.406
- Median: 0.376
- Std Dev: 0.216

## Distribution Overlap

The positive and negative score distributions overlap.

A single cosine similarity threshold cannot perfectly separate answerable and unanswerable queries on the current dataset.

## Best Candidate Threshold

- Threshold: 0.312
- Accuracy: 0.833
- Precision: 0.808
- Recall: 1.000
- F1: 0.894

## Confusion Matrix

| | Predicted Answerable | Predicted Unanswerable |
|---|---:|---:|
| Actually Answerable | 21 | 0 |
| Actually Unanswerable | 5 | 4 |

## Interpretation

False positives are particularly important in RAG systems because they represent unsupported queries that would still be treated as answerable, increasing hallucination risk.

False negatives represent valid questions that the system incorrectly rejects.

The recommended threshold is only a candidate derived from the current small synthetic evaluation dataset. It should not be considered production-calibrated.