# RRF Fusion Design (Step 9.4)

## Goal

Complete Step 9.4 only: implement and understand Reciprocal Rank
Fusion (RRF) as a small, independently testable component. Hybrid
evaluation, production-pipeline integration, reranking, and query
rewriting will be designed and implemented in later learning stages.

## Scope

Included:

- RRF fusion with deterministic ranking and duplicate chunk handling.
- The existing `RankFusionStrategy` abstraction required by RRF.
- An executable, manual RRF demonstration using the existing Dense and
  BM25 retrievers, kept outside the API pipeline.

Excluded:

- Hybrid-retrieval evaluation (Step 9.5).
- Reranking and reranking evaluation (Steps 9.6–9.7).
- Query rewrite (Step 9.8).
- Final retrieval architecture and API-pipeline integration (Step 9.9).
- Step 10 packaging, deployment, and portfolio documentation.

## Architecture

RRF accepts ranked result lists from existing retrievers and returns a
single ranked `list[RetrievalResult]`.

```text
dense ranked candidates ─┐
                         ├─> RRF ─> fused ranked candidates
BM25 ranked candidates ──┘
```

The existing API remains unchanged and continues to use semantic
retrieval. The manual demonstration is a learning aid, not a new
production path.

### RRF

Each retriever returns the same candidate count. RRF assigns a chunk
score of `1 / (k + rank)` for every list in which the chunk occurs,
sums those scores by `chunk_id`, and sorts descending by the fused
score. Ties are broken deterministically by chunk ID so repeated runs
produce the same order. RRF only uses rank positions; it intentionally
does not compare incompatible Dense cosine scores with BM25 scores.

## Testing

Tests precede implementation and cover:

- RRF score accumulation, duplicate chunks, deterministic ties, and
  top-K truncation.
- A manual end-to-end demonstration showing each source ranking and
  the fused RRF ranking for the same question.

Unit tests for RRF use small in-memory `RetrievalResult` objects and
therefore do not download models or call external services.

## Acceptance criteria

- RRF correctly accumulates contributions for chunks shared by two
  rankings.
- RRF correctly handles chunks found by only one retriever.
- Equal fused scores have a stable, documented order.
- The manual demonstration clearly shows Dense, BM25, and fused output
  for the same query.
- The existing API and evaluation pipeline remain unchanged.
