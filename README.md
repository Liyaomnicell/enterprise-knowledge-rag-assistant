# Enterprise Knowledge RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) system for enterprise knowledge search and grounded question answering.

This project demonstrates the evolution of a RAG retrieval pipeline from a dense semantic-search baseline to a hybrid architecture combining persistent vector search, lexical retrieval, rank fusion, and cross-encoder reranking.

A central focus of the project is **evaluation-driven engineering**: retrieval techniques are measured against a fixed evaluation dataset rather than assumed to improve quality.

---

## Key Results

| Metric | Dense Baseline | Final Reranked Hybrid |
|---|---:|---:|
| Hit@1 | 0.905 | **0.952** |
| Hit@3 | 1.000 | **1.000** |
| MRR | 0.952 | **0.976** |
| Recall@1 | 0.857 | **0.905** |
| Recall@3 | 1.000 | **1.000** |

Hybrid retrieval with RRF alone reduced Hit@1 from `0.905` to `0.857`. Adding cross-encoder reranking subsequently improved Hit@1 to `0.952`.

A lightweight query rewriting experiment was also evaluated and rejected because it produced no measurable improvement.

---

## Final Retrieval Architecture

### Indexing Pipeline

```text
Documents
    |
    v
Document Loader
    |
    v
Chunking
    |
    v
Embedding Service
    |
    v
Indexing Service
    |
    v
FAISS HNSW
    |
    v
Persistent Vector Index
```

### Query Pipeline

```text
                         User Query
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Dense Retrieval                BM25 Retrieval
              |                             |
              v                             v
         FAISS HNSW                  Lexical Ranking
              |                             |
              +--------------+--------------+
                             |
                             v
                  Reciprocal Rank Fusion
                             |
                             v
                     Candidate Top-N
                             |
                             v
                  Cross-Encoder Reranker
                             |
                             v
                     Final Top-K Chunks
                             |
                             v
                  Grounded Answer Generation
```

The final retrieval architecture consists of:

- dense semantic retrieval
- FAISS persistent vector indexing
- HNSW approximate nearest neighbor search
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- cross-encoder reranking

---

## Technology Stack

- Python
- Sentence Transformers
- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L6-v2`
- FAISS
- HNSW
- BM25
- OpenAI-compatible LLM API
- Pytest

---

## Project Architecture

```text
EmbeddingService
       |
       v
SemanticRetriever
       |
       v
VectorStore
       |
       +-- InMemoryVectorStore
       |
       +-- FaissVectorStore


BM25Retriever
       |
       +------------------+
                          |
SemanticRetriever --------+--> HybridRetriever
                                  |
                                  v
                         RankFusionStrategy
                                  |
                                  v
                    ReciprocalRankFusion


HybridRetriever
       |
       v
RerankingRetriever
       |
       v
CrossEncoderReranker
```

This separation allows the retrieval infrastructure to evolve without coupling storage, retrieval strategy, and ranking logic.

---

## Retrieval Evolution

### 1. Dense Semantic Retrieval

The initial retrieval pipeline used dense embeddings and cosine similarity with `sentence-transformers/all-MiniLM-L6-v2`.

| Metric | Result |
|---|---:|
| Hit@1 | 0.905 |
| Hit@3 | 1.000 |
| MRR | 0.952 |
| Recall@1 | 0.857 |
| Recall@3 | 1.000 |

### 2. Vector Store Abstraction

`SemanticRetriever` was refactored to depend on a `VectorStore` abstraction. Implemented backends include:

- in-memory brute-force search
- FAISS Flat exact search
- FAISS HNSW approximate nearest neighbor search

All three produced identical retrieval metrics on the current evaluation dataset.

### 3. Persistent FAISS Index

FAISS persistence was added so document embeddings do not need to be regenerated every time the application starts.

The implementation supports `IndexFlatIP` and `IndexHNSWFlat`, with L2-normalized vectors so inner-product search preserves cosine-similarity semantics.

### 4. Metadata and Filtering

Metadata is propagated from source documents to chunks.

```json
{
  "category": "database",
  "document_type": "troubleshooting"
}
```

The current FAISS implementation uses lightweight post-filtering.

### 5. BM25 Lexical Retrieval

BM25 complements dense semantic retrieval for exact technical terminology such as HTTP status codes, error codes, version numbers, and discriminative keywords.

Dense and BM25 achieved identical aggregate metrics, but case-level evaluation demonstrated complementary behavior.

### 6. Hybrid Retrieval and RRF

Dense and BM25 rankings are fused using Reciprocal Rank Fusion:

```text
RRF(d) = SUM 1 / (k + rank_i(d))
```

with conventional default `k = 60`.

Hybrid + RRF alone reduced Top-1 ranking quality:

| Metric | Dense | BM25 | Hybrid + RRF |
|---|---:|---:|---:|
| Hit@1 | 0.905 | 0.905 | 0.857 |
| Hit@3 | 1.000 | 1.000 | 1.000 |
| MRR | 0.952 | 0.952 | 0.929 |
| Recall@1 | 0.857 | 0.857 | 0.810 |
| Recall@3 | 1.000 | 1.000 | 1.000 |

This demonstrated that hybrid retrieval must be evaluated empirically rather than assumed to improve quality.

### 7. Cross-Encoder Reranking

The hybrid candidate set is reranked using `cross-encoder/ms-marco-MiniLM-L6-v2`.

The reranked pipeline produced the best result:

| Metric | Dense Baseline | Hybrid + RRF | Reranked Hybrid |
|---|---:|---:|---:|
| Hit@1 | 0.905 | 0.857 | **0.952** |
| Hit@3 | 1.000 | 1.000 | **1.000** |
| MRR | 0.952 | 0.929 | **0.976** |
| Recall@1 | 0.857 | 0.810 | **0.905** |
| Recall@3 | 1.000 | 1.000 | **1.000** |

### 8. Query Rewriting Experiment

A lightweight rule-based query rewriter was evaluated.

| Strategy | Hit@1 | MRR |
|---|---:|---:|
| Dense | 0.905 | 0.952 |
| Rewritten Dense | 0.905 | 0.944 |
| Reranked Hybrid | 0.952 | 0.976 |
| Rewritten Reranked Hybrid | 0.952 | 0.976 |

The rewrite produced no measurable improvement in the final pipeline and was therefore not adopted.

---

## Evaluation Framework

The evaluation dataset contains 30 cases:

- 21 answerable queries
- 9 unanswerable queries

It includes direct questions, paraphrased questions, hard-negative cases, multi-document questions, and unanswerable questions.

Metrics include Hit@1, Hit@3, Mean Reciprocal Rank (MRR), Recall@1, and Recall@3.

Evaluation is performed at both aggregate and case level.

---

## Testing

Run the non-LLM test suite:

```bash
python -m pytest -m "not llm" -v
```

Run the complete test suite:

```bash
python -m pytest -v
```

Run retrieval evaluation:

```bash
python -m evaluation.evaluate_retrieval
```

---

## Grounded Answer Generation

Retrieved chunks are passed to the answer-generation layer as supporting context.

The RAG pipeline returns an answer together with sources, and an evidence checker is used as part of the quality-engineering workflow.

---

## Engineering Decisions

### Why separate VectorStore from SemanticRetriever?

To decouple retrieval behavior from storage/search infrastructure and allow migration from InMemory to FAISS Flat and FAISS HNSW without changing the retriever interface.

### Why BM25?

Dense retrieval and lexical retrieval fail in different ways. BM25 adds a strong signal for exact technical terminology.

### Why RRF?

Dense cosine similarity and BM25 scores are not directly comparable. RRF combines ranking positions without requiring score calibration.

### Why keep RRF after RRF alone reduced Hit@1?

Because the complete candidate-generation + reranking pipeline achieved the best overall result.

### Why use a Cross-Encoder?

Top-3 recall was already perfect, indicating that the remaining issue was primarily candidate ordering rather than candidate recall.

### Why reject query rewriting?

Because evaluation showed no measurable benefit.

---

## Design Principles

1. Separate retrieval behavior from storage infrastructure.
2. Establish measurable baselines before optimization.
3. Preserve regression tests when changing architecture.
4. Treat retrieval, fusion, and reranking as separate responsibilities.
5. Evaluate both aggregate metrics and individual failure cases.
6. Avoid comparing incompatible raw relevance scores directly.
7. Prefer architecture changes supported by measurable evidence.
8. Reject features that do not provide measurable value.
9. Avoid artificial scalability claims on a small dataset.

---

## Limitations

- small synthetic enterprise knowledge corpus
- limited evaluation dataset
- lightweight BM25 tokenization
- FAISS metadata filtering implemented through post-filtering
- fixed conventional RRF parameter `k=60`
- general-purpose MS MARCO cross-encoder rather than a domain-specific model
- no distributed vector infrastructure
- no production authentication or authorization layer
- no large-scale ANN latency benchmark

---

## Out of Scope

The following techniques are intentionally outside this project's scope:

- GraphRAG
- Self-RAG
- Corrective RAG
- Adaptive RAG
- ColBERT
- complex HyDE pipelines
- multi-query retrieval
- agentic retrieval
- MCP
- multi-agent orchestration

The objective is to demonstrate a well-engineered enterprise RAG system rather than accumulate every available RAG technique.

---

## Final Result

The retrieval architecture evolved from dense semantic retrieval to:

```text
FAISS HNSW Dense Retrieval
          +
        BM25
          |
          v
         RRF
          |
          v
Cross-Encoder Reranking
          |
          v
     Final Top-K
```

The final pipeline improved Hit@1 from `0.905` to `0.952`, MRR from `0.952` to `0.976`, and Recall@1 from `0.857` to `0.905`, while maintaining Hit@3 and Recall@3 at `1.000`.

The primary result of the project is not simply the final metric improvement, but the evaluation-driven engineering process used to arrive at the final architecture.
