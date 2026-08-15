# RAG Evaluation Report

## 1. Objective

The goal of this evaluation is to measure the quality and
reliability of the Enterprise Knowledge RAG Assistant.

The evaluation focuses on three major questions:

1. Can the retriever find relevant knowledge?
2. Can the system distinguish answerable questions from questions
   that do not have sufficient supporting evidence?
3. Can retrieval quality regressions be detected automatically
   when the RAG architecture changes?

The evaluation framework covers:

- retrieval quality
- multi-document retrieval
- hard-negative queries
- evidence sufficiency
- chunking strategy comparison
- regression quality gates

## 2. Evaluation Dataset

The evaluation dataset contains 30 synthetic engineering questions.

The dataset includes:

- direct answerable questions
- paraphrased questions
- hard-negative questions
- unanswerable questions
- multi-document questions

Dataset summary:

- Total questions: 30
- Answerable: 21
- Unanswerable: 9

All source documents and evaluation questions use synthetic
engineering data and do not contain proprietary company information.

## 3. Retrieval Baseline

The baseline retriever uses:

- sentence-transformers/all-MiniLM-L6-v2
- 384-dimensional embeddings
- cosine similarity
- brute-force in-memory semantic search
- fixed-size chunking
- chunk size: 500 characters
- overlap: 100 characters

### Retrieval Metrics

| Metric | Result |
|---|---:|
| Hit@1 | 0.952 |
| Hit@3 | 1.000 |
| MRR | 0.976 |
| Recall@1 | 0.905 |
| Recall@3 | 1.000 |

The results indicate that relevant documents are consistently
retrieved within the top three results.

Multi-document queries demonstrate why Recall@K is required in
addition to Hit@K: Hit@K only verifies that at least one relevant
document was found, while Recall@K measures coverage of all required
documents.

## 4. Similarity Threshold Analysis

A similarity-score threshold was evaluated as a possible mechanism
for determining whether a question was answerable.

Initial results on the harder evaluation dataset:

- Precision: 0.808
- Recall: 1.000
- F1: 0.894
- Accuracy: 0.833

Hard-negative questions frequently received high semantic similarity
scores despite the required answer not being present in the source
documents.

Examples included questions asking for:

- an exact cache TTL
- an exact retry count
- an exact timeout duration
- a specific database backup product

These questions were highly related to the retrieved documents but
the requested facts were not actually present.

### Conclusion

Semantic similarity measures topical relevance, not answerability.

A similarity threshold should therefore not be used as the sole
mechanism for deciding whether sufficient evidence exists.

## 5. Evidence Sufficiency Evaluation

An LLM-based Evidence Sufficiency Checker was introduced after
retrieval.

Architecture:

Question
    ↓
Semantic Retriever
    ↓
Top-K Context
    ↓
Evidence Sufficiency Checker
    ↓
SUFFICIENT / INSUFFICIENT

Current baseline:

| Metric | Result |
|---|---:|
| Accuracy | 0.933 |
| Precision | 1.000 |
| Recall | 0.905 |
| F1 | 0.950 |
| False Positives | 0 |

The checker is deliberately conservative.

False positives are treated as particularly important because they
may allow unsupported questions to reach the answer-generation stage.

### Comparison

Similarity threshold:

- Precision: 0.808
- F1: 0.894

Evidence sufficiency checker:

- Precision: 1.000
- F1: 0.950

The evidence-based approach therefore provides substantially better
control over unsupported-answer risk.

## 6. Chunking Strategy Experiment

Two chunking strategies were compared:

1. Fixed-size character chunking
2. Paragraph-aware chunking

### Document-level Results

| Metric | Fixed-size | Paragraph-aware |
|---|---:|---:|
| Hit@1 | 0.952 | 0.952 |
| Hit@3 | 1.000 | 1.000 |
| MRR | 0.976 | 0.976 |
| Recall@1 | 0.905 | 0.905 |
| Recall@3 | 1.000 | 1.000 |

Aggregate document-level metrics did not change.

However, the retry_004 case showed a ranking improvement:

Fixed-size:
- expected document rank: #2
- best expected score: 0.389705

Paragraph-aware:
- expected document rank: #1
- best expected score: 0.407508

Further evidence-level analysis showed that the answer-bearing
evidence ("HTTP 503 responses") remained outside the Top-3 chunks
for both strategies.

Evidence appeared at rank #4 for both strategies.

### Conclusion

Paragraph-aware chunking improved document relevance for this case,
but did not improve Top-3 answer-bearing evidence recall.

This demonstrates that document-level retrieval metrics alone are not
sufficient to measure the quality of context actually provided to an
LLM.

## 7. Failure Analysis

Failure analysis identified several distinct failure modes.

### Retrieval Ranking Failure

A relevant document may be retrieved but ranked below another
semantically similar document.

### Evidence Retrieval Failure

A relevant document may be present in the ranking while the specific
answer-bearing chunk is outside the Top-K context.

### Hard-Negative Failure

Documents may have high semantic similarity even when the requested
specific fact is not present.

These observations motivated the separation of:

- document relevance
- evidence coverage
- evidence sufficiency

## 8. Regression Quality Gates

Automated regression tests protect the current RAG baseline.

### Aggregate Retrieval Gates

The following metrics are monitored:

- Hit@1
- Hit@3
- MRR
- Recall@1
- Recall@3

### Case-Level Gates

Representative anchor cases verify that aggregate metric improvements
do not hide regressions in individual queries.

Both single-document and multi-document retrieval cases are covered.

### LLM Evidence Sufficiency Gates

The Evidence Sufficiency Checker is evaluated separately because it
requires external LLM calls.

Current quality gates include:

- minimum accuracy
- minimum precision
- minimum recall
- minimum F1
- maximum false-positive count

### Running Regression Tests

Fast local regression:

```bash
python -m pytest -m "not llm" -v



# 7.10.10 最重要的一部分：Architecture Decisions

最后我建议增加：

```markdown
## 9. Architecture Decisions

Based on the evaluation results, the following design decisions were
made.

### Decision 1: Do not use similarity threshold alone

Similarity is useful for ranking but is not a reliable measure of
answer availability.

### Decision 2: Use a separate evidence sufficiency stage

Evidence sufficiency is treated as a separate decision after
retrieval and before answer generation.

### Decision 3: Evaluate multi-document coverage

Hit@K alone is insufficient for questions requiring multiple sources.
Recall@K is included in the evaluation framework.

### Decision 4: Evaluate retrieval at multiple levels

Document-level retrieval and answer-bearing evidence retrieval are
different quality dimensions.

### Decision 5: Treat RAG quality as a regression-tested system

Changes to chunking, embeddings, retrieval algorithms, vector stores,
or reranking must be compared against the established baseline.

