import json
from pathlib import Path
from datetime import datetime

from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever

EVALUATION_FILE = "data/evaluation/retrieval_eval.json"

RESULT_DIR = Path("evaluation/results")
JSON_RESULT_FILE = RESULT_DIR / "baseline_v1.json"
MARKDOWN_RESULT_FILE = RESULT_DIR / "baseline_v1.md"

def load_evaluation_dataset():
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_retriever():
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = EmbeddingService()

    retriever = SemanticRetriever(
        chunks=chunks,
        embedding_service=embedding_service,
    )

    return retriever


def get_document_ranking(results):
    documents = []

    for result in results:
        document_name = result.chunk.document_name

        if document_name not in documents:
            documents.append(document_name)

    return documents


def reciprocal_rank(
    ranked_documents,
    expected_documents,
):
    for index, document in enumerate(
        ranked_documents,
        start=1,
    ):
        if document in expected_documents:
            return 1.0 / index

    return 0.0


def evaluate_answerable_queries(
    dataset,
    retriever,
):
    results_detail = []

    total_answerable = 0
    hit_at_1_count = 0
    hit_at_3_count = 0
    reciprocal_rank_sum = 0.0

    for item in dataset:

        if not item["answerable"]:
            continue

        total_answerable += 1

        retrieval_results = retriever.retrieve(
            query=item["question"],
            top_k=10,
        )

        ranked_documents = get_document_ranking(
            retrieval_results
        )

        expected_documents = item[
            "expected_documents"
        ]

        top_1 = ranked_documents[:1]
        top_3 = ranked_documents[:3]

        hit1 = any(
            document in expected_documents
            for document in top_1
        )

        hit3 = any(
            document in expected_documents
            for document in top_3
        )

        rr = reciprocal_rank(
            ranked_documents,
            expected_documents,
        )

        if hit1:
            hit_at_1_count += 1

        if hit3:
            hit_at_3_count += 1

        reciprocal_rank_sum += rr

        retrieved_chunks = []

        for result in retrieval_results:
            retrieved_chunks.append({
                "document_name":
                    result.chunk.document_name,
                "chunk_id":
                    result.chunk.chunk_id,
                "score":
                    round(result.score, 6),
            })

        results_detail.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_documents":
                expected_documents,
            "ranked_documents":
                ranked_documents,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "reciprocal_rank":
                round(rr, 6),
            "retrieved_chunks":
                retrieved_chunks,
        })

    metrics = {
        "question_count":
            total_answerable,
        "hit_at_1":
            round(
                hit_at_1_count
                / total_answerable,
                6,
            ),
        "hit_at_3":
            round(
                hit_at_3_count
                / total_answerable,
                6,
            ),
        "mrr":
            round(
                reciprocal_rank_sum
                / total_answerable,
                6,
            ),
    }

    return metrics, results_detail


def evaluate_unknown_queries(
    dataset,
    retriever,
):
    unknown_results = []

    top_scores = []

    for item in dataset:

        if item["answerable"]:
            continue

        retrieval_results = retriever.retrieve(
            query=item["question"],
            top_k=3,
        )

        retrieved_chunks = []

        for result in retrieval_results:
            retrieved_chunks.append({
                "document_name":
                    result.chunk.document_name,
                "chunk_id":
                    result.chunk.chunk_id,
                "score":
                    round(result.score, 6),
            })

        top_score = (
            retrieval_results[0].score
            if retrieval_results
            else 0.0
        )

        top_scores.append(top_score)

        unknown_results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "top_score":
                round(top_score, 6),
            "retrieved_chunks":
                retrieved_chunks,
        })

    score_summary = {
        "question_count":
            len(unknown_results),
        "min_top_score":
            round(min(top_scores), 6)
            if top_scores else 0.0,
        "max_top_score":
            round(max(top_scores), 6)
            if top_scores else 0.0,
        "average_top_score":
            round(
                sum(top_scores)
                / len(top_scores),
                6,
            )
            if top_scores
            else 0.0,
    }

    return score_summary, unknown_results


def save_json_result(
    metrics,
    answerable_results,
    unknown_summary,
    unknown_results,
):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment = {
        "experiment": {
            "name": "baseline_v1",
            "created_at":
                datetime.now().isoformat(),
            "description":
                (
                    "Initial semantic retrieval "
                    "baseline using fixed-size "
                    "character chunking."
                ),
        },

        "configuration": {
            "embedding_model":
                (
                    "sentence-transformers/"
                    "all-MiniLM-L6-v2"
                ),
            "chunking_strategy":
                "fixed_character",
            "chunk_size":
                500,
            "chunk_overlap":
                100,
            "retrieval_method":
                "brute_force_cosine_similarity",
            "retrieval_level":
                "chunk",
        },

        "dataset": {
            "source":
                EVALUATION_FILE,
            "answerable_questions":
                metrics["question_count"],
            "unanswerable_questions":
                unknown_summary[
                    "question_count"
                ],
        },

        "metrics": metrics,

        "unknown_query_analysis":
            unknown_summary,

        "answerable_results":
            answerable_results,

        "unknown_results":
            unknown_results,
    }

    with open(
        JSON_RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment,
            file,
            indent=2,
            ensure_ascii=False,
        )


def save_markdown_result(
    metrics,
    answerable_results,
    unknown_summary,
    unknown_results,
):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_hit1 = [
        result
        for result in answerable_results
        if not result["hit_at_1"]
    ]

    lines = []

    lines.append(
        "# Retrieval Baseline v1"
    )

    lines.append("")

    lines.append(
        "## Experiment Objective"
    )

    lines.append("")

    lines.append(
        "Establish the initial retrieval "
        "quality baseline before introducing "
        "retrieval thresholds, alternative "
        "chunking strategies, vector databases, "
        "or reranking."
    )

    lines.append("")

    lines.append(
        "## Configuration"
    )

    lines.append("")

    lines.append(
        "- Embedding model: "
        "`sentence-transformers/"
        "all-MiniLM-L6-v2`"
    )

    lines.append(
        "- Chunking strategy: "
        "fixed-size character chunking"
    )

    lines.append(
        "- Chunk size: 500 characters"
    )

    lines.append(
        "- Chunk overlap: 100 characters"
    )

    lines.append(
        "- Similarity metric: cosine similarity"
    )

    lines.append(
        "- Vector search: brute-force "
        "in-memory scan"
    )

    lines.append("")

    lines.append(
        "## Evaluation Dataset"
    )

    lines.append("")

    lines.append(
        f"- Answerable questions: "
        f"{metrics['question_count']}"
    )

    lines.append(
        f"- Unanswerable questions: "
        f"{unknown_summary['question_count']}"
    )

    lines.append("")

    lines.append(
        "## Retrieval Metrics"
    )

    lines.append("")

    lines.append(
        "| Metric | Result |"
    )

    lines.append(
        "|---|---:|"
    )

    lines.append(
        f"| Hit@1 | "
        f"{metrics['hit_at_1']:.3f} |"
    )

    lines.append(
        f"| Hit@3 | "
        f"{metrics['hit_at_3']:.3f} |"
    )

    lines.append(
        f"| MRR | "
        f"{metrics['mrr']:.3f} |"
    )

    lines.append("")

    lines.append(
        "## Unknown Query Analysis"
    )

    lines.append("")

    lines.append(
        f"- Minimum Top-1 score: "
        f"{unknown_summary['min_top_score']:.3f}"
    )

    lines.append(
        f"- Maximum Top-1 score: "
        f"{unknown_summary['max_top_score']:.3f}"
    )

    lines.append(
        f"- Average Top-1 score: "
        f"{unknown_summary['average_top_score']:.3f}"
    )

    lines.append("")

    lines.append(
        "These values will be compared with "
        "relevant-query similarity scores "
        "before selecting a no-answer threshold."
    )

    lines.append("")

    lines.append(
        "## Hit@1 Failures"
    )

    lines.append("")

    if not failed_hit1:
        lines.append(
            "No Hit@1 failures were observed."
        )
    else:
        for result in failed_hit1:
            lines.append(
                f"### {result['id']}"
            )

            lines.append("")

            lines.append(
                f"Question: "
                f"{result['question']}"
            )

            lines.append("")

            lines.append(
                "Expected: "
                + ", ".join(
                    result[
                        "expected_documents"
                    ]
                )
            )

            lines.append("")

            lines.append(
                "Retrieved ranking: "
                + ", ".join(
                    result[
                        "ranked_documents"
                    ][:3]
                )
            )

            lines.append("")

    lines.append(
        "## Unknown Query Details"
    )

    lines.append("")

    for result in unknown_results:

        lines.append(
            f"### {result['id']}"
        )

        lines.append("")

        lines.append(
            f"Question: "
            f"{result['question']}"
        )

        lines.append("")

        lines.append(
            f"Top score: "
            f"{result['top_score']:.3f}"
        )

        lines.append("")

        if result["retrieved_chunks"]:
            first = (
                result[
                    "retrieved_chunks"
                ][0]
            )

            lines.append(
                "Top retrieved document: "
                f"`{first['document_name']}`"
            )

        lines.append("")

    lines.append(
        "## Initial Interpretation"
    )

    lines.append("")

    lines.append(
        "This experiment is a baseline, "
        "not a final quality result."
    )

    lines.append("")

    lines.append(
        "The next experiment should compare "
        "the score distribution of answerable "
        "and unanswerable queries and introduce "
        "a relevance threshold only if the "
        "distributions provide a defensible "
        "separation."
    )

    lines.append("")

    lines.append(
        "Future experiments will also compare "
        "fixed-size chunking against "
        "structure-aware paragraph chunking."
    )

    with open(
        MARKDOWN_RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )


def evaluate():

    dataset = load_evaluation_dataset()

    retriever = build_retriever()

    (
        metrics,
        answerable_results,
    ) = evaluate_answerable_queries(
        dataset,
        retriever,
    )

    (
        unknown_summary,
        unknown_results,
    ) = evaluate_unknown_queries(
        dataset,
        retriever,
    )

    save_json_result(
        metrics,
        answerable_results,
        unknown_summary,
        unknown_results,
    )

    save_markdown_result(
        metrics,
        answerable_results,
        unknown_summary,
        unknown_results,
    )

    print()
    print("=" * 80)
    print("RETRIEVAL BASELINE V1")
    print("=" * 80)

    print(
        f"Hit@1: "
        f"{metrics['hit_at_1']:.3f}"
    )

    print(
        f"Hit@3: "
        f"{metrics['hit_at_3']:.3f}"
    )

    print(
        f"MRR: "
        f"{metrics['mrr']:.3f}"
    )

    print()

    print(
        "Unknown query Top-1 scores:"
    )

    print(
        f"Min: "
        f"{unknown_summary['min_top_score']:.3f}"
    )

    print(
        f"Max: "
        f"{unknown_summary['max_top_score']:.3f}"
    )

    print(
        f"Avg: "
        f"{unknown_summary['average_top_score']:.3f}"
    )

    print()

    print(
        f"JSON result: "
        f"{JSON_RESULT_FILE}"
    )

    print(
        f"Markdown report: "
        f"{MARKDOWN_RESULT_FILE}"
    )

if __name__ == "__main__":
    evaluate()
