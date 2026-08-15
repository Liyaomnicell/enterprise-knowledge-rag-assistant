import json
from pathlib import Path

from app.rag.document_loader import load_documents
from app.rag.chunker import (
    chunk_documents,
    chunk_document_by_paragraph,
)
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever

from evaluation.evaluate_retrieval import (
    load_evaluation_dataset,
    evaluate_answerable_queries,
    evaluate_unknown_queries,
)

from app.rag.in_memory_vector_store import (
    InMemoryVectorStore,
)

from app.rag.indexing import (
    IndexingService,
)

OUTPUT_FILE = Path(
    "evaluation/results/chunking_comparison.json"
)


def build_fixed_size_retriever():
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = (
        EmbeddingService()
    )

    vector_store = (
        InMemoryVectorStore()
    )

    indexing_service = (
        IndexingService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    indexing_service.index(
        chunks
    )

    return SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def build_paragraph_retriever():
    documents = load_documents(
        "data/documents"
    )

    chunks = []

    for document in documents:
        document_chunks = (
            chunk_document_by_paragraph(
                document,
                max_chunk_size=500,
            )
        )

        chunks.extend(
            document_chunks
        )

    embedding_service = (
        EmbeddingService()
    )

    vector_store = (
        InMemoryVectorStore()
    )

    indexing_service = (
        IndexingService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    indexing_service.index(
        chunks
    )

    return SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )


def evaluate_strategy(
    name,
    retriever,
    dataset,
):
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

    return {
        "strategy": name,
        "metrics": metrics,
        "unknown_query_analysis":
            unknown_summary,
        "answerable_results":
            answerable_results,
        "unknown_results":
            unknown_results,
    }


def find_case(
    result,
    case_id,
):
    for item in result[
        "answerable_results"
    ]:
        if item["id"] == case_id:
            return item

    return None


def print_summary(
    fixed_result,
    paragraph_result,
):
    print()
    print("=" * 80)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 80)

    print()

    print("FIXED-SIZE CHUNKING")
    print(
        f"Hit@1: "
        f"{fixed_result['metrics']['hit_at_1']:.3f}"
    )
    print(
        f"Hit@3: "
        f"{fixed_result['metrics']['hit_at_3']:.3f}"
    )
    print(
        f"MRR: "
        f"{fixed_result['metrics']['mrr']:.3f}"
    )
    print(
        f"Recall@1: "
        f"{fixed_result['metrics']['recall_at_1']:.3f}"
    )
    print(
        f"Recall@3: "
        f"{fixed_result['metrics']['recall_at_3']:.3f}"
    )

    print()

    print("PARAGRAPH-AWARE CHUNKING")
    print(
        f"Hit@1: "
        f"{paragraph_result['metrics']['hit_at_1']:.3f}"
    )
    print(
        f"Hit@3: "
        f"{paragraph_result['metrics']['hit_at_3']:.3f}"
    )
    print(
        f"MRR: "
        f"{paragraph_result['metrics']['mrr']:.3f}"
    )
    print(
        f"Recall@1: "
        f"{paragraph_result['metrics']['recall_at_1']:.3f}"
    )
    print(
        f"Recall@3: "
        f"{paragraph_result['metrics']['recall_at_3']:.3f}"
    )

    fixed_retry = find_case(
        fixed_result,
        "retry_004",
    )

    paragraph_retry = find_case(
        paragraph_result,
        "retry_004",
    )

    print()
    print("=" * 80)
    print("retry_004 COMPARISON")
    print("=" * 80)

    if fixed_retry:
        print()
        print("Fixed-size:")
        print(
            "Ranking:",
            fixed_retry[
                "ranked_documents"
            ][:3],
        )
        print(
            "Best expected score:",
            fixed_retry[
                "best_expected_score"
            ],
        )

    if paragraph_retry:
        print()
        print("Paragraph-aware:")
        print(
            "Ranking:",
            paragraph_retry[
                "ranked_documents"
            ][:3],
        )
        print(
            "Best expected score:",
            paragraph_retry[
                "best_expected_score"
            ],
        )


def main():
    dataset = (
        load_evaluation_dataset()
    )

    print(
        "Building fixed-size retriever..."
    )

    fixed_retriever = (
        build_fixed_size_retriever()
    )

    print(
        "Building paragraph-aware retriever..."
    )

    paragraph_retriever = (
        build_paragraph_retriever()
    )

    print(
        "Evaluating fixed-size chunking..."
    )

    fixed_result = (
        evaluate_strategy(
            "fixed_size",
            fixed_retriever,
            dataset,
        )
    )

    print(
        "Evaluating paragraph-aware chunking..."
    )

    paragraph_result = (
        evaluate_strategy(
            "paragraph_aware",
            paragraph_retriever,
            dataset,
        )
    )

    result = {
        "fixed_size":
            fixed_result,

        "paragraph_aware":
            paragraph_result,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print_summary(
        fixed_result,
        paragraph_result,
    )

    print()
    print(
        f"Result saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()