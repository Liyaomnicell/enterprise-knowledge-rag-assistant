import json
from pathlib import Path

from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.evidence_checker import EvidenceSufficiencyChecker


EVALUATION_FILE = Path(
    "data/evaluation/retrieval_eval.json"
)

CASE_ID = "retry_004"


def load_case(case_id: str):
    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        dataset = json.load(file)

    for item in dataset:
        if item["id"] == case_id:
            return item

    raise ValueError(
        f"Evaluation case not found: {case_id}"
    )


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

    return SemanticRetriever(
        chunks=chunks,
        embedding_service=embedding_service,
    )


def debug_case():
    case = load_case(CASE_ID)

    retriever = build_retriever()

    checker = EvidenceSufficiencyChecker()

    question = case["question"]

    retrieval_results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    print()
    print("=" * 80)
    print("EVIDENCE SUFFICIENCY DEBUG")
    print("=" * 80)

    print()
    print(f"Case ID: {case['id']}")
    print(f"Category: {case['category']}")
    print(f"Question: {question}")
    print(
        f"Actual answerable: "
        f"{case['answerable']}"
    )

    print(
        f"Expected documents: "
        f"{case['expected_documents']}"
    )

    print()
    print("=" * 80)
    print("TOP-3 RETRIEVED CHUNKS")
    print("=" * 80)

    for index, result in enumerate(
        retrieval_results,
        start=1,
    ):
        print()
        print("-" * 80)

        print(f"Rank: {index}")

        print(
            f"Document: "
            f"{result.chunk.document_name}"
        )

        print(
            f"Chunk ID: "
            f"{result.chunk.chunk_id}"
        )

        print(
            f"Similarity: "
            f"{result.score:.6f}"
        )

        print()
        print("CONTENT:")
        print()

        print(
            result.chunk.content
        )

    print()
    print("=" * 80)
    print("CHECKER INPUT CONTEXT")
    print("=" * 80)

    context = checker._build_context(
        retrieval_results
    )

    print()
    print(context)

    print()
    print("=" * 80)
    print("CHECKER RESULT")
    print("=" * 80)

    result = checker.check(
        question=question,
        retrieval_results=retrieval_results,
    )

    print()
    print(
        f"Prediction: {result.value}"
    )


if __name__ == "__main__":
    debug_case()