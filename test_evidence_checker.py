from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.evidence_checker import (
    EvidenceSufficiencyChecker,
)


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

checker = EvidenceSufficiencyChecker()


questions = [
    (
        "What are common risks of application caching?"
    ),
    (
        "What exact TTL value should be configured "
        "for the application cache?"
    ),
]


for question in questions:

    retrieval_results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    result = checker.check(
        question=question,
        retrieval_results=retrieval_results,
    )

    print()
    print("=" * 80)

    print(
        f"Question: {question}"
    )

    print(
        f"Evidence sufficiency: "
        f"{result.value}"
    )

    print()

    print(
        "Retrieved evidence:"
    )

    for retrieval_result in retrieval_results:

        print(
            f"{retrieval_result.score:.3f} "
            f"{retrieval_result.chunk.document_name}"
        )