import pytest

from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.in_memory_vector_store import InMemoryVectorStore
from app.rag.indexing import IndexingService
from app.rag.evidence_checker import (
    EvidenceSufficiency,
    EvidenceSufficiencyChecker,
)


@pytest.mark.llm
def test_evidence_checker_classifies_answerable_and_unanswerable_questions():
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    embedding_service = EmbeddingService()

    vector_store = InMemoryVectorStore()

    indexing_service = IndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexing_service.index(chunks)

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    checker = EvidenceSufficiencyChecker()

    cases = [
        (
            "What are common risks of application caching?",
            EvidenceSufficiency.SUFFICIENT,
        ),
        (
            (
                "What exact TTL value should be configured "
                "for the application cache?"
            ),
            EvidenceSufficiency.INSUFFICIENT,
        ),
    ]

    for question, expected in cases:
        retrieval_results = retriever.retrieve(
            query=question,
            top_k=3,
        )

        result = checker.check(
            question=question,
            retrieval_results=retrieval_results,
        )

        assert result == expected