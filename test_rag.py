import pytest

from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.in_memory_vector_store import InMemoryVectorStore
from app.rag.indexing import IndexingService
from app.rag.generator import AnswerGenerator


@pytest.mark.llm
def test_rag_answers_database_downgrade_question():
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

    generator = AnswerGenerator()

    question = (
        "Why can database downgrade cause data loss?"
    )

    retrieval_results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    assert retrieval_results
    assert (
        retrieval_results[0].chunk.document_name
        == "database_downgrade.md"
    )

    answer = generator.generate(
        question=question,
        retrieval_results=retrieval_results,
    )

    assert (
        "data loss"
        in answer.lower()
    )
    assert (
        "dropping"
        in answer.lower()
    )
