from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.in_memory_vector_store import (
    InMemoryVectorStore,
)

from app.rag.indexing import (
    IndexingService,
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


queries = [
    "Why can database downgrade cause data loss?",
    "How should retry be implemented?",
    "Why can caching return stale data?",
    "How can I investigate an API timeout?",
    "What should be checked before a software release?",
]
for query in queries:
    results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    print()
    print(f"Query: {query}")
    print()

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)

        print(
            f"Result #{index}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Document: "
            f"{result.chunk.document_name}"
        )

        print(
            f"Chunk ID: "
            f"{result.chunk.chunk_id}"
        )

        print()

        print(
            result.chunk.content
        )