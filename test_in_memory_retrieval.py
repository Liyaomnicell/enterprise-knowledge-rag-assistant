from app.rag.document_loader import (
    load_documents,
)

from app.rag.chunker import (
    chunk_documents,
)

from app.rag.embedding import (
    EmbeddingService,
)

from app.rag.in_memory_vector_store import (
    InMemoryVectorStore,
)

from app.rag.indexing import (
    IndexingService,
)

from app.rag.retriever import (
    SemanticRetriever,
)


def main():
    # 1. Load documents
    documents = load_documents(
        "data/documents"
    )

    # 2. Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    # 3. Create shared embedding service
    embedding_service = (
        EmbeddingService()
    )

    # 4. Create vector store
    vector_store = (
        InMemoryVectorStore()
    )

    # 5. Build the index
    indexing_service = (
        IndexingService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    indexing_service.index(
        chunks
    )

    # 6. Create retriever
    retriever = (
        SemanticRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    # 7. Run a test query
    question = (
        "What are common risks "
        "of application caching?"
    )

    results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    # 8. Print results
    print()
    print("=" * 80)
    print("IN-MEMORY VECTOR STORE TEST")
    print("=" * 80)

    print()
    print(f"Question: {question}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("-" * 80)
        print(f"Rank: {rank}")
        print(
            f"Document: "
            f"{result.chunk.document_name}"
        )
        print(
            f"Chunk ID: "
            f"{result.chunk.chunk_id}"
        )
        print(
            f"Score: "
            f"{result.score:.6f}"
        )
        print()
        print(result.chunk.content)


if __name__ == "__main__":
    main()