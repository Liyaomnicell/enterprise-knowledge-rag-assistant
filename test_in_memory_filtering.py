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


def main():
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

    question = (
        "What are common risks "
        "of application caching?"
    )

    query_embedding = (
        embedding_service.embed_text(
            question
        )
    )

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
        metadata_filter={
            "category": "database",
        },
    )

    print()
    print("=" * 80)
    print("IN-MEMORY METADATA FILTER TEST")
    print("=" * 80)

    print()
    print(f"Question: {question}")
    print(
        "Filter: "
        "{'category': 'database'}"
    )

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
            f"Metadata: "
            f"{result.chunk.metadata}"
        )
        print(
            f"Score: "
            f"{result.score:.6f}"
        )


if __name__ == "__main__":
    main()