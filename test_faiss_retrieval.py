from app.rag.document_loader import (
    load_documents,
)

from app.rag.chunker import (
    chunk_documents,
)

from app.rag.embedding import (
    EmbeddingService,
)

from app.rag.faiss_vector_store import (
    FaissVectorStore,
)

from app.rag.indexing import (
    IndexingService,
)

from app.rag.retriever import (
    SemanticRetriever,
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

    vector_store = FaissVectorStore(
        dimension=384,
        index_type="hnsw",
    )

    indexing_service = IndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexing_service.index(chunks)

    vector_store.save(
        index_path="data/index/faiss.index",
        chunks_path="data/index/chunks.json",
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    question = (
        "What are common risks "
        "of application caching?"
    )

    results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    print()
    print("=" * 80)
    print("FAISS VECTOR STORE TEST")
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