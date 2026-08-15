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


INDEX_PATH = "data/index/faiss.index"
CHUNKS_PATH = "data/index/chunks.json"


def build_and_save_index():
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
        index_path=INDEX_PATH,
        chunks_path=CHUNKS_PATH,
    )

    print("Index saved.")


def load_and_search():
    embedding_service = EmbeddingService()
    
    loaded_store = FaissVectorStore(
        dimension=1,
        index_type="flat",
    )

    loaded_store.load(
        index_path=INDEX_PATH,
        chunks_path=CHUNKS_PATH,
    )
    
    print()
    print("LOADED METADATA")
    print("=" * 80)

    for chunk in loaded_store.chunks[:5]:
        print(
            chunk.chunk_id,
            chunk.metadata,
        )

    print(
        f"Loaded index type: "
        f"{loaded_store.index_type}"
    )

    print(
        f"Loaded dimension: "
        f"{loaded_store.dimension}"
    )

    print(
        f"FAISS class: "
        f"{type(loaded_store.index).__name__}"
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=loaded_store,
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
    print("FAISS PERSISTENCE TEST")
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


def main():
    build_and_save_index()
    load_and_search()


if __name__ == "__main__":
    main()