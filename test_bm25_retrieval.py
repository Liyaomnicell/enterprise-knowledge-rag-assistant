from app.rag.document_loader import (
    load_documents,
)

from app.rag.chunker import (
    chunk_documents,
)

from app.rag.bm25_retriever import (
    BM25Retriever,
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

    retriever = BM25Retriever(
        chunks=chunks,
    )

    question = (
        "What should happen after "
        "a downstream service returns HTTP 503?"
    )

    results = retriever.retrieve(
        query=question,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("BM25 RETRIEVAL TEST")
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