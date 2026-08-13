from app.rag.document_loader import (
    load_documents,
)

from app.rag.chunker import (
    chunk_documents,
    chunk_document_by_paragraph,
)

from app.rag.embedding import (
    EmbeddingService,
)

from app.rag.retriever import (
    SemanticRetriever,
)


QUESTION = (
    "A downstream service temporarily "
    "returns HTTP 503. Should the client "
    "attempt the operation again?"
)


def build_fixed():
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    return SemanticRetriever(
        chunks=chunks,
        embedding_service=
            EmbeddingService(),
    )


def build_paragraph():
    documents = load_documents(
        "data/documents"
    )

    chunks = []

    for document in documents:

        chunks.extend(
            chunk_document_by_paragraph(
                document,
                max_chunk_size=500,
            )
        )

    return SemanticRetriever(
        chunks=chunks,
        embedding_service=
            EmbeddingService(),
    )


def print_results(
    title,
    retriever,
):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    results = retriever.retrieve(
        query=QUESTION,
        top_k=5,
    )

    for index, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("-" * 80)

        print(
            f"Rank: {index}"
        )

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

        print(
            result.chunk.content
        )


def main():
    fixed = build_fixed()
    paragraph = build_paragraph()

    print_results(
        "FIXED-SIZE",
        fixed,
    )

    print_results(
        "PARAGRAPH-AWARE",
        paragraph,
    )


if __name__ == "__main__":
    main()