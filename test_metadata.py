from app.rag.document_loader import (
    load_documents,
)

from app.rag.chunker import (
    chunk_documents,
)


def main():
    documents = load_documents(
        "data/documents"
    )

    print("RAW DOCUMENT METADATA")
    print("=" * 80)

    for document in documents:
        print(
            document.document_name,
            document.metadata,
        )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    print()
    print("CHUNK METADATA")
    print("=" * 80)

    for chunk in chunks[:5]:
        print(
            chunk.chunk_id,
            chunk.document_name,
            chunk.metadata,
        )


if __name__ == "__main__":
    main()