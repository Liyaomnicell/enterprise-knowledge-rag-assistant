from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_document_by_paragraph


documents = load_documents("data/documents")

for document in documents:

    chunks = chunk_document_by_paragraph(
        document,
        max_chunk_size=500,
    )

    print()
    print("#" * 80)
    print(document.document_name)

    for chunk in chunks:
        print("-" * 80)
        print(chunk.content)