from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents


documents = load_documents("data/documents")

print(f"Loaded documents: {len(documents)}")

chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
)

print(f"Generated chunks: {len(chunks)}")

for chunk in chunks:
    print("=" * 80)
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Document: {chunk.document_name}")
    print(f"Index: {chunk.chunk_index}")
    print()
    print(chunk.content)
