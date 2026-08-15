import faiss
import numpy as np

from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService


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

    embeddings = embedding_service.embed_texts(
        [chunk.content for chunk in chunks]
    )

    vectors = np.array(
        embeddings,
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    index = faiss.IndexHNSWFlat(
        384,
        32,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.hnsw.efSearch = 64

    index.add(vectors)

    question = (
        "What are common risks "
        "of application caching?"
    )

    query_embedding = (
        embedding_service.embed_text(
            question
        )
    )

    query_vector = np.array(
        [query_embedding],
        dtype="float32",
    )

    faiss.normalize_L2(
        query_vector
    )

    scores, indices = index.search(
        query_vector,
        3,
    )

    print()
    print("=" * 80)
    print("FAISS HNSW TEST")
    print("=" * 80)

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        chunk = chunks[index_id]

        print()
        print(f"Rank: {rank}")
        print(
            f"Document: "
            f"{chunk.document_name}"
        )
        print(
            f"Chunk ID: "
            f"{chunk.chunk_id}"
        )
        print(
            f"Score: "
            f"{float(score):.6f}"
        )


if __name__ == "__main__":
    main()