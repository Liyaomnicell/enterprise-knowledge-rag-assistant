from app.rag.bm25_retriever import BM25Retriever
from app.rag.chunker import chunk_documents
from app.rag.document_loader import load_documents
from app.rag.embedding import EmbeddingService
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.in_memory_vector_store import (
    InMemoryVectorStore,
)
from app.rag.indexing import IndexingService
from app.rag.retriever import SemanticRetriever
from app.rag.rrf import ReciprocalRankFusion


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

    dense_retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks,
    )

    rrf = ReciprocalRankFusion()

    hybrid_retriever = HybridRetriever(
        retrievers=[
            dense_retriever,
            bm25_retriever,
        ],
        fusion_strategy=rrf,
        candidate_k=10,
    )

    question = (
        "What processing approach is "
        "recommended for long-running "
        "API work?"
    )

    results = hybrid_retriever.retrieve(
        query=question,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 80)

    print()
    print(f"Question: {question}")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print("-" * 80)

        print(
            f"Rank: {rank}"
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
            f"RRF Score: "
            f"{result.score:.6f}"
        )

    dense_results = dense_retriever.retrieve(
        query=question,
        top_k=5,
    )

    bm25_results = bm25_retriever.retrieve(
        query=question,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("DENSE")
    print("=" * 80)

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        print(
            rank,
            result.chunk.chunk_id,
            f"{result.score:.6f}",
        )

    print()
    print("=" * 80)
    print("BM25")
    print("=" * 80)

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):
        print(
            rank,
            result.chunk.chunk_id,
            f"{result.score:.6f}",
        )

if __name__ == "__main__":
    main()