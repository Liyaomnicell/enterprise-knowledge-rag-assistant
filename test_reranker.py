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
from app.rag.reranker import CrossEncoderReranker
from app.rag.rrf import ReciprocalRankFusion


def main():

    # -------------------------------------------------
    # Build corpus
    # -------------------------------------------------
    documents = load_documents(
        "data/documents"
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    # -------------------------------------------------
    # Build dense retriever
    # -------------------------------------------------
    embedding_service = EmbeddingService()

    vector_store = InMemoryVectorStore()

    indexing_service = IndexingService(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    indexing_service.index(
        chunks
    )

    dense_retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    # -------------------------------------------------
    # Build BM25 retriever
    # -------------------------------------------------
    bm25_retriever = BM25Retriever(
        chunks=chunks,
    )

    # -------------------------------------------------
    # Build Hybrid Retriever
    # -------------------------------------------------
    rrf = ReciprocalRankFusion(
        k=60,
    )

    hybrid_retriever = HybridRetriever(
        retrievers=[
            dense_retriever,
            bm25_retriever,
        ],
        fusion_strategy=rrf,
        candidate_k=10,
    )

    # -------------------------------------------------
    # Build Cross-Encoder Reranker
    # -------------------------------------------------
    reranker = CrossEncoderReranker()

    # api_004
    question = (
        "Requests are occasionally taking much "
        "longer than expected. What areas should "
        "engineers inspect to identify the bottleneck?"
    )

    # -------------------------------------------------
    # Hybrid candidate retrieval
    # -------------------------------------------------
    candidates = hybrid_retriever.retrieve(
        query=question,
        top_k=10,
    )

    print()
    print("=" * 80)
    print("HYBRID CANDIDATES BEFORE RERANKING")
    print("=" * 80)

    for rank, result in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{rank:2d} "
            f"{result.chunk.chunk_id:30s} "
            f"{result.score:.6f}"
        )

    # -------------------------------------------------
    # Reranking
    # -------------------------------------------------
    reranked_results = reranker.rerank(
        query=question,
        candidates=candidates,
        top_k=5,
    )

    print()
    print("=" * 80)
    print("RESULTS AFTER CROSS-ENCODER RERANKING")
    print("=" * 80)

    for rank, result in enumerate(
        reranked_results,
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
            f"Cross-Encoder Score: "
            f"{result.score:.6f}"
        )

        print()

        print(
            result.chunk.content
        )


if __name__ == "__main__":
    main()
