import numpy as np

from app.rag.embedding import (
    EmbeddingService,
    cosine_similarity,
)
from app.rag.models import (
    DocumentChunk,
    RetrievalResult,
)


class SemanticRetriever:

    def __init__(
        self,
        chunks: list[DocumentChunk],
        embedding_service: EmbeddingService,
    ):
        self.chunks = chunks
        self.embedding_service = embedding_service

        chunk_texts = [
            chunk.content
            for chunk in chunks
        ]

        self.chunk_embeddings = (
            self.embedding_service.embed_texts(
                chunk_texts
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        query_embedding = (
            self.embedding_service.embed_text(query)
        )

        results: list[RetrievalResult] = []

        for chunk, chunk_embedding in zip(
            self.chunks,
            self.chunk_embeddings,
        ):

            score = cosine_similarity(
                query_embedding,
                chunk_embedding,
            )

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]