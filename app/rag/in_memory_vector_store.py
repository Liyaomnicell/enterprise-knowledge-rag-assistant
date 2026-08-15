from app.rag.embedding import (
    cosine_similarity,
)

from app.rag.models import (
    DocumentChunk,
    RetrievalResult,
)

from app.rag.vector_store import (
    VectorStore,
)


class InMemoryVectorStore(VectorStore):

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match "
                "the number of embeddings."
            )

        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[
            str,
            str | int | float | bool,
        ] | None = None,
    ) -> list[RetrievalResult]:

        results: list[RetrievalResult] = []

        for chunk, chunk_embedding in zip(
            self.chunks,
            self.embeddings,
        ):
            if metadata_filter is not None:
                matches_filter = all(
                    chunk.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )

                if not matches_filter:
                    continue

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