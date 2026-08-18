from sentence_transformers import CrossEncoder

from app.rag.models import RetrievalResult


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/"
            "ms-marco-MiniLM-L6-v2"
        ),
    ):
        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        if not candidates:
            return []

        query_chunk_pairs = [
            (
                query,
                result.chunk.content,
            )
            for result in candidates
        ]

        scores = self.model.predict(
            query_chunk_pairs
        )

        reranked_results = [
            RetrievalResult(
                chunk=result.chunk,
                score=float(score),
            )
            for result, score in zip(
                candidates,
                scores,
            )
        ]

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked_results[:top_k]