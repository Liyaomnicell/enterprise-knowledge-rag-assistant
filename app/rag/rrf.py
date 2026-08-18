from app.rag.models import RetrievalResult
from app.rag.rank_fusion import RankFusionStrategy


class ReciprocalRankFusion(
    RankFusionStrategy
):

    def __init__(
        self,
        k: int = 60,
    ):
        self.k = k

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:

        fused_scores: dict[str, float] = {}
        chunks_by_id = {}

        for results in result_lists:

            for rank, result in enumerate(
                results,
                start=1,
            ):
                chunk_id = (
                    result.chunk.chunk_id
                )

                rrf_score = (
                    1.0
                    / (self.k + rank)
                )

                fused_scores[chunk_id] = (
                    fused_scores.get(
                        chunk_id,
                        0.0,
                    )
                    + rrf_score
                )

                chunks_by_id[chunk_id] = (
                    result.chunk
                )

        fused_results = [
            RetrievalResult(
                chunk=chunks_by_id[chunk_id],
                score=score,
            )
            for chunk_id, score
            in fused_scores.items()
        ]

        fused_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return fused_results[:top_k]

