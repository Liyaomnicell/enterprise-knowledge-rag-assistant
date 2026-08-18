from app.rag.models import RetrievalResult


class HybridRetriever:

    def __init__(
        self,
        retrievers,
        fusion_strategy,
        candidate_k: int = 10,
    ):
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        result_lists = []

        for retriever in self.retrievers:

            results = retriever.retrieve(
                query=query,
                top_k=self.candidate_k,
            )

            result_lists.append(results)

        return self.fusion_strategy.fuse(
            result_lists=result_lists,
            top_k=top_k,
        )

