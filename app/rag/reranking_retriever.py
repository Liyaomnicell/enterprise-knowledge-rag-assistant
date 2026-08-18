from app.rag.models import RetrievalResult


class RerankingRetriever:

    def __init__(
        self,
        base_retriever,
        reranker,
        candidate_k: int = 10,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        candidates = (
            self.base_retriever.retrieve(
                query=query,
                top_k=self.candidate_k,
            )
        )

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )