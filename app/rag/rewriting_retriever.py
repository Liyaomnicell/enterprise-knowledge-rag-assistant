from app.rag.models import RetrievalResult


class RewritingRetriever:

    def __init__(
        self,
        base_retriever,
        query_rewriter,
    ):
        self.base_retriever = base_retriever
        self.query_rewriter = query_rewriter

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        rewritten_query = (
            self.query_rewriter.rewrite(
                query
            )
        )

        return self.base_retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
        ) 

        