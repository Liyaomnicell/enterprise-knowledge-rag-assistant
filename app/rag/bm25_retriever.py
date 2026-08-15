from rank_bm25 import BM25Okapi

from app.rag.models import (
    DocumentChunk,
    RetrievalResult,
)

import re

class BM25Retriever:

    def __init__(
        self,
        chunks: list[DocumentChunk],
    ):
        self.chunks = chunks

        self.tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        tokenized_query = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        results = [
            RetrievalResult(
                chunk=chunk,
                score=float(score),
            )
            for chunk, score in zip(
                self.chunks,
                scores,
            )
        ]

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )