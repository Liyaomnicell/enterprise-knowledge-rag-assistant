from abc import ABC, abstractmethod

from app.rag.models import (
    DocumentChunk,
    RetrievalResult,
)


class VectorStore(ABC):

    @abstractmethod
    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievalResult]:
        pass