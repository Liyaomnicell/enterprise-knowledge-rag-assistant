from abc import ABC, abstractmethod

from app.rag.models import RetrievalResult


class RankFusionStrategy(ABC):

    @abstractmethod
    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        pass