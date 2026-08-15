from app.rag.embedding import (
    EmbeddingService,
)

from app.rag.models import (
    DocumentChunk,
)

from app.rag.vector_store import (
    VectorStore,
)


class IndexingService:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ):
        self.embedding_service = (
            embedding_service
        )

        self.vector_store = (
            vector_store
        )

    def index(
        self,
        chunks: list[DocumentChunk],
    ) -> None:

        chunk_texts = [
            chunk.content
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_service.embed_texts(
                chunk_texts
            )
        )

        self.vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )