import faiss
import numpy as np
import json
from pathlib import Path

from app.rag.models import (
    DocumentChunk,
    RetrievalResult,
)

from app.rag.vector_store import (
    VectorStore,
)


class FaissVectorStore(VectorStore):

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_search: int = 32,
    ):
        self.dimension = dimension
        self.index_type = index_type

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(
                dimension
            )

        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(
                dimension,
                hnsw_m,
                faiss.METRIC_INNER_PRODUCT,
            )

            self.index.hnsw.efSearch = (
                hnsw_ef_search
            )

        else:
            raise ValueError(
                f"Unsupported index type: "
                f"{index_type}"
            )

        self.chunks: list[DocumentChunk] = []


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

        if len(embeddings) == 0:
            return

        vectors = np.array(
            embeddings,
            dtype="float32",
        )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"but got {vectors.shape[1]}."
            )

        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        self.chunks.extend(chunks)


    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[
            str,
            str | int | float | bool,
        ] | None = None,
    ) -> list[RetrievalResult]:

        if self.index.ntotal == 0:
            return []

        query_vector = np.array(
            [query_embedding],
            dtype="float32",
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"but got {query_vector.shape[1]}."
            )

        faiss.normalize_L2(query_vector)

        search_k = top_k

        if metadata_filter is not None:
            search_k = self.index.ntotal

        scores, indices = self.index.search(
            query_vector,
            search_k,
        )

        results: list[RetrievalResult] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            chunk = self.chunks[index]

            if metadata_filter is not None:
                matches_filter = all(
                    chunk.metadata.get(key) == value
                    for key, value
                    in metadata_filter.items()
                )

                if not matches_filter:
                    continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                )
            )

            if len(results) == top_k:
                break

        return results


    def save(
        self,
        index_path: str,
        chunks_path: str,
    ) -> None:

        index_file = Path(index_path)
        chunks_file = Path(chunks_path)

        index_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_file),
        )

        chunks_data = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]

        with chunks_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                chunks_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load(
        self,
        index_path: str,
        chunks_path: str,
    ) -> None:

        index_file = Path(index_path)
        chunks_file = Path(chunks_path)

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: "
                f"{index_path}"
            )

        if not chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: "
                f"{chunks_path}"
            )

        self.index = faiss.read_index(
            str(index_file)
        )
        self.dimension = self.index.d

        if isinstance(
            self.index,
            faiss.IndexHNSWFlat,
        ):
            self.index_type = "hnsw"
        else:
            self.index_type = "flat"

        with chunks_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            chunks_data = json.load(file)

        self.chunks = [
            DocumentChunk(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                document_name=item["document_name"],
                content=item["content"],
                chunk_index=item["chunk_index"],
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )
            for item in chunks_data
        ]
