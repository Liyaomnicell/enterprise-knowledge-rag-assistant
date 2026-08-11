from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)


    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)

        return np.asarray(embedding)


    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts)

        return np.asarray(embeddings)


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:

    dot_product = np.dot(vector_a, vector_b)

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    return float(similarity)