from app.rag.embedding import (
    EmbeddingService,
    cosine_similarity,
)


embedding_service = EmbeddingService()

texts = [
    "Database downgrade may cause data loss.",
    "Dropping database columns can destroy information.",
    "The weather is sunny today.",
]

embeddings = embedding_service.embed_texts(texts)

similarity_01 = cosine_similarity(
    embeddings[0],
    embeddings[1],
)

similarity_02 = cosine_similarity(
    embeddings[0],
    embeddings[2],
)

print(
    "Database downgrade vs database data loss:",
    similarity_01,
)

print(
    "Database downgrade vs weather:",
    similarity_02,
)
