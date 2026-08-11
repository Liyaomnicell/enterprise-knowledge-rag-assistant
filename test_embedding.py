from app.rag.embedding import EmbeddingService


embedding_service = EmbeddingService()

texts = [
    "Database downgrade may cause data loss.",
    "Dropping database columns can destroy information.",
    "The weather is sunny today.",
]

embeddings = embedding_service.embed_texts(texts)

print("Shape:")
print(embeddings.shape)

for index, embedding in enumerate(embeddings):
    print()
    print(f"Text {index}: {texts[index]}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")