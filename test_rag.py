from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.generator import AnswerGenerator


documents = load_documents(
    "data/documents"
)

chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100,
)

embedding_service = EmbeddingService()

retriever = SemanticRetriever(
    chunks=chunks,
    embedding_service=embedding_service,
)

generator = AnswerGenerator()


question = (
    "Why can database downgrade cause data loss?"
)

retrieval_results = retriever.retrieve(
    query=question,
    top_k=3,
)

print()
print("QUESTION")
print(question)

print()
print("RETRIEVED CONTEXT")

for result in retrieval_results:
    print("-" * 80)

    print(
        f"{result.score:.4f} "
        f"{result.chunk.document_name}"
    )

    print(
        result.chunk.content
    )


answer = generator.generate(
    question=question,
    retrieval_results=retrieval_results,
)

print()
print("=" * 80)
print("ANSWER")
print("=" * 80)

print(answer)
