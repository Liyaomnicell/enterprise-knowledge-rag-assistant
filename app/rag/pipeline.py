from app.rag.document_loader import load_documents
from app.rag.chunker import chunk_documents
from app.rag.embedding import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.generator import AnswerGenerator
from app.rag.models import RetrievalResult


class RAGPipeline:

    def __init__(
        self,
        document_directory: str,
    ):
        documents = load_documents(
            document_directory
        )

        chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=100,
        )

        embedding_service = (
            EmbeddingService()
        )

        self.retriever = SemanticRetriever(
            chunks=chunks,
            embedding_service=embedding_service,
        )

        self.generator = AnswerGenerator()

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> tuple[str, list[RetrievalResult]]:

        retrieval_results = (
            self.retriever.retrieve(
                query=question,
                top_k=top_k,
            )
        )

        answer = self.generator.generate(
            question=question,
            retrieval_results=retrieval_results,
        )

        return answer, retrieval_results
