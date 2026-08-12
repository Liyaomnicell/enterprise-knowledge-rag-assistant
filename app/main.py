from fastapi import FastAPI

from app.models.schemas import (
    AskRequest,
    AskResponse,
    SourceDocument,
)

from app.rag.pipeline import RAGPipeline


app = FastAPI(
    title="Enterprise Knowledge RAG Assistant",
    version="0.2.0",
)


rag_pipeline = RAGPipeline(
    document_directory="data/documents"
)


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
def ask_question(
    request: AskRequest,
):

    answer, retrieval_results = (
        rag_pipeline.ask(
            question=request.question,
            top_k=3,
        )
    )

    sources = [
        SourceDocument(
            document_name=(
                result.chunk.document_name
            ),
            content=(
                result.chunk.content
            ),
        )
        for result in retrieval_results
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
    )