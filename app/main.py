from fastapi import FastAPI

from app.models.schemas import (
    AskRequest,
    AskResponse,
    SourceDocument,
)


app = FastAPI(
    title="Enterprise Knowledge RAG Assistant",
    version="0.1.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    return AskResponse(
        answer=f"You asked: {request.question}",
        sources=[
            SourceDocument(
                document_name="placeholder.md",
                content="Document retrieval will be implemented in the next step."
            )
        ]
    )