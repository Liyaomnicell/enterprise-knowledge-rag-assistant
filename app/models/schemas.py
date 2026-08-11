from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Question to ask the enterprise knowledge base"
    )


class SourceDocument(BaseModel):
    document_name: str
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]

