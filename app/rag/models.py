from dataclasses import dataclass


@dataclass
class RawDocument:
    document_id: str
    document_name: str
    content: str


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    chunk_index: int

@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float