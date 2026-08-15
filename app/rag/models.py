from dataclasses import dataclass, field

@dataclass
class RawDocument:
    document_id: str
    document_name: str
    content: str
    metadata: dict[
        str,
        str | int | float | bool,
    ] = field(
        default_factory=dict
    )


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    chunk_index: int
    metadata: dict[
        str,
        str | int | float | bool,
    ] = field(
        default_factory=dict
    )


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
