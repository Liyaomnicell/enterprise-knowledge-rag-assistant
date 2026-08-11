from app.rag.models import DocumentChunk, RawDocument


def chunk_document(
    document: RawDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[DocumentChunk] = []

    text = document.content

    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk = DocumentChunk(
                chunk_id=f"{document.document_id}-{chunk_index}",
                document_id=document.document_id,
                document_name=document.document_name,
                content=chunk_text,
                chunk_index=chunk_index,
            )

            chunks.append(chunk)

        start += chunk_size - chunk_overlap
        chunk_index += 1

    return chunks


def chunk_document_by_paragraph(
    document: RawDocument,
    max_chunk_size: int = 500,
) -> list[DocumentChunk]:

    paragraphs = [
        paragraph.strip()
        for paragraph in document.content.split("\n\n")
        if paragraph.strip()
    ]

    chunks: list[DocumentChunk] = []

    current_chunk = ""
    chunk_index = 0

    for paragraph in paragraphs:

        candidate = (
            f"{current_chunk}\n\n{paragraph}".strip()
            if current_chunk
            else paragraph
        )

        if len(candidate) <= max_chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}-{chunk_index}",
                    document_id=document.document_id,
                    document_name=document.document_name,
                    content=current_chunk,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

        current_chunk = paragraph

    if current_chunk:
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}-{chunk_index}",
                document_id=document.document_id,
                document_name=document.document_name,
                content=current_chunk,
                chunk_index=chunk_index,
            )
        )

    return chunks


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:

    all_chunks: list[DocumentChunk] = []

    for document in documents:
        document_chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(document_chunks)

    return all_chunks