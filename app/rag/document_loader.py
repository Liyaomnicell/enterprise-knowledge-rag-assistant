from pathlib import Path

from app.rag.models import RawDocument


def load_documents(directory: str) -> list[RawDocument]:
    documents: list[RawDocument] = []

    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory}"
        )

    for file_path in sorted(directory_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        document = RawDocument(
            document_id=file_path.stem,
            document_name=file_path.name,
            content=content,
        )

        documents.append(document)

    return documents