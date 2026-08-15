from pathlib import Path

from app.rag.models import RawDocument

DOCUMENT_METADATA = {
    "database_downgrade.md": {
        "category": "database",
        "document_type": "troubleshooting",
    },
    "api_timeout.md": {
        "category": "api",
        "document_type": "guideline",
    },
    "cache_strategy.md": {
        "category": "performance",
        "document_type": "design_guideline",
    },
    "service_retry_policy.md": {
        "category": "reliability",
        "document_type": "policy",
    },
    "release_checklist.md": {
        "category": "release",
        "document_type": "checklist",
    },
}

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
            metadata=DOCUMENT_METADATA.get(
                file_path.name,
                {},
            ).copy(),
        )

        documents.append(document)

    return documents