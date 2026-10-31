
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader

from app.rag.models import Document


def load_text_documents(directory: str) -> list[Document]:
    documents: list[Document] = []
    data_directory = Path(directory)

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {data_directory}"
        )

    # Load text files.
    for file_path in sorted(data_directory.rglob("*.txt")):
        content = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if content:
            documents.append(
                Document(
                    document_id=str(uuid4()),
                    title=file_path.stem,
                    content=content,
                    source=str(file_path),
                )
            )

    # Load PDFs one page at a time to preserve citations.
    for file_path in sorted(data_directory.rglob("*.pdf")):
        reader = PdfReader(str(file_path))

        for page_index, page in enumerate(reader.pages, start=1):
            content = (page.extract_text() or "").strip()

            if not content:
                continue

            documents.append(
                Document(
                    document_id=str(uuid4()),
                    title=file_path.stem,
                    content=content,
                    source=str(file_path),
                    page_number=page_index,
                )
            )

    return documents
