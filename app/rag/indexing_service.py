
import os
from pathlib import Path

from app.rag.chunker import chunk_document
from app.rag.document_loader import load_text_documents
from app.rag.embedding_service import EmbeddingService
from app.rag.faiss_vector_store import FaissVectorStore
from app.rag.vector_store import InMemoryVectorStore


RESEARCH_DIRECTORY = Path("data/research_papers")
UPLOAD_DIRECTORY = Path("data/uploads")
INDEX_DIRECTORY = Path("data/index")

VECTOR_STORE_PATH = INDEX_DIRECTORY / "vector_store.json"
FAISS_INDEX_PATH = INDEX_DIRECTORY / "faiss.index"


def rebuild_index() -> dict[str, int]:
    """Rebuild the combined index from research papers and uploaded files."""
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    INDEX_DIRECTORY.mkdir(parents=True, exist_ok=True)

    documents = []

    for directory in (RESEARCH_DIRECTORY, UPLOAD_DIRECTORY):
        if directory.exists():
            documents.extend(load_text_documents(str(directory)))

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    if not chunks:
        raise ValueError("No document chunks found to index.")

    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_texts(
        [chunk.content for chunk in chunks]
    )

    vector_store = InMemoryVectorStore()
    vector_store.add_many(chunks, embeddings)

    faiss_store = FaissVectorStore()
    faiss_store.build(vector_store)

    # Prepare both files before replacing the active index files.
    vector_temp = INDEX_DIRECTORY / "vector_store.json.tmp"
    faiss_temp = INDEX_DIRECTORY / "faiss.index.tmp"

    try:
        vector_store.save(str(vector_temp))
        faiss_store.save(str(faiss_temp))

        os.replace(vector_temp, VECTOR_STORE_PATH)
        os.replace(faiss_temp, FAISS_INDEX_PATH)
    finally:
        vector_temp.unlink(missing_ok=True)
        faiss_temp.unlink(missing_ok=True)

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "vectors": vector_store.count(),
    }
