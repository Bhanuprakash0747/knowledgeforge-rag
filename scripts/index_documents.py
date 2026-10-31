
from pathlib import Path

from app.rag.chunker import chunk_document
from app.rag.document_loader import load_text_documents
from app.rag.embedding_service import EmbeddingService
from app.rag.vector_store import InMemoryVectorStore
from app.rag.faiss_vector_store import FaissVectorStore


DATA_DIRECTORY = "data/research_papers"
INDEX_DIRECTORY = Path("data/index")

VECTOR_STORE_PATH = INDEX_DIRECTORY / "vector_store.json"
FAISS_INDEX_PATH = INDEX_DIRECTORY / "faiss.index"


def build_index() -> InMemoryVectorStore:
    documents = load_text_documents(DATA_DIRECTORY)

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    if not chunks:
        raise ValueError(
            f"No document chunks found in {DATA_DIRECTORY}"
        )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    embedding_service = EmbeddingService()

    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = embedding_service.embed_texts(chunk_texts)

    vector_store = InMemoryVectorStore()
    vector_store.add_many(chunks, embeddings)

    print(f"Vectors stored: {vector_store.count()}")

    return vector_store


def save_index(vector_store: InMemoryVectorStore) -> None:
    INDEX_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save chunks and their embeddings.
    vector_store.save(str(VECTOR_STORE_PATH))

    # Build and save the FAISS search index.
    faiss_store = FaissVectorStore()
    faiss_store.build(vector_store)
    faiss_store.save(str(FAISS_INDEX_PATH))

    print(f"Vector metadata saved: {VECTOR_STORE_PATH}")
    print(f"FAISS index saved: {FAISS_INDEX_PATH}")
    print("Index saved successfully.")


if __name__ == "__main__":
    store = build_index()
    save_index(store)
