
from dataclasses import dataclass


@dataclass
class Document:
    document_id: str
    title: str
    content: str
    source: str
    page_number: int | None = None


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    source: str
    chunk_index: int
    page_number: int | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]
