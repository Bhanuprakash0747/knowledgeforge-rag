
from pathlib import Path
import shutil
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.indexing_service import rebuild_index
from app.rag.rag_pipeline import RAGPipeline
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Hybrid RAG API with semantic and keyword retrieval",
    version="1.1.0",
)

pipeline = RAGPipeline()
index_lock = Lock()

UPLOAD_DIRECTORY = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=3, ge=1, le=10)


class EvidenceResponse(BaseModel):
    source: str
    page_number: int | None = None
    chunk_index: int
    score: float
    content: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    evidence: list[EvidenceResponse]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        with index_lock:
            response = pipeline.answer(
                question=request.question,
                top_k=request.top_k,
            )

        evidence = [
            EvidenceResponse(
                source=item.chunk.source,
                page_number=item.chunk.page_number,
                chunk_index=item.chunk.chunk_index,
                score=item.score,
                content=item.chunk.content,
            )
            for item in response.retrieved_chunks
        ]

        return AskResponse(
            question=response.question,
            answer=response.answer,
            sources=response.sources,
            evidence=evidence,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # A unique stored name prevents path traversal and filename collisions.
    stored_name = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIRECTORY / stored_name
    temporary = UPLOAD_DIRECTORY / f"{stored_name}.uploading"

    size = 0

    try:
        with index_lock:
            with temporary.open("wb") as output:
                while True:
                    block = file.file.read(1024 * 1024)
                    if not block:
                        break

                    size += len(block)
                    if size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="Maximum upload size is 10 MB.",
                        )

                    output.write(block)

            if size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded file is empty.",
                )

            temporary.replace(destination)

            try:
                stats = rebuild_index()
                pipeline.refresh_retriever()
            except Exception as exc:
                logger.exception("Document indexing failed")
                destination.unlink(missing_ok=True)
                # Rebuild the previous corpus so the persisted index remains
                # aligned with the files still present.
                try:
                    rebuild_index()
                    pipeline.refresh_retriever()
                except Exception:
                    pass

                raise HTTPException(
                    status_code=500,
                    detail="Document indexing failed. Check the server logs.",
                ) from exc

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "document_id": stored_name,
            "size_bytes": size,
            **stats,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the uploaded file.",
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
        file.file.close()
