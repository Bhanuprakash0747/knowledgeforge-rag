from app.rag.context_builder import ContextBuilder
from app.rag.generation_service import GenerationService
from app.rag.models import RAGResponse, RetrievedChunk
from app.rag.prompt_template import build_rag_prompt
from app.rag.hybrid_retriever import HybridRetriever


class RAGPipeline:
    def __init__(self):
        # Load persisted FAISS and chunk metadata.
        # BM25 is built from the same persisted chunks.
        self.retriever = HybridRetriever()

        self.context_builder = ContextBuilder()
        self.generator = GenerationService()

    def refresh_retriever(self) -> None:
        """Reload the retriever using the latest persisted index."""
        self.retriever = HybridRetriever()

    def answer(
        self,
        question: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> RAGResponse:

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        # Hybrid RRF scores are not cosine similarity scores.
        if min_score is not None:
            results = [
                (chunk, score)
                for chunk, score in results
                if float(score) >= min_score
            ]

        if not results:
            return RAGResponse(
                question=question,
                answer=(
                    "I don't have enough information "
                    "in the provided documents."
                ),
                sources=[],
                retrieved_chunks=[],
            )

        context = self.context_builder.build_context(results)

        if not context.strip():
            return RAGResponse(
                question=question,
                answer=(
                    "I don't have enough information "
                    "in the provided documents."
                ),
                sources=[],
                retrieved_chunks=[],
            )

        prompt = build_rag_prompt(question, context)
        answer = self.generator.generate(prompt)

        retrieved_chunks = [
            RetrievedChunk(
                chunk=chunk,
                score=float(score),
            )
            for chunk, score in results
        ]

        sources = list(dict.fromkeys(
            item.chunk.source
            for item in retrieved_chunks
        ))

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
        )
