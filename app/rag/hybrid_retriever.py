
from app.rag.retriever import SemanticRetriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.vector_store import InMemoryVectorStore


class HybridRetriever:
    def __init__(
        self,
        vector_store_path: str = "data/index/vector_store.json",
        faiss_index_path: str = "data/index/faiss.index",
    ):
        # Load persisted chunks and embeddings once.
        metadata_store = InMemoryVectorStore()
        metadata_store.load(vector_store_path)

        records = metadata_store.get_all()
        if not records:
            raise ValueError("The persisted vector store is empty")

        chunks = [record.chunk for record in records]

        # Both retrievers now use the persisted document chunks.
        self.semantic = SemanticRetriever(
            vector_store_path=vector_store_path,
            faiss_index_path=faiss_index_path,
        )
        self.bm25 = BM25Retriever(chunks)

    def retrieve(self, query: str, top_k: int = 3):
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidate_k = min(20, len(self.bm25.chunks))

        dense_results = self.semantic.retrieve(
            query, top_k=candidate_k
        )
        keyword_results = self.bm25.retrieve(
            query, top_k=candidate_k
        )

        fused_scores = {}
        chunks_by_key = {}

        def identity(chunk):
            return chunk.chunk_id

        def add_ranked_results(results):
            seen = set()

            for rank, (chunk, _) in enumerate(results, start=1):
                key = identity(chunk)

                if key in seen:
                    continue

                seen.add(key)
                chunks_by_key[key] = chunk
                fused_scores[key] = (
                    fused_scores.get(key, 0.0)
                    + 1.0 / (60 + rank)
                )

        add_ranked_results(dense_results)
        add_ranked_results(keyword_results)

        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )

        return [
            (chunks_by_key[key], fused_scores[key])
            for key in ranked_keys[:top_k]
        ]
