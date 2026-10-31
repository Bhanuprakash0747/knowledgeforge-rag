import unittest

from app.rag.hybrid_retriever import HybridRetriever


class TestHybridRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the existing persisted research-paper index once.
        cls.retriever = HybridRetriever(
            vector_store_path="data/index/vector_store.json",
            faiss_index_path="data/index/faiss.index",
        )

    def test_retrieves_relevant_research_chunks(self):
        question = "What is retrieval-augmented generation (RAG)?"

        results = self.retriever.retrieve(question, top_k=3)

        self.assertTrue(results, "Expected relevant chunks from the research papers")

        for chunk, score in results:
            self.assertTrue(chunk.content.strip())
            self.assertIsInstance(score, float)

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

    def test_non_positive_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve(
                "What is attention in Transformers?",
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
