
import unittest

from app.rag.bm25_retriever import BM25Retriever
from app.rag.models import Chunk


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content=(
                    "Retrieval augmented generation uses a retriever "
                    "to find relevant passages for a language model."
                ),
                source="rag_survey.pdf",
                chunk_index=0,
                page_number=1,
            ),
            Chunk(
                chunk_id="chunk-2",
                document_id="doc-2",
                content=(
                    "The Transformer architecture uses self-attention "
                    "to process sequences."
                ),
                source="attention_is_all_you_need.pdf",
                chunk_index=0,
                page_number=1,
            ),
        ]
        self.retriever = BM25Retriever(self.chunks)

    def test_retrieves_relevant_chunks(self):
        results = self.retriever.retrieve(
            "How does retrieval augmented generation find passages?",
            top_k=2,
        )

        self.assertTrue(results)
        self.assertEqual(
            results[0][0].source,
            "rag_survey.pdf",
        )

    def test_empty_query_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve("", top_k=2)

    def test_non_positive_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.retrieve(
                "retrieval augmented generation",
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
