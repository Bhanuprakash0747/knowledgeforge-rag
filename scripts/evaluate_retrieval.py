
import json
import time

from app.rag.retriever import SemanticRetriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import InMemoryVectorStore


def load_evaluation_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant(chunk, item: dict) -> bool:
    relevant_chunks = item.get("relevant_chunks", [])

    # Prefer exact chunk-level matching.
    if relevant_chunks:
        return chunk.chunk_id in relevant_chunks

    # Backward compatibility with document-level evaluation.
    expected_document = item.get("expected_document")

    if expected_document:
        return expected_document.lower() in chunk.source.lower()

    raise ValueError(
        f"No relevance criteria found for question: {item['question']}"
    )


def evaluate_retriever(retriever, questions, k: int, name: str):
    hits = 0
    reciprocal_ranks = []
    latencies = []

    for item in questions:
        start = time.perf_counter()

        results = retriever.retrieve(item["question"], top_k=k)

        latencies.append(time.perf_counter() - start)

        matching_rank = next(
            (
                rank
                for rank, (chunk, _) in enumerate(results, start=1)
                if is_relevant(chunk, item)
            ),
            None,
        )

        if matching_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / matching_rank)
        else:
            reciprocal_ranks.append(0.0)

    total = len(questions)

    return {
        "retriever": name,
        "recall": hits / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "latency_ms": (
            sum(latencies) / total * 1000 if total else 0.0
        ),
    }


def print_diagnostics(retriever, questions, name: str):
    print(f"\n{'=' * 75}")
    print(f"{name.upper()} - INDIVIDUAL QUESTION DIAGNOSTICS")
    print(f"{'=' * 75}")

    for index, item in enumerate(questions, start=1):
        results = retriever.retrieve(item["question"], top_k=5)

        matching_rank = next(
            (
                rank
                for rank, (chunk, _) in enumerate(results, start=1)
                if is_relevant(chunk, item)
            ),
            None,
        )

        print(f"\nQuestion {index}: {item['question']}")

        if matching_rank:
            print(f"First relevant result rank: {matching_rank}")
        else:
            print("First relevant result rank: NOT FOUND IN TOP 5")

        for rank, (chunk, score) in enumerate(results, start=1):
            marker = " <-- RELEVANT" if is_relevant(chunk, item) else ""

            print(
                f"  Rank {rank}: "
                f"chunk_id={chunk.chunk_id} | "
                f"page={chunk.page_number} | "
                f"score={score:.4f}"
                f"{marker}"
            )


if __name__ == "__main__":
    questions = load_evaluation_questions(
        "data/evaluation/retrieval_questions.json"
    )

    metadata_store = InMemoryVectorStore()
    metadata_store.load("data/index/vector_store.json")

    chunks = [record.chunk for record in metadata_store.get_all()]

    retrievers = [
        ("Semantic", SemanticRetriever()),
        ("BM25", BM25Retriever(chunks)),
        ("Hybrid", HybridRetriever()),
    ]

    print(f"\nEvaluation questions: {len(questions)}")

    for name, retriever in retrievers:
        print(f"\n{name} retrieval")

        for k in [1, 3, 5]:
            metrics = evaluate_retriever(
                retriever,
                questions,
                k,
                name,
            )

            print(
                f"Recall@{k}: {metrics['recall']:.2%} | "
                f"MRR@{k}: {metrics['mrr']:.3f} | "
                f"Avg latency: {metrics['latency_ms']:.2f} ms"
            )

        print_diagnostics(retriever, questions, name)
