
from pathlib import Path

from app.rag.models import Chunk


class ContextBuilder:
    def __init__(self, max_chars: int = 6000):
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")

        self.max_chars = max_chars

    def build_context(
        self,
        results: list[tuple[Chunk, float]],
    ) -> str:
        if not results:
            return ""

        context_parts: list[str] = []
        current_length = 0

        for rank, (chunk, score) in enumerate(results, start=1):
            source_name = Path(chunk.source).name

            if chunk.page_number is not None:
                citation = f"{source_name}, PDF page {chunk.page_number}"
            else:
                citation = source_name

            section = (
                f"[Source {rank}: {citation} | "
                f"Chunk {chunk.chunk_index}]\n"
                f"{chunk.content.strip()}"
            )

            separator = "\n\n" if context_parts else ""
            addition = separator + section
            remaining = self.max_chars - current_length

            if len(addition) > remaining:
                if remaining > len(separator) + 40:
                    context_parts.append(addition[:remaining])
                break

            context_parts.append(addition)
            current_length += len(addition)

        return "".join(context_parts)
