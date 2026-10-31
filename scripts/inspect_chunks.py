import json
from pathlib import Path

INDEX_PATH = Path("data/index/vector_store.json")
OUTPUT_PATH = Path("data/evaluation/chunk_catalog.txt")

def main():
    with INDEX_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        for record in records:
            chunk = record["chunk"]

            source = Path(chunk["source"]).name

            output.write("=" * 100 + "\n")
            output.write(f"Source: {source}\n")
            output.write(f"Chunk ID: {chunk['chunk_id']}\n")
            output.write(f"Document ID: {chunk['document_id']}\n")
            output.write(f"Chunk Index: {chunk['chunk_index']}\n")
            output.write(f"Page Number: {chunk.get('page_number')}\n")
            output.write("-" * 100 + "\n")
            output.write(chunk["content"].strip() + "\n\n")

    print(f"Total records: {len(records)}")
    print(f"Chunk catalog created: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
