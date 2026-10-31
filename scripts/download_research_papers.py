from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "research_papers"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PAPERS = {
    "attention_is_all_you_need.pdf":
        "https://arxiv.org/pdf/1706.03762",
    "retrieval_augmented_generation.pdf":
        "https://arxiv.org/pdf/2005.11401",
    "rag_survey.pdf":
        "https://arxiv.org/pdf/2312.10997",
}

for filename, url in PAPERS.items():
    destination = DATA_DIR / filename

    if destination.exists() and destination.stat().st_size > 1000:
        print(f"Already downloaded: {filename}")
        continue

    try:
        print(f"Downloading {filename} ...")
        request = Request(url, headers={"User-Agent": "EnterpriseRAGResearch/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()

        if not content.startswith(b"%PDF"):
            raise ValueError("The server response is not a valid PDF.")

        destination.write_bytes(content)
        print(f"Saved: {destination} ({len(content):,} bytes)")
    except Exception as exc:
        print(f"FAILED: {filename}: {exc}")

print(f"\nPaper folder: {DATA_DIR}")
