import re
from pathlib import Path
from pypdf import PdfReader


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    pages: list[str] = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    # Collapse whitespace runs and strip PDF artefacts without touching sentence structure.
    return re.sub(r"\s+", " ", raw).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start: int = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def load_documents(raw_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    raw_dir_path = Path(raw_dir)

    if not raw_dir_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir_path}")

    records: list[dict] = []
    for pdf_path in raw_dir_path.glob("*.pdf"):
        cleaned = clean_text(extract_text(pdf_path))
        for i, chunk in enumerate(chunk_text(cleaned, chunk_size, overlap)):
            records.append({
                "source": pdf_path.name,
                "chunk_index": i,
                "text": chunk,
            })

    return records
