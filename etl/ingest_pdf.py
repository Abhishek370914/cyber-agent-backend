"""
etl/ingest_pdf.py
─────────────────
Step 1 of the ETL pipeline:
  1. Download the Cyber Ireland 2022 PDF (skips if already present)
  2. Extract text page-by-page using PyMuPDF (fitz)
  3. Save per-page records to data/processed/pages.json

Output schema per record:
  {
    "page": int,               # 1-indexed
    "text": str,               # cleaned page text
    "source": "Cyber Ireland 2022"
  }

Usage:
    python etl/ingest_pdf.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ── Allow running from project root ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    PDF_PATH,
    PDF_URL,
    PROCESSED_DIR,
    RAW_PDF_DIR,
    DOCUMENT_SOURCE,
)

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is not installed. Run: pip install pymupdf")


# ─────────────────────────────────────────────────────────────────────────────
def download_pdf(url: str, dest: Path) -> None:
    """Download PDF from *url* to *dest*, showing a progress bar.
    
    If the URL is unavailable (404 / network error), prints a clear manual
    placement instruction instead of crashing the entire pipeline.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[ingest] PDF already exists at {dest} — skipping download.")
        return

    print(f"[ingest] Attempting to download PDF from {url} …")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="Downloading"
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"[ingest] Saved PDF → {dest}")
    except requests.exceptions.HTTPError as e:
        print(f"\n[ingest] ⚠️  Download failed: {e}")
        print(f"[ingest] The PDF is no longer hosted at the original URL.")
        print(f"[ingest] Please manually download the report and place it at:")
        print(f"[ingest]   {dest}")
        print(f"[ingest] You can find the report at:")
        print(f"[ingest]   https://cyberireland.ie/publications/")
        print(f"[ingest] Then re-run:  python etl/ingest_pdf.py")
        sys.exit(1)


def _clean_text(raw: str) -> str:
    """Remove excessive whitespace and control characters."""
    # Collapse multiple newlines / spaces
    text = re.sub(r"\r\n", "\n", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)  # keep printable ASCII + newline
    return text.strip()


def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from every page of the PDF.

    Returns a list of page records:
        [{"page": 1, "text": "...", "source": "Cyber Ireland 2022"}, ...]
    """
    print(f"[ingest] Opening PDF: {pdf_path}")
    pages: list[dict] = []

    doc = fitz.open(str(pdf_path))
    for page_num, page in enumerate(tqdm(doc, desc="Extracting pages"), start=1):
        raw_text = page.get_text("text")
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue  # skip blank pages
        pages.append(
            {
                "page": page_num,
                "text": cleaned,
                "source": DOCUMENT_SOURCE,
            }
        )

    doc.close()
    print(f"[ingest] Extracted {len(pages)} non-blank pages.")
    return pages


def save_pages(pages: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "pages.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Pages saved → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Download
    download_pdf(PDF_URL, PDF_PATH)

    # 2. Extract text
    pages = extract_pages(PDF_PATH)

    # 3. Save
    save_pages(pages, PROCESSED_DIR)


if __name__ == "__main__":
    main()
