"""
etl/chunking.py
───────────────
Step 3 of the ETL pipeline:
  1. Load pages.json and tables.json from data/processed/
  2. Apply RecursiveCharacterTextSplitter to page text
  3. Infer section headings from text heuristics
  4. Merge table records (they are not split — kept whole)
  5. Attach metadata: {page, section, source, type}
  6. Save to data/processed/chunks.json

Output schema per chunk:
  {
    "id": str,             # unique chunk identifier
    "text": str,
    "page": int,
    "section": str,
    "source": "Cyber Ireland 2022",
    "type": "text" | "table"
  }

Usage:
    python etl/chunking.py
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, PROCESSED_DIR, DOCUMENT_SOURCE

try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    raise ImportError("langchain required: pip install langchain langchain-text-splitters")


# ─────────────────────────────────────────────────────────────────────────────
# Section inference
# ─────────────────────────────────────────────────────────────────────────────

# Common heading patterns found in the Cyber Ireland 2022 report
_HEADING_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*\d+\.\s+[A-Z][^.]{5,60}\s*$", re.M),       # "1. Introduction"
    re.compile(r"^\s*[A-Z][A-Z ]{4,40}\s*$", re.M),              # "EXECUTIVE SUMMARY"
    re.compile(r"^\s*(Chapter|Section|Appendix)\s+\w", re.M | re.I),
]

_KNOWN_SECTIONS: list[str] = [
    "Executive Summary",
    "Introduction",
    "Methodology",
    "National Overview",
    "Regional Analysis",
    "Pure-Play Companies",
    "Employment",
    "Revenue",
    "Forecast",
    "South-West",
    "Conclusion",
    "Appendix",
]


def _infer_section(text: str, page: int) -> str:
    """
    Heuristic: try to find a heading in the text chunk.
    Falls back to 'Page <n>'.
    """
    # Check for known section keywords (case-insensitive)
    text_lower = text.lower()
    for section in _KNOWN_SECTIONS:
        if section.lower() in text_lower:
            return section

    # Try regex heading patterns
    for pattern in _HEADING_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group().strip()[:80]

    return f"Page {page}"


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page text into overlapping semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for page_rec in pages:
        page_num = page_rec["page"]
        page_text = page_rec["text"]

        splits = splitter.split_text(page_text)
        for split in splits:
            split = split.strip()
            if len(split) < 40:          # skip trivially short chunks
                continue
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": split,
                    "page": page_num,
                    "section": _infer_section(split, page_num),
                    "source": DOCUMENT_SOURCE,
                    "type": "text",
                }
            )
    return chunks


def include_tables(tables: list[dict]) -> list[dict]:
    """
    Table records are kept whole (not re-split) because splitting
    a Markdown table would destroy its row/column structure.
    """
    table_chunks: list[dict] = []
    for tbl in tables:
        text = tbl.get("text", "").strip()
        if not text:
            continue
        table_chunks.append(
            {
                "id": str(uuid.uuid4()),
                "text": text,
                "page": tbl["page"],
                "section": _infer_section(text, tbl["page"]),
                "source": DOCUMENT_SOURCE,
                "type": "table",
            }
        )
    return table_chunks


# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "chunks.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[chunking] {len(chunks)} chunks saved → {dest}")
    return dest


def main() -> None:
    pages_path = PROCESSED_DIR / "pages.json"
    tables_path = PROCESSED_DIR / "tables.json"

    if not pages_path.exists():
        raise FileNotFoundError(
            f"pages.json not found at {pages_path}. Run etl/ingest_pdf.py first."
        )

    pages = load_json(pages_path)
    print(f"[chunking] Loaded {len(pages)} pages.")

    text_chunks = chunk_pages(pages)
    print(f"[chunking] Created {len(text_chunks)} text chunks.")

    table_chunks: list[dict] = []
    if tables_path.exists():
        tables = load_json(tables_path)
        table_chunks = include_tables(tables)
        print(f"[chunking] Included {len(table_chunks)} table chunks.")
    else:
        print("[chunking] tables.json not found — proceeding without table chunks.")

    all_chunks = text_chunks + table_chunks
    save_chunks(all_chunks, PROCESSED_DIR)


if __name__ == "__main__":
    main()
