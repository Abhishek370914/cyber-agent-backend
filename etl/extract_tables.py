"""
etl/extract_tables.py
─────────────────────
Step 2 of the ETL pipeline:
  1. Try Camelot (lattice + stream) to extract tables from the PDF
  2. Fall back to pdfplumber on pages where Camelot finds nothing or errors
  3. Convert each table to a Markdown-formatted string for embedding
  4. Save results to data/processed/tables.json

Output schema per record:
  {
    "page": int,
    "table_index": int,       # table number on that page (0-indexed)
    "text": str,              # Markdown table text
    "source": "Cyber Ireland 2022",
    "type": "table"
  }

Usage:
    python etl/extract_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PDF_PATH, PROCESSED_DIR, DOCUMENT_SOURCE

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_markdown(df) -> str:
    """Convert a pandas DataFrame to a compact Markdown table string."""
    import pandas as pd  # noqa: F401 – ensure pandas available

    # Clean column headers
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("").astype(str)

    lines = []
    header = " | ".join(df.columns)
    separator = " | ".join(["---"] * len(df.columns))
    lines.append(header)
    lines.append(separator)
    for _, row in df.iterrows():
        lines.append(" | ".join(row.values))
    return "\n".join(lines)


def _extract_with_camelot(pdf_path: Path, page_num: int) -> list[str]:
    """
    Attempt Camelot extraction for a single page.
    Returns a list of Markdown table strings (may be empty).
    """
    try:
        import camelot  # type: ignore
    except ImportError:
        return []

    tables_md: list[str] = []
    for flavor in ("lattice", "stream"):
        try:
            tbls = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_num),
                flavor=flavor,
                suppress_stdout=True,
            )
            for tbl in tbls:
                if tbl.df is not None and not tbl.df.empty:
                    tables_md.append(_df_to_markdown(tbl.df))
            if tables_md:
                break  # stop trying flavors once we have results
        except Exception:
            continue
    return tables_md


def _extract_with_pdfplumber(pdf_path: Path, page_num: int) -> list[str]:
    """
    Fallback: use pdfplumber for table extraction on a single page.
    Returns a list of Markdown table strings.
    """
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return []

    tables_md: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        # pdfplumber pages are 0-indexed
        page = pdf.pages[page_num - 1]
        for table in page.extract_tables():
            if not table:
                continue
            import pandas as pd  # noqa: F811

            df = pd.DataFrame(table[1:], columns=table[0])
            tables_md.append(_df_to_markdown(df))
    return tables_md


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction logic
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables(pdf_path: Path) -> list[dict]:
    """
    Extract all tables from the PDF.

    Strategy (per page):
      1. Try Camelot (lattice → stream)
      2. Fall back to pdfplumber
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    records: list[dict] = []
    print(f"[tables] Scanning {total_pages} pages for tables …")

    for page_num in tqdm(range(1, total_pages + 1), desc="Extracting tables"):
        tables_md = _extract_with_camelot(pdf_path, page_num)
        if not tables_md:
            tables_md = _extract_with_pdfplumber(pdf_path, page_num)

        for idx, md_text in enumerate(tables_md):
            records.append(
                {
                    "page": page_num,
                    "table_index": idx,
                    "text": md_text,
                    "source": DOCUMENT_SOURCE,
                    "type": "table",
                }
            )

    print(f"[tables] Found {len(records)} tables across all pages.")
    return records


def save_tables(tables: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "tables.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"[tables] Tables saved → {dest}")
    return dest


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}. Run etl/ingest_pdf.py first."
        )
    tables = extract_tables(PDF_PATH)
    save_tables(tables, PROCESSED_DIR)


if __name__ == "__main__":
    main()
