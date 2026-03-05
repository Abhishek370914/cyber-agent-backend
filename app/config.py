"""
app/config.py
─────────────
Central configuration loaded from environment variables (.env file).
All other modules import from here instead of touching os.environ directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Locate project root and load .env ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ── Google Gemini (free tier) ────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")

# ── OpenAI (optional — not required when using free embeddings + Gemini) ─────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: Path = ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "cyber_ireland_2022")

# ── ETL paths ─────────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT / "data"
RAW_PDF_DIR: Path = DATA_DIR / "raw_pdf"
PROCESSED_DIR: Path = DATA_DIR / "processed"
PDF_URL: str = (
    "https://cyberireland.ie/wp-content/uploads/2022/09/"
    "State-of-the-Cyber-Security-Sector-in-Ireland-2022.pdf"
)
PDF_PATH: Path = RAW_PDF_DIR / "cyber_ireland_2022.pdf"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR: Path = ROOT / "logs"
LOG_FILE: Path = ROOT / os.getenv("LOG_FILE", "logs/agent_traces.json")

# ── Source label ──────────────────────────────────────────────────────────────
DOCUMENT_SOURCE: str = "Cyber Ireland 2022"

# ── Validate required keys ────────────────────────────────────────────────────
def validate() -> None:
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/app/apikey "
            "and add it to your .env file."
        )
