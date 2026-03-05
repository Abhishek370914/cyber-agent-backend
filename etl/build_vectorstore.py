"""
etl/build_vectorstore.py
────────────────────────
Step 4 (final) of the ETL pipeline:
  1. Load data/processed/chunks.json
  2. Generate embeddings via HuggingFace sentence-transformers (FREE, local)
  3. Upsert into a ChromaDB persistent collection
  4. Print a summary of stored documents

Model used: all-MiniLM-L6-v2 (384-dim, fast, good quality, no API key needed)

Usage:
    python etl/build_vectorstore.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    PROCESSED_DIR,
)

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb required: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers required: pip install sentence-transformers"
    )


# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # free, local, 384-dim
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_collection(persist_dir: Path, collection_name: str) -> Any:
    """Create or connect to the ChromaDB persistent collection."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    # Delete existing collection to avoid dimension mismatch with old OpenAI embeddings
    try:
        client.delete_collection(name=collection_name)
        print(f"[vectorstore] Deleted existing collection '{collection_name}' to rebuild.")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def embed_texts(texts: list[str], model: SentenceTransformer) -> list[list[float]]:
    """Generate embeddings locally using sentence-transformers."""
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def build_vectorstore(
    chunks: list[dict],
    collection: Any,
    model: SentenceTransformer,
) -> None:
    """Embed chunks in batches and upsert them into ChromaDB."""
    total = len(chunks)
    print(f"[vectorstore] Embedding {total} chunks in batches of {BATCH_SIZE} …")

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Upserting"):
        batch = chunks[start : start + BATCH_SIZE]

        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [
            {
                "page": c["page"],
                "section": c["section"],
                "source": c["source"],
                "type": c["type"],
            }
            for c in batch
        ]

        embeddings = embed_texts(texts, model)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    print(
        f"[vectorstore] ✓ {collection.count()} documents stored "
        f"in collection '{collection.name}'."
    )


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {chunks_path}. Run etl/chunking.py first."
        )

    chunks = load_chunks(chunks_path)
    print(f"[vectorstore] Loaded {len(chunks)} chunks.")

    print(f"[vectorstore] Loading embedding model '{EMBEDDING_MODEL_NAME}' (downloads once) …")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("[vectorstore] Model loaded.")

    collection = get_collection(CHROMA_PERSIST_DIR, CHROMA_COLLECTION)
    build_vectorstore(chunks, collection, model)


if __name__ == "__main__":
    main()
