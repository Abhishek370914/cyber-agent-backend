"""
app/retriever.py
────────────────
ChromaDB retrieval wrapper using local HuggingFace sentence-transformers.

Provides a single function `query_documents` that accepts
a natural-language query and optional metadata filters,
returns the top-k matching chunks with full metadata.
"""
from __future__ import annotations

from typing import Any

from app.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    RETRIEVAL_TOP_K,
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


# ── Singletons (lazily initialised) ──────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model: "SentenceTransformer | None" = None
_collection: Any = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection() -> Any:
    global _collection
    if _collection is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        db = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        _collection = db.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ─────────────────────────────────────────────────────────────────────────────

def _embed_query(query: str) -> list[float]:
    model = _get_model()
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding[0].tolist()


def query_documents(
    query: str,
    n_results: int = RETRIEVAL_TOP_K,
    where: dict | None = None,
) -> list[dict]:
    """
    Semantic search over the ChromaDB collection.

    Parameters
    ----------
    query:     Natural-language search string.
    n_results: Number of top chunks to return.
    where:     Optional ChromaDB metadata filter, e.g. {"type": "table"}.

    Returns
    -------
    List of dicts with keys: text, page, section, source, type, distance.
    """
    collection = _get_collection()

    # Guard: if empty, return nothing
    if collection.count() == 0:
        return []

    query_embedding = _embed_query(query)

    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "page": meta.get("page", 0),
                "section": meta.get("section", ""),
                "source": meta.get("source", ""),
                "type": meta.get("type", "text"),
                "distance": round(dist, 4),
            }
        )
    return hits


def collection_count() -> int:
    """Return number of documents in the collection."""
    return _get_collection().count()
