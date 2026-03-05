"""
app/main.py
───────────
FastAPI application entry point.

Endpoints:
  GET  /health    – readiness probe
  POST /query     – run the multi-step agent on a user question

Usage:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.config import CHROMA_COLLECTION, CHROMA_PERSIST_DIR


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cyber Ireland 2022 — Autonomous Intelligence Backend",
    description=(
        "A multi-step reasoning agent that ingests the Cyber Ireland 2022 report "
        "and answers complex queries with citations and full reasoning traces."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The natural-language question to ask the agent.",
        example="What is the total number of cybersecurity jobs reported?",
    )


class Citation(BaseModel):
    page: int
    quote: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    reasoning_trace: list[dict]
    elapsed_seconds: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    collection: str
    document_count: int
    timestamp: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """
    Readiness probe.
    Returns whether the vectorstore is populated and ready to answer queries.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        count = col.count()
        ready = count > 0
    except Exception:
        count = 0
        ready = False

    return HealthResponse(
        status="ok" if ready else "degraded",
        vectorstore_ready=ready,
        collection=CHROMA_COLLECTION,
        document_count=count,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
def query(request: QueryRequest) -> QueryResponse:
    """
    Submit a question to the autonomous reasoning agent.

    The agent will:
    1. Plan a retrieval strategy
    2. Query the vector database
    3. Execute math / verification tools as needed
    4. Return a cited, traceable answer
    """
    # Check vectorstore is ready
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        if col.count() == 0:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Vector store is empty. "
                    "Run the ETL pipeline first: "
                    "python etl/ingest_pdf.py && "
                    "python etl/extract_tables.py && "
                    "python etl/chunking.py && "
                    "python etl/build_vectorstore.py"
                ),
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ChromaDB unavailable: {exc}") from exc

    start = time.perf_counter()
    try:
        result = run_agent(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    elapsed = round(time.perf_counter() - start, 2)

    citations = [
        Citation(page=c.get("page", 0), quote=c.get("quote", ""))
        for c in result.get("citations", [])
    ]

    return QueryResponse(
        query=request.query,
        answer=result.get("answer", ""),
        citations=citations,
        reasoning_trace=result.get("reasoning_trace", []),
        elapsed_seconds=elapsed,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )
