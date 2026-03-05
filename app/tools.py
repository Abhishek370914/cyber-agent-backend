"""
app/tools.py
────────────
LangChain-compatible tool definitions for the reasoning agent.

Tool 1 – document_retrieval    : Semantic search over the vector store
Tool 2 – table_data_extractor  : Filter for table-type chunks only
Tool 3 – math_calculator       : Safe Python expression evaluator
Tool 4 – citation_verifier     : Locate an exact quote in raw page text
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from langchain.tools import tool

from app.retriever import query_documents
from app.config import PROCESSED_DIR, RETRIEVAL_TOP_K


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 – Document Retrieval
# ─────────────────────────────────────────────────────────────────────────────

@tool
def document_retrieval(query: str) -> str:
    """
    Search the Cyber Ireland 2022 knowledge base using semantic similarity.
    Returns the most relevant text passages with page numbers and section titles.
    Use this tool to find factual information, statistics, or narrative content.

    Input:  A natural-language search query.
    Output: JSON array of {text, page, section, source, type} objects.
    """
    hits = query_documents(query, n_results=RETRIEVAL_TOP_K)
    if not hits:
        return json.dumps({"result": "No relevant documents found."})
    return json.dumps(hits, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 – Table Data Extractor
# ─────────────────────────────────────────────────────────────────────────────

@tool
def table_data_extractor(query: str) -> str:
    """
    Search specifically for table data in the Cyber Ireland 2022 report.
    Use this tool when the answer involves structured data, regional breakdowns,
    statistics, percentages, or comparative metrics from tables.

    Input:  A natural-language description of the data you are looking for.
    Output: JSON array of matching table chunks with page numbers.
    """
    hits = query_documents(query, n_results=RETRIEVAL_TOP_K, where={"type": "table"})
    if not hits:
        # Fallback: search all types if no table-specific results found
        hits = query_documents(query, n_results=RETRIEVAL_TOP_K)
        if not hits:
            return json.dumps({"result": "No table data found for this query."})
    return json.dumps(hits, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 – Math Calculator
# ─────────────────────────────────────────────────────────────────────────────

# Safe built-ins whitelist for the eval sandbox
_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sum": sum,
    "len": len,
    "math": math,
    "log": math.log,
    "sqrt": math.sqrt,
    "exp": math.exp,
}


@tool
def math_calculator(expression: str) -> str:
    """
    Execute a Python math expression safely.
    Supports: arithmetic (+, -, *, /, **), math.log(), math.sqrt(), round(), etc.

    Use this tool for:
    - CAGR: ( (end/start) ** (1/n) - 1 ) * 100
    - Percentages, ratios, differences
    - Any numeric computation required to answer the query

    Input:  A valid Python math expression string (e.g., "((6500/3000)**(1/8)-1)*100")
    Output: The computed result as a string.
    """
    # Strip markdown code fences if the LLM included them
    expression = re.sub(r"```[a-z]*\n?|```", "", expression).strip()

    # Block obviously unsafe patterns
    forbidden = ["import", "open", "exec", "eval", "os", "sys", "__"]
    for token in forbidden:
        if token in expression:
            return f"Error: expression contains forbidden token '{token}'."

    try:
        result = eval(expression, _SAFE_GLOBALS, {})  # noqa: S307 – sandboxed
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 – Citation Verifier
# ─────────────────────────────────────────────────────────────────────────────

def _load_pages() -> list[dict]:
    """Load pages.json lazily."""
    pages_path = PROCESSED_DIR / "pages.json"
    if not pages_path.exists():
        return []
    with open(pages_path, encoding="utf-8") as f:
        return json.load(f)


@tool
def citation_verifier(quote: str) -> str:
    """
    Verify that a specific text quote exists verbatim (or near-verbatim) in the
    Cyber Ireland 2022 document, and return the page number where it appears.
    Use this tool to validate facts before including them in the final answer.

    Input:  A short quote or key phrase (5–30 words) to search for.
    Output: JSON with {found: bool, page: int, context: str} or a list if found
            on multiple pages.
    """
    pages = _load_pages()
    if not pages:
        return json.dumps({"found": False, "error": "pages.json not available."})

    quote_lower = quote.lower().strip()
    # Remove punctuation for fuzzy matching
    quote_clean = re.sub(r"[^\w\s]", "", quote_lower)
    words = quote_clean.split()
    # Use a sliding partial match: require ≥60% of words to appear contiguously
    min_match = max(3, int(len(words) * 0.6))

    matches: list[dict] = []
    for page_rec in pages:
        text_lower = page_rec["text"].lower()
        text_clean = re.sub(r"[^\w\s]", "", text_lower)

        # Exact substring check first
        if quote_lower in text_lower:
            # Extract a context snippet
            idx = text_lower.index(quote_lower)
            context = page_rec["text"][max(0, idx - 80): idx + len(quote) + 80]
            matches.append(
                {
                    "found": True,
                    "page": page_rec["page"],
                    "match_type": "exact",
                    "context": context.strip(),
                }
            )
            continue

        # Partial / fuzzy match: sliding window over cleaned text words
        text_words = text_clean.split()
        for i in range(len(text_words) - min_match + 1):
            window = text_words[i: i + len(words)]
            common = sum(1 for w in words if w in window)
            if common >= min_match:
                matches.append(
                    {
                        "found": True,
                        "page": page_rec["page"],
                        "match_type": "partial",
                        "matched_words": common,
                        "query": quote,
                    }
                )
                break

    if not matches:
        return json.dumps({"found": False, "quote": quote})
    return json.dumps(matches if len(matches) > 1 else matches[0], indent=2)


# ── Exported list for agent registration ─────────────────────────────────────
ALL_TOOLS = [
    document_retrieval,
    table_data_extractor,
    math_calculator,
    citation_verifier,
]
