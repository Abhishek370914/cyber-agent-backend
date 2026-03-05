"""
app/agent.py
────────────
Simplified reasoning agent:
  1. Embed the query locally (sentence-transformers, free)
  2. Retrieve top-k chunks from ChromaDB
  3. Call Gemini (free tier) to synthesize a cited answer

No LangGraph / tool-calling complexity — direct, reliable, and fully free.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from app.config import LOG_DIR, LOG_FILE, GOOGLE_API_KEY, validate
from app.retriever import query_documents
import math
import re


# ── Logging ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _append_log(query: str, step: str, tool: str, inp: str, out: str, trace: list) -> None:
    record = {
        "step": step,
        "tool_used": tool,
        "input": inp[:500],
        "output": out[:800],
        "timestamp": _ts(),
    }
    trace.append(record)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append({"query": query, **record})
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


# ── Math helper ───────────────────────────────────────────────────────────────

def _safe_eval(expression: str) -> str:
    """Evaluate a math expression safely."""
    expression = re.sub(r"```[a-z]*\n?|```", "", expression).strip()
    safe_globals = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum, "len": len, "math": math,
        "log": math.log, "sqrt": math.sqrt, "exp": math.exp,
    }
    try:
        result = eval(expression, safe_globals, {})  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# ── Citation extractor ────────────────────────────────────────────────────────

def _extract_citations(chunks: list[dict]) -> list[dict]:
    seen: set[int] = set()
    citations = []
    for c in chunks:
        page = c.get("page", 0)
        if page and page not in seen:
            seen.add(page)
            citations.append({
                "page": page,
                "quote": c.get("text", "")[:200].replace("\n", " "),
            })
    return citations


# ── Gemini call ───────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    """Call Gemini API. Falls back gracefully if unavailable."""
    try:
        import google.generativeai as genai
        import warnings
        warnings.filterwarnings("ignore")
        genai.configure(api_key=GOOGLE_API_KEY)

        model_names = [
            "gemini-2.0-flash",
            "gemini-2.5-flash-preview-04-17",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-lite",
        ]
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception:
                continue
    except Exception:
        pass
    return ""  # Signal that LLM is unavailable — caller will fallback


def _format_answer_from_chunks(query: str, chunks: list[dict], math_result: str) -> str:
    """
    Build a structured answer directly from retrieved chunks, no LLM needed.
    """
    lines = [
        f"**Query:** {query}\n",
        "**Retrieved Evidence from Cyber Ireland 2022 Report:**\n",
    ]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"{i}. [Page {c['page']} | {c.get('section', '')}]\n"
            f"   {c['text'][:300].replace(chr(10), ' ')}\n"
        )
    if math_result:
        lines.append(f"\n**Calculation Result:** {math_result}\n")
    lines.append(
        "\n*Note: Answer synthesised directly from document chunks. "
        "For AI-generated prose, add a valid GOOGLE_API_KEY with Gemini access.*"
    )
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(query: str) -> dict:
    """
    Run the reasoning agent on a user query.

    Steps:
      1. Retrieve relevant chunks from ChromaDB (local, free)
      2. Check if math calculation needed and run it
      3. Call Gemini to synthesize a cited answer (free tier)

    Returns:
        {
            "answer": str,
            "citations": [{"page": int, "quote": str}],
            "reasoning_trace": [{"step", "tool_used", "input", "output", "timestamp"}]
        }
    """
    validate()
    trace: list[dict] = []

    # ── Step 1: Retrieve relevant chunks ─────────────────────────────────────
    _append_log(query, "retrieve", "document_retrieval", query, "Searching vectorstore…", trace)
    chunks = query_documents(query, n_results=6)

    if not chunks:
        return {
            "answer": "No relevant information found in the document for this query.",
            "citations": [],
            "reasoning_trace": trace,
        }

    context_text = "\n\n".join([
        f"[Page {c['page']} | {c['section']}]\n{c['text']}"
        for c in chunks
    ])
    _append_log(
        query, "retrieve", "document_retrieval", query,
        f"Found {len(chunks)} chunks across pages: {sorted(set(c['page'] for c in chunks))}",
        trace,
    )

    # ── Step 2: Check if math is needed and compute ───────────────────────────
    math_result = ""
    math_keywords = ["cagr", "growth rate", "calculate", "percent", "ratio", "compound"]
    if any(kw in query.lower() for kw in math_keywords):
        _append_log(query, "math", "math_calculator", query, "Math analysis required", trace)
        # Ask Gemini to extract the expression first
        math_prompt = (
            f"Based on this context from the Cyber Ireland 2022 report:\n{context_text[:2000]}\n\n"
            f"For the query: {query}\n\n"
            "If a calculation is needed, write ONLY the Python math expression (e.g. ((6500/3000)**(1/8)-1)*100). "
            "If no calculation is needed, write NONE."
        )
        expr = _call_gemini(math_prompt).strip()
        if expr.upper() != "NONE" and expr:
            expr_clean = re.sub(r"[^0-9+\-*/().** ]", "", expr).strip()
            if expr_clean:
                math_result = _safe_eval(expr_clean)
                _append_log(query, "math", "math_calculator", expr_clean, math_result, trace)

    # ── Step 3: Synthesize answer with Gemini ────────────────────────────────
    _append_log(query, "synthesize", "gemini", query, "Composing answer…", trace)

    math_section = f"\nMath calculation result: {math_result}\n" if math_result else ""

    synthesis_prompt = f"""You are an expert research analyst for the Cyber Ireland 2022 Cybersecurity Sector Report.

Answer the following query using ONLY the provided context. Be specific, cite page numbers for every fact.

Context from the report:
{context_text}
{math_section}
Query: {query}

Provide your answer in this format:
1. Direct Answer: [specific answer with numbers]
2. Evidence: [exact quotes and page numbers from the context]
3. Reasoning: [how you derived the answer]

Be concise and factually accurate. Only use information from the provided context."""

    answer = _call_gemini(synthesis_prompt)

    # ── Fallback: format directly from chunks if LLM unavailable ─────────────
    if not answer:
        _append_log(query, "synthesize", "fallback", query, "LLM unavailable, using chunk-based answer", trace)
        answer = _format_answer_from_chunks(query, chunks, math_result)

    _append_log(query, "synthesize", "gemini", query, answer[:400], trace)
    citations = _extract_citations(chunks)

    return {
        "answer": answer,
        "citations": citations,
        "reasoning_trace": trace,
    }
