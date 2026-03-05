"""
tests/test_queries.py
──────────────────────
Automated test script that exercises all three "Moments of Truth" scenarios
against a running FastAPI backend.

Requires:
    uvicorn app.main:app --reload
    (running in a separate terminal from project root)

Usage:
    python tests/test_queries.py

Saves structured results to logs/test_results.json
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
TIMEOUT = 120  # seconds – agent may take time for multi-step reasoning
ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "logs" / "test_results.json"

# ── Test definitions ──────────────────────────────────────────────────────────

TESTS = [
    {
        "name": "Test 1 – The Verification Challenge",
        "description": (
            "Retrieve the exact integer job count reported in the document, "
            "with page citation and verifiable quote."
        ),
        "query": (
            "What is the total number of jobs reported, "
            "and where exactly is this stated?"
        ),
        "checks": [
            # We look for presence of a number and a page citation in the answer
            lambda r: any(char.isdigit() for char in r["answer"]),
            lambda r: len(r["citations"]) > 0,
            lambda r: len(r["reasoning_trace"]) > 0,
        ],
    },
    {
        "name": "Test 2 – The Data Synthesis Challenge",
        "description": (
            "Navigate regional tables, extract Pure-Play firm metrics, "
            "and compare South-West concentration to National Average."
        ),
        "query": (
            "Compare the concentration of 'Pure-Play' cybersecurity firms "
            "in the South-West against the National Average."
        ),
        "checks": [
            lambda r: "south" in r["answer"].lower() or "south-west" in r["answer"].lower(),
            lambda r: len(r["citations"]) > 0,
            lambda r: len(r["reasoning_trace"]) > 0,
        ],
    },
    {
        "name": "Test 3 – The Forecasting Challenge (CAGR)",
        "description": (
            "Find 2022 baseline, find 2030 job target, use math_calculator "
            "tool to compute the required Compound Annual Growth Rate."
        ),
        "query": (
            "Based on our 2022 baseline and the stated 2030 job target, "
            "what is the required compound annual growth rate (CAGR) to hit that goal?"
        ),
        "checks": [
            lambda r: "cagr" in r["answer"].lower() or "%" in r["answer"],
            lambda r: len(r["citations"]) > 0,
            # Verify the math_calculator tool was actually invoked
            lambda r: any(
                "math_calculator" in str(step.get("tool_used", ""))
                or "math" in str(step.get("tool_used", ""))
                or "calculator" in str(step.get("tool_used", ""))
                for step in r.get("reasoning_trace", [])
            ),
        ],
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def _check_health(client: httpx.Client) -> bool:
    try:
        resp = client.get(f"{BASE_URL}/health", timeout=10)
        data = resp.json()
        ready = data.get("vectorstore_ready", False)
        doc_count = data.get("document_count", 0)
        print(f"[health] status={data.get('status')}  docs={doc_count}  ready={ready}")
        return ready
    except Exception as exc:
        print(f"[health] ✗ Cannot reach server: {exc}")
        return False


def _run_test(client: httpx.Client, test: dict) -> dict:
    print(f"\n{_divider('═')}")
    print(f"  {test['name']}")
    print(f"  {test['description']}")
    print(_divider("─"))
    print(f"  Query: {test['query']}")
    print(_divider("─"))

    try:
        resp = client.post(
            f"{BASE_URL}/query",
            json={"query": test["query"]},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        print("  ✗ Request timed out.")
        return {"name": test["name"], "status": "TIMEOUT", "error": "Request timed out"}
    except Exception as exc:
        print(f"  ✗ Request failed: {exc}")
        return {"name": test["name"], "status": "ERROR", "error": str(exc)}

    # ── Print answer ──────────────────────────────────────────────────────────
    print("\n  ANSWER:")
    for line in textwrap.wrap(data.get("answer", "(empty)"), width=68):
        print(f"    {line}")

    # ── Print citations ───────────────────────────────────────────────────────
    citations = data.get("citations", [])
    if citations:
        print(f"\n  CITATIONS ({len(citations)}):")
        for c in citations[:5]:
            snippet = str(c.get("quote", ""))[:100].replace("\n", " ")
            print(f"    • Page {c.get('page', '?')}: \"{snippet}…\"")
    else:
        print("\n  CITATIONS: none returned")

    # ── Print reasoning trace summary ─────────────────────────────────────────
    trace = data.get("reasoning_trace", [])
    print(f"\n  REASONING TRACE ({len(trace)} steps):")
    for step in trace:
        print(
            f"    [{step.get('timestamp', '?')[:19]}] "
            f"{step.get('step', '?'):12s} | "
            f"tool={step.get('tool_used', '?')}"
        )

    # ── Evaluate checks ───────────────────────────────────────────────────────
    checks = test.get("checks", [])
    passed = 0
    for i, check_fn in enumerate(checks, 1):
        try:
            ok = check_fn(data)
        except Exception:
            ok = False
        status = "✓" if ok else "✗"
        print(f"\n  Check {i}: {status}")
        if ok:
            passed += 1

    overall = "PASS" if passed == len(checks) else "PARTIAL" if passed > 0 else "FAIL"
    print(f"\n  Result: {overall} ({passed}/{len(checks)} checks passed)")
    print(f"  Elapsed: {data.get('elapsed_seconds', '?')}s")

    return {
        "name": test["name"],
        "query": test["query"],
        "status": overall,
        "checks_passed": passed,
        "checks_total": len(checks),
        "answer": data.get("answer", ""),
        "citations": citations,
        "reasoning_trace_steps": len(trace),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "timestamp": data.get("timestamp"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(_divider("═"))
    print("  Cyber Ireland 2022 — Autonomous Agent Test Suite")
    print(f"  Target: {BASE_URL}")
    print(f"  Time:   {datetime.now(tz=timezone.utc).isoformat()}")
    print(_divider("═"))

    with httpx.Client() as client:
        healthy = _check_health(client)
        if not healthy:
            print(
                "\n✗ Vectorstore is not ready.\n"
                "  Please run the ETL pipeline first:\n"
                "    python etl/ingest_pdf.py\n"
                "    python etl/extract_tables.py\n"
                "    python etl/chunking.py\n"
                "    python etl/build_vectorstore.py\n"
                "  Then start the server:\n"
                "    uvicorn app.main:app --reload"
            )
            sys.exit(1)

        results = [_run_test(client, t) for t in TESTS]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{_divider('═')}")
    print("  SUMMARY")
    print(_divider("─"))
    for r in results:
        icon = "✓" if r["status"] == "PASS" else ("~" if r["status"] == "PARTIAL" else "✗")
        print(f"  {icon} {r['name']}: {r['status']}")
    print(_divider("═"))

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"run_at": datetime.now(tz=timezone.utc).isoformat(), "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n  Results saved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
