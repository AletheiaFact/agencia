#!/usr/bin/env python3
"""End-to-end test runner for the Agencia fact-checking pipeline.

Sends claims from test_claims.json to a running server, streams NDJSON
responses, and saves full results to tests/results/ for analysis.

Usage:
    # Run all claims:
    python tests/run_e2e.py

    # Run specific claim(s) by ID:
    python tests/run_e2e.py online-01 gazette-01

    # Run against a different server:
    python tests/run_e2e.py --url http://localhost:8080

    # Tag the run (e.g., provider name for A/B comparison):
    python tests/run_e2e.py --tag openai
    python tests/run_e2e.py --tag anthropic
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"
CLAIMS_FILE = FIXTURES_DIR / "test_claims.json"
DEFAULT_URL = "http://localhost:8080"


def load_claims(claim_ids: list[str] | None = None) -> list[dict]:
    with open(CLAIMS_FILE) as f:
        data = json.load(f)
    claims = data["claims"]
    if claim_ids:
        claims = [c for c in claims if c["id"] in claim_ids]
        found = {c["id"] for c in claims}
        missing = set(claim_ids) - found
        if missing:
            print(f"WARNING: claim IDs not found: {missing}")
    return claims


def run_claim(base_url: str, claim_def: dict, session_id: str) -> dict:
    """Send a claim to the server and collect the full NDJSON stream."""
    payload = {
        "session_id": session_id,
        "input": {
            "claim": claim_def["claim"],
            "language": claim_def.get("language", "pt"),
            "search_type": claim_def.get("search_type", "online"),
            "context": claim_def.get("context", {}),
        },
    }

    result = {
        "claim_id": claim_def["id"],
        "claim": claim_def["claim"],
        "search_type": claim_def.get("search_type", "online"),
        "language": claim_def.get("language", "pt"),
        "expected_classifications": claim_def.get("expected_classifications", []),
        "session_id": session_id,
        "execution_id": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "duration_seconds": None,
        "status": None,
        "classification": None,
        "classification_match": None,
        "steps": [],
        "final_message": None,
        "error": None,
    }

    t0 = time.monotonic()
    print(f"\n{'='*60}")
    print(f"  [{claim_def['id']}] {claim_def['claim'][:70]}")
    print(f"  search_type={claim_def.get('search_type')}  language={claim_def.get('language')}")
    print(f"{'='*60}")

    try:
        with httpx.stream(
            "POST",
            f"{base_url}/invoke",
            json=payload,
            timeout=httpx.Timeout(connect=10, read=300, write=10, pool=10),
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  [!] Bad NDJSON line: {line[:100]}")
                    continue

                status = event.get("status")

                if status == "started":
                    result["execution_id"] = event.get("execution_id")
                    print(f"  -> started  execution_id={result['execution_id']}")

                elif status == "processing":
                    step = event.get("step", "")
                    result["steps"].append(step)
                    # Truncate long step messages for display
                    display = step[:100] + "..." if len(step) > 100 else step
                    print(f"  -> step: {display}")

                elif status == "complete":
                    result["status"] = "complete"
                    result["final_message"] = event.get("message")

                elif status == "error":
                    result["status"] = "error"
                    result["error"] = event.get("detail")
                    print(f"  -> ERROR: {event.get('detail', '')[:200]}")

    except httpx.HTTPStatusError as e:
        result["status"] = "http_error"
        result["error"] = f"{e.response.status_code}: {e.response.text[:500]}"
        print(f"  -> HTTP ERROR: {result['error']}")
    except httpx.ConnectError:
        result["status"] = "connection_error"
        result["error"] = f"Could not connect to {base_url}"
        print(f"  -> CONNECTION ERROR: {result['error']}")
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
        print(f"  -> EXCEPTION: {e}")

    elapsed = time.monotonic() - t0
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["duration_seconds"] = round(elapsed, 2)

    # Extract classification from final message
    if result["final_message"]:
        msg = result["final_message"]
        # The final message may be a dict (state) with "messages" key containing JSON
        messages = msg.get("messages") if isinstance(msg, dict) else msg
        if isinstance(messages, str):
            try:
                parsed = json.loads(messages.strip().strip("`").strip())
                result["classification"] = parsed.get("classification")
            except (json.JSONDecodeError, AttributeError):
                # Try to find classification in the raw string
                for known in [
                    "False", "Trustworthy, but", "Trustworthy", "Arguable",
                    "Misleading", "Exaggerated", "Unsustainable",
                    "Unverifiable", "Not Fact",
                ]:
                    if known in str(messages):
                        result["classification"] = known
                        break

    # Check if classification matches expected (normalize for comma/spacing variants)
    expected = result["expected_classifications"]
    if result["classification"] and expected:
        def _normalize(s: str) -> str:
            return s.lower().replace(",", "").replace("  ", " ").strip()
        norm_classification = _normalize(result["classification"])
        result["classification_match"] = any(
            _normalize(e) == norm_classification for e in expected
        )

    # Print summary
    match_icon = {True: "PASS", False: "FAIL", None: "N/A"}[result["classification_match"]]
    print(f"\n  Result: {result['classification'] or 'N/A'}  [{match_icon}]")
    print(f"  Expected: {expected}")
    print(f"  Duration: {result['duration_seconds']}s")
    print(f"  Steps: {len(result['steps'])}")

    return result


def save_results(results: list[dict], tag: str | None = None):
    """Save results to tests/results/ with timestamp."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{tag}" if tag else ""
    filename = f"run_{timestamp}{tag_suffix}.json"
    filepath = RESULTS_DIR / filename

    run_data = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "provider": os.environ.get("LLM_PROVIDER", "openai"),
        "model": os.environ.get("LLM_MODEL", "default"),
        "model_mini": os.environ.get("LLM_MODEL_MINI", "default"),
        "total_claims": len(results),
        "passed": sum(1 for r in results if r["classification_match"] is True),
        "failed": sum(1 for r in results if r["classification_match"] is False),
        "errors": sum(1 for r in results if r["status"] != "complete"),
        "total_duration_seconds": round(sum(r["duration_seconds"] or 0 for r in results), 2),
        "results": results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {filepath}")
    print(f"  Total: {run_data['total_claims']}  Pass: {run_data['passed']}  "
          f"Fail: {run_data['failed']}  Errors: {run_data['errors']}")
    print(f"  Total duration: {run_data['total_duration_seconds']}s")
    print(f"{'='*60}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run E2E tests against the Agencia server")
    parser.add_argument("claim_ids", nargs="*", help="Specific claim IDs to run (default: all)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL})")
    parser.add_argument("--tag", default=None, help="Tag for this run (e.g., 'openai', 'anthropic')")
    args = parser.parse_args()

    claims = load_claims(args.claim_ids or None)
    if not claims:
        print("No claims to run.")
        sys.exit(1)

    print(f"\nRunning {len(claims)} claim(s) against {args.url}")
    if args.tag:
        print(f"Tag: {args.tag}")

    session_id = f"e2e-test-{uuid.uuid4().hex[:8]}"
    results = []

    for claim_def in claims:
        result = run_claim(args.url, claim_def, session_id)
        results.append(result)

    save_results(results, tag=args.tag)


if __name__ == "__main__":
    main()
