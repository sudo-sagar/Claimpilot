"""
Smoke test for tracing — works with agent.py.
"""

import json
from agents import triage
from tracer import get_trace, ops_summary


def main():
    print("Running triage with tracing...")
    result = triage("CLM-0001", verbose=False)

    print("")
    print("=" * 70)
    print("TRACE ID: " + result.get("trace_id", "MISSING"))
    print("Cost:     $" + str(result.get("cost_usd", 0)))
    print("Duration: " + str(result.get("duration_ms", 0)) + " ms")
    print("=" * 70)

    if result.get("trace_id"):
        tr = get_trace(result["trace_id"])
        print("")
        print("Full trace (first 1500 chars):")
        print(json.dumps(tr, indent=2, ensure_ascii=False)[:1500])

    print("")
    print("OPS SUMMARY (triage):")
    print(json.dumps(ops_summary(kind="triage"), indent=2))


if __name__ == "__main__":
    main()