"""
Smoke test for the ClaimPilot agent.

Picks one claim of each expected decision and runs the agent.
"""

import json
from pathlib import Path

from agent import triage

ROOT = Path(__file__).parent.parent
EVAL_CASES = ROOT / "data" / "eval_cases.jsonl"


def pick_examples():
    """Pick one claim per expected decision."""
    cases = [json.loads(l) for l in EVAL_CASES.open(encoding="utf-8")]
    by_decision = {}
    for c in cases:
        d = c["expected_decision"]
        if d not in by_decision:
            by_decision[d] = c
    return by_decision


def main():
    examples = pick_examples()

    for expected, case in examples.items():
        print("=" * 80)
        print("CLAIM: " + case["claim_id"] + " | expected=" + expected)
        print("Incident: " + case["incident_description"])
        print("=" * 80)

        result = triage(case["claim_id"], verbose=True)
        d = result["decision"]

        match = "MATCH" if d["decision"] == expected else "MISMATCH"
        print("")
        print("Decision: " + d["decision"] + "  (" + match + ")")
        print("Confidence: " + str(d["confidence"]))
        print("Rationale: " + d["rationale"])
        print("Citations: " + str(len(d["citations"])))
        for c in d["citations"][:2]:
            print("  - " + c["source_pdf"] + " p" + str(c["page"]) + " (" + c["section"] + ")")
        print("Tokens: " + str(result["total_tokens"]))
        print("")


if __name__ == "__main__":
    main()