"""
ClaimPilot tools: callable functions the agent uses.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CLAIMS_DIR = ROOT / "app" / "claims"

# Lazy import to avoid loading retriever unless needed
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from upretreival import get_retriever
        _retriever = get_retriever()
    return _retriever


# ---- Tool 1: search_policies ----

def search_policies(query: str, doc_type: str = None) -> str:
    """Returns JSON string of top-5 chunks."""
    r = _get_retriever()
    filters = {"doc_type": doc_type} if doc_type else None
    results = r.search(query, top_k=5, candidate_k=50, filters=filters)

    trimmed = [
        {
            "chunk_id": x["chunk_id"],
            "source_pdf": x["source_pdf"],
            "plan_name": x["plan_name"],
            "doc_type": x["doc_type"],
            "page": x["page"],
            "section": x["section"],
            "text": x["text"][:1200],
        }
        for x in results
    ]
    return json.dumps(trimmed, ensure_ascii=False)


# ---- Tool 2: lookup_claim ----

def lookup_claim(claim_id: str) -> str:
    path = CLAIMS_DIR / (claim_id + ".json")
    if not path.exists():
        return json.dumps({"error": "claim not found: " + claim_id})
    return path.read_text(encoding="utf-8")


# ---- Tool 3: check_fraud_signals ----

def check_fraud_signals(claim_id: str) -> str:
    path = CLAIMS_DIR / (claim_id + ".json")
    if not path.exists():
        return json.dumps({"error": "claim not found: " + claim_id})

    claim = json.loads(path.read_text(encoding="utf-8"))
    flags = []
    risk = 0.0

    # Simple rules — extend as needed
    amount = claim.get("amount_claimed", 0)
    if amount > 1_000_000:
        flags.append("high_amount")
        risk += 0.3

    desc = claim.get("incident_description", "").lower()
    suspicious_terms = ["staged", "fake", "never happened", "not sure", "don't remember"]
    for term in suspicious_terms:
        if term in desc:
            flags.append("suspicious_language:" + term)
            risk += 0.2

    # Filed too late (>60 days)
    from datetime import datetime, timedelta
    try:
        filed = datetime.strptime(claim["filed_date"], "%Y-%m-%d")
        if datetime.now() - filed > timedelta(days=60):
            flags.append("late_filing")
            risk += 0.15
    except Exception:
        pass

    return json.dumps(
        {
            "claim_id": claim_id,
            "risk_score": round(min(risk, 1.0), 2),
            "flags": flags,
        },
        ensure_ascii=False,
    )


# ---- Tool 4: request_human_review ----

def request_human_review(reason: str) -> str:
    return json.dumps({"acknowledged": True, "reason": reason})


# ---- Registry ----

TOOL_FUNCTIONS = {
    "search_policies": search_policies,
    "lookup_claim": lookup_claim,
    "check_fraud_signals": check_fraud_signals,
    "request_human_review": request_human_review,
}


def dispatch(name: str, args: dict) -> str:
    if name not in TOOL_FUNCTIONS:
        return json.dumps({"error": "unknown tool: " + name})
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return json.dumps({"error": type(e).__name__ + ": " + str(e)})