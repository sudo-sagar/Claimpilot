"""
ClaimPilot graders.

Each grader takes (case, result) and returns a structured judgment dict.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = "nvidia/nemotron-3.5-lightning:free"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["API_KEY"],
)


def grade_decision(case, result):
    expected = case["expected_decision"]
    actual = result["decision"]["decision"]
    return {
        "grader": "decision_match",
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
        "score": 1.0 if expected == actual else 0.0,
    }


def grade_citation_recall(case, result):
    citations = result["decision"].get("citations", [])
    has_citation = len(citations) > 0
    well_formed = all(
        isinstance(c, dict) and c.get("chunk_id") for c in citations
    )
    return {
        "grader": "citation_recall",
        "num_citations": len(citations),
        "well_formed": well_formed,
        "passed": has_citation and well_formed,
        "score": 1.0 if (has_citation and well_formed) else 0.0,
    }


STOPWORDS = set(
    "the a an and or of to in on for with is are was were be been being "
    "this that these those it its as at by from has have had not no".split()
)


def _key_terms(text, top_k=8):
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    words = [w for w in words if w not in STOPWORDS]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return set(sorted(freq, key=lambda w: -freq[w])[:top_k])


def grade_citation_grounding(case, result, chunks_lookup):
    citations = result["decision"].get("citations", [])
    if not citations:
        return {
            "grader": "citation_grounding",
            "passed": False,
            "score": 0.0,
            "reason": "no citations",
        }

    incident_terms = _key_terms(case["incident_description"])
    cited_texts = []
    for c in citations:
        cid = c.get("chunk_id")
        chunk = chunks_lookup.get(cid)
        if chunk:
            cited_texts.append((chunk.get("text") or "").lower())

    if not cited_texts:
        return {
            "grader": "citation_grounding",
            "passed": False,
            "score": 0.0,
            "reason": "no cited chunks found in corpus",
        }

    combined = " ".join(cited_texts)
    hits = sum(1 for t in incident_terms if t in combined)
    ratio = hits / max(len(incident_terms), 1)
    return {
        "grader": "citation_grounding",
        "incident_terms": list(incident_terms)[:5],
        "hits": hits,
        "ratio": round(ratio, 2),
        "passed": ratio >= 0.3,
        "score": min(ratio, 1.0),
    }


JUDGE_PROMPT = (
    "You are a strict evaluator for an insurance claims AI.\n\n"
    "Score the RATIONALE on a 1-5 scale:\n"
    "5 = Specific, cites the exact policy rule that applies, unambiguous.\n"
    "4 = Good, cites relevant policy but slightly vague.\n"
    "3 = Reasonable but generic; could apply to many claims.\n"
    "2 = Vague, doesn't clearly explain the decision.\n"
    "1 = Wrong or contradictory to the decision.\n\n"
    "Return ONLY valid JSON: {\"score\": <int 1-5>, \"reason\": \"<short reason>\"}"
)


def grade_rationale(case, result):
    decision = result["decision"]
    prompt = (
        "Claim:\n"
        "  Incident: " + str(case.get("incident_description", "")) + "\n"
        "  Amount claimed: " + str(case.get("amount_claimed", 0)) + "\n\n"
        "Agent's decision: " + decision.get("decision", "?") + "\n"
        "Agent's rationale: " + str(decision.get("rationale", "")) + "\n"
        "Number of citations: " + str(len(decision.get("citations", []))) + "\n\n"
        "Grade the rationale."
    )

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content.startswith("```"):
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip("` \n")
            data = json.loads(content)
            score = int(data.get("score", 0))
            return {
                "grader": "rationale_quality",
                "score": score,
                "normalized": score / 5.0,
                "reason": str(data.get("reason", ""))[:200],
                "passed": score >= 3,
            }
        except Exception as e:
            if attempt == 2:
                return {
                    "grader": "rationale_quality",
                    "passed": False,
                    "score": 0,
                    "error": str(e)[:200],
                }
            time.sleep(2)

    return {"grader": "rationale_quality", "passed": False, "score": 0}


def grade_cost(case, result):
    cost = result.get("cost_usd", 0.0)
    duration = result.get("duration_ms", 0.0)
    return {
        "grader": "cost_latency",
        "cost_usd": round(cost, 6),
        "duration_ms": round(duration, 1),
        "under_cost_budget": cost < 0.02,
        "under_latency_budget": duration < 200000,
        "passed": cost < 0.02 and duration < 200000,
        "score": 1.0 if (cost < 0.02 and duration < 200000) else 0.0,
    }