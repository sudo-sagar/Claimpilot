"""
ClaimPilot agent: ReAct loop with function calling + structured output.

Usage:
    from agent import triage
    result = triage("CLM-0001")
"""
import sys
import json
import os
from pathlib import Path


from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from tools import dispatch
from schemas import ALL_TOOLS, TriageDecision
from tracer import Tracer


load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))
CLAIMS_DIR = ROOT / "app" / "claims"

MODEL = "nvidia/nemotron-3.5-lightning:free"
MAX_STEPS = 8
MAX_COST_USD = 0.05  # hard cap per claim

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # <-- The critical line
    api_key=os.environ["API_KEY"]       # Your sk-or-v1-... key
)


SYSTEM_PROMPT = """You are ClaimPilot, an AI assistant for insurance claims triage.

CRITICAL: When you return a TriageDecision, you MUST include ALL of these fields:
  decision (approve/reject/escalate), confidence (0.0-1.0),
  rationale (string), citations (array of objects), amount_recommended (number),
  escalation_reason (string).

If you don't have full confidence, use confidence=0.7 as a default. Never omit a field.
Return ONLY valid JSON matching the TriageDecision schema. No markdown, no text outside JSON.

Your job: given a claim, decide whether to approve, reject, or escalate it
based on the actual policy documents.

PROCESS:
1. Call lookup_claim to get the claim details.
2. Call search_policies with specific queries (e.g., "coverage for X",
   "exclusions for Y", "waiting period for Z") to find relevant policy text.
3. If needed, call check_fraud_signals to look for red flags.
4. Based on the policy text, decide: approve, reject, or escalate.
   - approve: incident is clearly covered, no exclusions apply.
   - reject: incident clearly matches an exclusion.
   - escalate: policy language is ambiguous, or fraud risk is high,
     or confidence is below 0.7.
5. Return a TriageDecision with citations from the search_policies results.

RULES:
- NEVER cite a chunk_id that didn't come from search_policies.
- Prefer 'wordings' documents over 'brochures' — wordings are legally binding.
- If nothing in the policy covers the incident, escalate (don't guess).
- Keep rationale concise (2-4 sentences).
"""


def _load_claim(claim_id):
    path = CLAIMS_DIR / (claim_id + ".json")
    if not path.exists():
        raise FileNotFoundError("Claim not found: " + claim_id)
    return json.loads(path.read_text(encoding="utf-8"))


def triage(claim_id: str, verbose: bool = False) -> dict:
    """Run the agent on a claim. Returns a dict with decision + trace."""
    claim = _load_claim(claim_id)
    tracer = Tracer(kind="triage", meta={"claim_id": claim_id})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Triage this claim.\n\n"
                "Claim ID: " + claim_id + "\n"
                "Policy ID: " + claim["policy_id"] + "\n"
                "Policy type: " + claim["policy_type"] + "\n"
                "Amount claimed (INR): " + str(claim["amount_claimed"]) + "\n"
                "Incident: " + claim["incident_description"] + "\n\n"
                "Use the tools. Then return a TriageDecision."
            ),
        },
    ]

    trace = []
    total_tokens = 0

    for step in range(MAX_STEPS):
        if verbose:
            print("--- step " + str(step + 1) + " ---")

        tool_choice = "auto" if step < MAX_STEPS - 1 else "none"

        
        import time as _time
        _t0 = _time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice=tool_choice,
            temperature=0.1,
        )
        _llm_ms = (_time.time() - _t0) * 1000

        usage = resp.usage
        if usage:
            total_tokens += usage.total_tokens
            tracer.record_llm(
                model=MODEL,
                tokens_in=usage.prompt_tokens,
                tokens_out=usage.completion_tokens,
                latency_ms=_llm_ms,
                step_name="step_" + str(step + 1),
            )
        

        msg = resp.choices[0].message

        messages.append(msg.model_dump(exclude_none=True))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                
                _t0 = _time.time()
                result = dispatch(name, args)
                _tool_ms = (_time.time() - _t0) * 1000

                tracer.record_tool(name=name, args=args, result_preview=result, latency_ms=_tool_ms)

                trace.append({"step": step + 1, "tool": name, "args": args,
                              "result_preview": result[:300]})
                

                if verbose:
                    print("  tool: " + name + "(" + json.dumps(args)[:80] + ")")
                    print("  result: " + result[:200].replace("\n", " "))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        content = msg.content or ""
        decision = _parse_decision(content, claim_id)
        if decision is not None:
            trace_record = tracer.finish()
            return {
                "claim_id": claim_id,
                "decision": decision.model_dump(),
                "trace": trace,
                "trace_id": trace_record["trace_id"],
                "cost_usd": trace_record["total_cost_usd"],
                "duration_ms": trace_record["duration_ms"],
                "steps": step + 1,
                "total_tokens": total_tokens,
                "model": MODEL,
            }
            

        messages.append({
            "role": "user",
            "content": "Your answer was not valid JSON matching TriageDecision. "
                       "Return ONLY a JSON object matching the schema. No markdown."
        })

        # Graceful fallback: escalate instead of crashing
    decision = _fallback_decision(claim_id)
    trace_record = tracer.finish()
    return {
        "claim_id": claim_id,
        "decision": decision.model_dump(),
        "trace": trace,
        "trace_id": trace_record["trace_id"],
        "cost_usd": trace_record["total_cost_usd"],
        "duration_ms": trace_record["duration_ms"],
        "steps": MAX_STEPS,
        "total_tokens": total_tokens,
        "model": MODEL,
    }

def _parse_decision(content: str, claim_id: str):
    """Try to parse the model's final content as a TriageDecision."""
    if not content:
        return None

    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip("` \n")

    try:
        data = json.loads(text)
    except Exception:
        return None

    # Fill missing confidence with a safe default
    if "confidence" not in data:
        data["confidence"] = 0.7

    try:
        return TriageDecision(**data)
    except ValidationError as e:
        print("Validation error: " + str(e)[:200])
        return None


def _fallback_decision(claim_id: str) -> TriageDecision:
    """If agent fails to produce valid JSON, escalate for human review."""
    return TriageDecision(
        decision="escalate",
        confidence=0.5,
        rationale="Agent failed to produce a valid decision. Escalating for human review.",
        citations=[],
        amount_recommended=0.0,
        escalation_reason="parse_failure",
    )


if __name__ == "__main__":
    import sys

    claim_id = sys.argv[1] if len(sys.argv) > 1 else "CLM-0001"
    result = triage(claim_id, verbose=True)
    print("")
    print("=" * 70)
    print("FINAL DECISION")
    print("=" * 70)
    print(json.dumps(result, indent=2, ensure_ascii=False))