"""
ClaimPilot schemas: the contract between agent and system.

The agent MUST return a TriageDecision. Anything else fails validation.
"""

from typing import Literal, List
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str = Field(description="Chunk ID from search_policies results.")
    source_pdf: str = Field(default="unknown", description="PDF filename the chunk came from.")
    page: int = Field(default=0, description="Page number.")
    section: str = Field(default="Unknown", description="Section name, e.g. 'EXCLUSIONS'.")


class TriageDecision(BaseModel):
    decision: Literal["approve", "reject", "escalate"] = Field(
        description="approve if covered by policy; reject if excluded; escalate if ambiguous."
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="0 = pure guess, 1 = certain. Default 0.7 if unsure."
    )
    rationale: str = Field(
        description="Short explanation (2-4 sentences) citing the policy sections."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Chunks the decision is based on. Must come from search_policies results.",
    )
    amount_recommended: float = Field(
        default=0.0, description="Payout amount in INR if approving, else 0."
    )
    escalation_reason: str = Field(
        default="", description="If escalating, why."
    )


# OpenAI tools API wants JSON Schema. Pydantic gives it to us.
TRIAGE_DECISION_SCHEMA = TriageDecision.model_json_schema()


# Tool schemas (function calling)
SEARCH_POLICIES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_policies",
        "description": (
            "Search insurance policy documents. Returns chunks with "
            "source_pdf, page, section, and text. Use this to check "
            "coverage rules, exclusions, waiting periods, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "doc_type": {
                    "type": "string",
                    "enum": ["wordings", "brochure"],
                    "description": "Optional filter. Use 'wordings' for legal text.",
                },
            },
            "required": ["query"],
        },
    },
}


LOOKUP_CLAIM_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_claim",
        "description": "Fetch full details of a claim by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
            },
            "required": ["claim_id"],
        },
    },
}


CHECK_FRAUD_SIGNALS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_fraud_signals",
        "description": (
            "Run rule-based fraud signals on a claim. Returns risk score "
            "and flags (e.g., amount > sum insured, filed too late)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim_id": {"type": "string"},
            },
            "required": ["claim_id"],
        },
    },
}


REQUEST_HUMAN_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "request_human_review",
        "description": (
            "Escalate the claim to a human reviewer. Use when the policy "
            "language is ambiguous or confidence is low."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
            "required": ["reason"],
        },
    },
}


ALL_TOOLS = [
    SEARCH_POLICIES_TOOL,
    LOOKUP_CLAIM_TOOL,
    CHECK_FRAUD_SIGNALS_TOOL,
    REQUEST_HUMAN_REVIEW_TOOL,
]