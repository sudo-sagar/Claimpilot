# Claimpilot
**An enterprise-grade AI system for insurance claims triage.**  Grounded RAG over real insurance policy documents · Multi-agent workflow with tools and guardrails · 5-grader eval harness with CI gate · Full observability (traces, cost, latency).
## Architecture
┌──────────────────────────────────────────────────────────────┐
│ ClaimPilot │
├──────────────────────────────────────────────────────────────┤
│ │
│ Streamlit UI ──▶ Agent Layer ──▶ Retrieval Layer │
│ (3 tabs) (ReAct + (hybrid BM25 + dense │
│ tools + + RRF fusion) │
│ guardrails) │
│ │ │
│ ▼ │
│ Tracer ──▶ data/traces.jsonl │
│ (every LLM call, tool call, cost, │
│ latency, tokens) │
│ │ │
│ ▼ │
│ Eval Harness │
│ (5 graders + CI gate) │
│ │
│ Data: 12 Star Health PDFs → 1.6M chars → 1,592 chunks │
│ │
└──────────────────────────────────────────────────────────────┘
