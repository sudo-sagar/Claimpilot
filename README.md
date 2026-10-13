# Claimpilot
**An enterprise-grade AI system for insurance claims triage.**  Grounded RAG over real insurance policy documents · Multi-agent workflow with tools and guardrails · 5-grader eval harness with CI gate · Full observability (traces, cost, latency).

## Architecture

```mermaid
graph TD
    A[Streamlit UI<br/>3 tabs] --> B[Agent Layer<br/>ReAct + tools + guardrails]
    B --> C[Retrieval Layer<br/>BM25 + dense + RRF fusion]
    B --> D[Tracer<br/>every LLM call, tool call,<br/>cost, latency, tokens]
    C --> E[Data<br/>12 PDFs, 1.6M chars, 1,592 chunks]
    B --> F[Eval Harness<br/>5 graders + CI gate]
    D --> G[(data/traces.jsonl)]
```
