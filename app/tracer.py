"""
ClaimPilot tracer.

Records every LLM call and tool call to data/traces.jsonl.
"""

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRACES_PATH = ROOT / "data" / "traces.jsonl"

# USD per 1M tokens
COST_TABLE = {
    "openai/gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "openai/gpt-4o": {"in": 2.50, "out": 10.00},
    "openai/gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "openai/text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "openai/text-embedding-3-large": {"in": 0.13, "out": 0.0},
    "qwen/qwen3-embedding-8b": {"in": 0.01, "out": 0.0},
    "nvidia/nemotron-3-embed-1b:free": {"in": 0.0, "out": 0.0},
    "nvidia/nemotron-3.5-lightning:free": {"in": 0.0, "out": 0.0},
    "baai/bge-m3": {"in": 0.01, "out": 0.0},
}


def cost_of(model: str, tokens_in: int, tokens_out: int) -> float:
    entry = COST_TABLE.get(model)
    if not entry:
        return 0.0
    return (tokens_in * entry["in"] + tokens_out * entry["out"]) / 1_000_000


class Tracer:
    def __init__(self, kind: str, meta: dict = None):
        self.trace_id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.meta = meta or {}
        self.start_time = time.time()
        self.steps = []
        self.total_cost = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0

    def record_llm(self, model: str, tokens_in: int, tokens_out: int,
                   latency_ms: float, step_name: str = ""):
        cost = cost_of(model, tokens_in, tokens_out)
        self.total_cost += cost
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.steps.append({
            "step": len(self.steps) + 1,
            "type": "llm",
            "step_name": step_name,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "latency_ms": round(latency_ms, 1),
        })

    def record_tool(self, name: str, args: dict, result_preview: str, latency_ms: float):
        self.steps.append({
            "step": len(self.steps) + 1,
            "type": "tool",
            "tool_name": name,
            "tool_args": args,
            "result_preview": result_preview[:300],
            "latency_ms": round(latency_ms, 1),
        })

    def finish(self) -> dict:
        duration_ms = round((time.time() - self.start_time) * 1000, 1)
        record = {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "meta": self.meta,
            "start_time": self.start_time,
            "duration_ms": duration_ms,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "num_steps": len(self.steps),
            "steps": self.steps,
        }
        self._append(record)
        return record

    @staticmethod
    def _append(record: dict):
        TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@contextmanager
def timed():
    t0 = time.time()
    result = {"ms": 0.0}
    try:
        yield result
    finally:
        result["ms"] = (time.time() - t0) * 1000


def load_traces(limit: int = None):
    if not TRACES_PATH.exists():
        return []
    with TRACES_PATH.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records[-limit:] if limit else records


def get_trace(trace_id: str):
    for r in load_traces():
        if r["trace_id"] == trace_id:
            return r
    return None


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((p / 100.0) * (len(values) - 1)))
    return values[k]


def ops_summary(kind: str = None):
    records = load_traces()
    if kind:
        records = [r for r in records if r["kind"] == kind]
    if not records:
        return {"count": 0}
    latencies = [r["duration_ms"] for r in records]
    costs = [r["total_cost_usd"] for r in records]
    return {
        "count": len(records),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
        },
        "cost_usd": {
            "total": round(sum(costs), 4),
            "avg": round(sum(costs) / len(costs), 6),
            "max": round(max(costs), 6),
        },
        "tokens": {
            "total_in": sum(r["total_tokens_in"] for r in records),
            "total_out": sum(r["total_tokens_out"] for r in records),
        },
    }