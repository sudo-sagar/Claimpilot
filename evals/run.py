"""
UI-friendly wrapper around the eval harness.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

ROOT = Path(__file__).parent.parent

DATASET = ROOT / "evals" / "dataset.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
REPORT = ROOT / "evals" / "report.md"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"


def load_chunks_lookup():
    lookup = {}
    if not CHUNKS_PATH.exists():
        return lookup
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            lookup[c["chunk_id"]] = c
    return lookup


def load_cases(limit=None):
    if not DATASET.exists():
        return []
    cases = [json.loads(l) for l in DATASET.open(encoding="utf-8")]
    return cases[:limit] if limit else cases


def run_eval(limit=None, progress_callback=None):
    from agents import triage
    from graders import (
        grade_decision,
        grade_citation_recall,
        grade_citation_grounding,
        grade_rationale,
        grade_cost,
    )

    cases = load_cases(limit=limit)
    if not cases:
        return {}, []

    chunks_lookup = load_chunks_lookup()
    results = []

    for i, case in enumerate(cases, 1):
        try:
            result = triage(case["claim_id"], verbose=False)
        except Exception as e:
            case_result = {
                "claim_id": case["claim_id"],
                "expected": case["expected_decision"],
                "error": type(e).__name__ + ": " + str(e)[:200],
                "graders": {},
                "passed": False,
            }
        else:
            graders = {
                "decision": grade_decision(case, result),
                "citation_recall": grade_citation_recall(case, result),
                "citation_grounding": grade_citation_grounding(case, result, chunks_lookup),
                "rationale": grade_rationale(case, result),
                "cost": grade_cost(case, result),
            }
            case_result = {
                "claim_id": case["claim_id"],
                "expected": case["expected_decision"],
                "actual": result["decision"]["decision"],
                "rationale": result["decision"].get("rationale", ""),
                "trace_id": result.get("trace_id"),
                "cost_usd": result.get("cost_usd", 0),
                "duration_ms": result.get("duration_ms", 0),
                "graders": graders,
            }

        results.append(case_result)

        if progress_callback:
            progress_callback(
                i,
                len(cases),
                case["claim_id"],
                case_result.get("graders", {}).get("decision", {}).get("passed", False),
            )

    summary = summarize(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / (run_id + ".jsonl")
    with results_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_report(summary, results, REPORT)

    return summary, results


def summarize(results):
    n = len(results)
    if n == 0:
        return {}

    def pct(key):
        return sum(1 for r in results if r.get("graders", {}).get(key, {}).get("passed")) / n

    latencies = sorted([r.get("duration_ms", 0) for r in results])
    return {
        "total": n,
        "decision_acc": pct("decision"),
        "citation_recall": pct("citation_recall"),
        "citation_grounding": pct("citation_grounding"),
        "rationale_avg": sum(r.get("graders", {}).get("rationale", {}).get("score", 0) for r in results) / n,
        "cost_avg": sum(r.get("cost_usd", 0) for r in results) / n,
        "cost_total": sum(r.get("cost_usd", 0) for r in results),
        "latency_p50": latencies[n // 2] if latencies else 0,
        "latency_p95": latencies[int(n * 0.95)] if n > 1 else (latencies[0] if latencies else 0),
    }


def write_report(summary, results, path):
    lines = []
    lines.append("# ClaimPilot Eval Report")
    lines.append("")
    lines.append("_Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append("| Cases | " + str(summary["total"]) + " |")
    lines.append("| Decision accuracy | " + f"{summary['decision_acc']:.1%}" + " |")
    lines.append("| Citation recall | " + f"{summary['citation_recall']:.1%}" + " |")
    lines.append("| Citation grounding | " + f"{summary['citation_grounding']:.1%}" + " |")
    lines.append("| Rationale (avg / 5) | " + f"{summary['rationale_avg']:.2f}" + " |")
    lines.append("| Cost / case (avg) | $" + f"{summary['cost_avg']:.5f}" + " |")
    lines.append("| Latency p50 | " + str(summary["latency_p50"]) + " ms |")
    lines.append("| Latency p95 | " + str(summary["latency_p95"]) + " ms |")
    lines.append("")

    lines.append("## Decision accuracy by expected class")
    lines.append("")
    lines.append("| Expected | Total | Correct | Accuracy |")
    lines.append("|---|---|---|---|")
    by_class = {}
    for r in results:
        exp = r["expected"]
        by_class.setdefault(exp, {"total": 0, "correct": 0})
        by_class[exp]["total"] += 1
        if r.get("graders", {}).get("decision", {}).get("passed"):
            by_class[exp]["correct"] += 1
    for exp, stats in by_class.items():
        acc = stats["correct"] / max(stats["total"], 1)
        lines.append("| " + exp + " | " + str(stats["total"]) + " | " + str(stats["correct"]) + " | " + f"{acc:.1%}" + " |")
    lines.append("")

    fails = [r for r in results if not r.get("graders", {}).get("decision", {}).get("passed")]
    lines.append("## Failures (" + str(len(fails)) + ")")
    lines.append("")
    for r in fails[:15]:
        lines.append("- **" + r["claim_id"] + "** — expected `" + r["expected"] + "`, got `" + str(r.get("actual")) + "`")
        if r.get("rationale"):
            lines.append("  - Rationale: _" + str(r["rationale"])[:200] + "_")
        if r.get("trace_id"):
            lines.append("  - Trace: `" + str(r["trace_id"]) + "`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def load_report():
    if not REPORT.exists():
        return None
    return REPORT.read_text(encoding="utf-8")


def list_recent_runs(limit=10):
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]