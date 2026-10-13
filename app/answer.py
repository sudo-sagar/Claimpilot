"""
ClaimPilot answer generation.

Takes a natural-language question, retrieves the top-K chunks
using the hybrid retriever, builds a grounded prompt, and calls an
LLM (via OpenRouter) to synthesize a final answer with citations.

Usage:
    python app/answer.py
    python app/answer.py "What is the waiting period for pre-existing diseases?"
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from upretreival import get_retriever

load_dotenv()

ROOT = Path(__file__).parent.parent

# ---- Generation configuration ----
CHAT_MODEL = "nvidia/nemotron-3.5-lightning:free"   # reasoning model
TOP_K = 5
MAX_TOKENS = 3500
TEMPERATURE = 0.1
DEBUG = False

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["API_KEY"],
)

# ---- System prompt ----
SYSTEM_PROMPT = """You are ClaimPilot, an assistant that answers questions about health insurance policies.

RULES:
1. Answer ONLY using the CONTEXT chunks provided below.
2. If the context does not contain the answer, say: "I couldn't find this in the provided policy documents."
3. Be concise. Give the direct answer first, then supporting detail.
4. Cite EVERY claim using the EXACT citation shown in each "SOURCE" line, e.g. [Optima Enhance, p4] or [My Optima Secure, p28, PRE-EXISTING DISEASES].
   - Do NOT cite using chunk numbers like "[Chunk 1]" or "[SOURCE 2]" — those are internal labels only.
   - Do NOT use "[Def. 43]" or "[Chunk N, Def. X]" formats.
   - Always use the format [Plan Name, pPage] or [Plan Name, pPage, Section].
5. If different plans have different answers, list them separately.
6. Do NOT invent numbers, durations, or clause names. If a number is not in the context, do not state one.
7. Do NOT use prior knowledge about insurance. Only use the CONTEXT.
8. Keep your final answer under 250 words. Do not write a long "thinking process" — just give the answer.
"""


def build_context(chunks):
    """
    Format retrieved chunks into a context block.
    Each chunk header embeds the citation inline so the model
    reuses it verbatim instead of inventing '[Chunk N]' labels.
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        # Citation embedded right in the header
        citation = "[" + c["plan_name"] + ", p" + str(c["page"]) + "]"
        header = (
            "SOURCE " + str(i) + " — cite this as " + citation + "\n"
            + "Section: " + c["section"] + " | Doc type: " + c["doc_type"]
        )
        lines.append(header)
        lines.append(c["text"].strip())
        lines.append("")  # blank line between chunks
    return "\n".join(lines)


def build_user_prompt(question, chunks):
    """Combine question + context into the user message."""
    context = build_context(chunks)
    return (
        "QUESTION:\n" + question + "\n\n"
        "CONTEXT:\n" + context + "\n"
        "Answer the question using only the context above. "
        "Cite sources using the EXACT citation from each SOURCE header, "
        "e.g. [Optima Enhance, p4]. Do NOT use [Chunk N] labels. "
        "Keep it short — just the answer, no thinking process."
    )


def ask(question, top_k=TOP_K, filters=None, verbose=False):
    """
    End-to-end: retrieve -> prompt -> generate.
    Returns a dict with 'answer', 'sources', 'chunks', 'usage', 'warning'.

    Handles these failure modes gracefully:
      - No chunks retrieved
      - Model returns empty content
      - Model returns reasoning but no final answer
      - Model hits token limit during reasoning
      - Model returns no choices at all
      - API errors (network, auth, rate limit)
    """
    retriever = get_retriever()

    # 1. Retrieve
    chunks = retriever.search(question, top_k=top_k, filters=filters)
    if not chunks:
        return {
            "answer": "No relevant chunks found for this question.",
            "sources": [],
            "chunks": [],
            "usage": None,
            "warning": "no_chunks_retrieved",
            
        }

    # 2. Build prompt
    user_prompt = build_user_prompt(question, chunks)

    if verbose or DEBUG:
        print("=" * 78)
        print("SYSTEM PROMPT")
        print("=" * 78)
        print(SYSTEM_PROMPT)
        print("=" * 78)
        print("USER PROMPT")
        print("=" * 78)
        print(user_prompt)
        print("=" * 78)

    # 3. Call the model
    warning = None
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as e:
        return {
            "answer": "[API ERROR] " + type(e).__name__ + ": " + str(e),
            "sources": [],
            "chunks": chunks,
            "usage": None,
            "warning": "api_error",
        }

    # 4. Extract answer defensively
    if not getattr(resp, "choices", None):
        answer = "[Model returned no choices. Full response: " + str(resp) + "]"
        warning = "no_choices"
    else:
        choice = resp.choices[0]
        msg = choice.message
        finish_reason = getattr(choice, "finish_reason", "unknown")

        content = getattr(msg, "content", None)
        answer = (content or "").strip()

        if not answer:
            reasoning = getattr(msg, "reasoning", None)

            if reasoning:
                answer = (
                    "[WARNING] Model produced reasoning but no final answer. "
                    "finish_reason=" + str(finish_reason) + ". "
                    "Try increasing MAX_TOKENS or switching to a non-reasoning model.\n\n"
                    "--- Partial reasoning (NOT a final answer) ---\n"
                    + reasoning.strip()
                )
                warning = "reasoning_only"

            elif finish_reason == "length":
                answer = (
                    "[WARNING] Model hit token limit (" + str(MAX_TOKENS)
                    + ") before producing an answer. "
                    "Increase MAX_TOKENS or switch to a non-reasoning model."
                )
                warning = "hit_token_limit"

            elif finish_reason == "content_filter":
                answer = (
                    "[WARNING] Model response was filtered by content moderation. "
                    "Rephrase the question or try a different model."
                )
                warning = "content_filter"

            else:
                answer = (
                    "[WARNING] Model returned empty content. "
                    "finish_reason=" + str(finish_reason) + ". "
                    "Try a different model or check the OpenRouter dashboard."
                )
                warning = "empty_content"

    # 5. Package sources
    sources = []
    for c in chunks:
        sources.append(
            {
                "plan_name": c["plan_name"],
                "doc_type": c["doc_type"],
                "page": c["page"],
                "section": c["section"],
                "chunk_id": c["chunk_id"],
                "score": c["score"],
                "text": c["text"],
            }
        )

    # 6. Usage info
    usage = None
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
            "total_tokens": getattr(resp.usage, "total_tokens", 0),
        }

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "usage": usage,
        "warning": warning,
    }


def print_result(result, show_chunks=False):
    """Pretty-print the answer + sources."""
    print("\n" + "=" * 78)
    print("ANSWER")
    print("=" * 78)
    print(result["answer"])

    if result.get("warning"):
        print("\n[warning flag: " + result["warning"] + "]")

    print("\n" + "-" * 78)
    print("SOURCES (retrieved chunks)")
    print("-" * 78)
    for i, s in enumerate(result["sources"], 1):
        print(
            str(i) + ". ["
            + s["doc_type"] + "] "
            + s["plan_name"]
            + " p" + str(s["page"])
            + " (" + s["section"] + ")"
            + "  score=" + str(s["score"])
        )

    if show_chunks:
        print("\n" + "-" * 78)
        print("FULL CHUNK TEXT")
        print("-" * 78)
        for i, c in enumerate(result["chunks"], 1):
            print("\n[" + str(i) + "] "
                  + c["plan_name"] + " p" + str(c["page"])
                  + " (" + c["section"] + ")")
            print(c["text"][:600].replace("\n", " "))
            if len(c["text"]) > 600:
                print("...")

    if result["usage"]:
        u = result["usage"]
        print("\n" + "-" * 78)
        print("USAGE: prompt=" + str(u["prompt_tokens"])
              + " completion=" + str(u["completion_tokens"])
              + " total=" + str(u["total_tokens"]))
        print("-" * 78)


# ---- Default demo queries ----
DEMO_QUERIES = [
    "What is the waiting period for pre-existing diseases?",
    "How long is the waiting period for maternity cover?",
    "What is the grace period for renewal?",
    "Is AYUSH treatment covered?",
    "What is the room rent limit?",
    "Are pre-existing diseases excluded?",
    "What documents are required for claim settlement?",
]


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = ask(query, top_k=TOP_K, verbose=True)
        print_result(result, show_chunks=True)
        return

    warnings = []
    for i, q in enumerate(DEMO_QUERIES, 1):
        print("\n\n" + "#" * 78)
        print("# QUERY " + str(i) + "/" + str(len(DEMO_QUERIES)) + ": " + q)
        print("#" * 78)
        result = ask(q, top_k=TOP_K)
        print_result(result, show_chunks=False)
        if result.get("warning"):
            warnings.append((i, q, result["warning"]))

    if warnings:
        print("\n\n" + "=" * 78)
        print("SUMMARY OF WARNINGS")
        print("=" * 78)
        for i, q, w in warnings:
            print("  Query " + str(i) + " [" + w + "]: " + q)
    else:
        print("\n\nAll queries completed without warnings.")


if __name__ == "__main__":
    main()