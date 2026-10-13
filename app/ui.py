"""
ClaimPilot Streamlit UI.

Three tabs:
  1. Policy Q&A    - natural-language questions over policy documents
  2. Claim Triage  - run the React agent on a claim, see decision + trace
  3. Evals         - run the 5-grader eval harness, see metrics

Run:
    streamlit run app/ui.py
"""

import html
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Make sure we can import sibling modules when Streamlit runs this file
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))



from upretreival import get_retriever
from answer import ask, CHAT_MODEL, TOP_K, MAX_TOKENS

load_dotenv()

# ---- Page config ----
st.set_page_config(
    page_title="ClaimPilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown(
    """
    <style>
    .stChatMessage { padding: 0.5rem 0.75rem; }

    .source-card {
        border-left: 3px solid #4f8cff;
        padding: 0.65rem 0.85rem;
        margin: 0.5rem 0;
        background: #f7f9fc;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .source-header {
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.15rem;
    }
    .source-meta {
        color: #555;
        font-size: 0.78rem;
        margin-bottom: 0.35rem;
    }
    .source-score {
        color: #888;
        font-size: 0.72rem;
    }
    .source-snippet {
        color: #333;
        font-size: 0.82rem;
        line-height: 1.35;
        margin-top: 0.4rem;
        padding: 0.5rem 0.6rem;
        background: #ffffff;
        border: 1px solid #e5e9f0;
        border-radius: 3px;
        white-space: pre-wrap;
        max-height: 180px;
        overflow-y: auto;
    }
    .decision-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        color: white;
        font-size: 1rem;
    }
    .badge-approve  { background: #16a34a; }
    .badge-reject   { background: #dc2626; }
    .badge-escalate { background: #ca8a04; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---- Cached retriever ----
@st.cache_resource(show_spinner="Loading retriever...")
def load_retriever():
    return get_retriever()


# ---- Session state ----
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "plan_filter" not in st.session_state:
        st.session_state.plan_filter = None
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


init_state()


# ---- Sidebar ----
with st.sidebar:
    st.title("ClaimPilot")
    st.caption("Ask questions about health insurance policy documents.")

    st.divider()

    st.subheader("Filters")
    retriever = load_retriever()

    plans = sorted({c["plan_name"] for c in retriever.chunks})
    plan_options = ["All plans"] + plans
    selected_plan = st.selectbox("Plan", plan_options, index=0, key="sb_plan")
    st.session_state.plan_filter = (
        None if selected_plan == "All plans" else selected_plan
    )

    doc_types = sorted({c["doc_type"] for c in retriever.chunks})
    selected_doc_type = st.selectbox(
        "Document type", ["All types"] + doc_types, index=0, key="sb_doc_type"
    )

    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Chunks to retrieve (top-K)", 3, 10, TOP_K, 1, key="sb_top_k")

    show_sources = st.checkbox("Show retrieved sources", value=True, key="sb_show_sources")
    if show_sources:
        show_scores = st.checkbox("Show similarity scores", value=False, key="sb_show_scores")
        show_snippets = st.checkbox("Show chunk text", value=True, key="sb_show_snippets")
    else:
        show_scores = False
        show_snippets = False

    st.divider()

    st.subheader("About")
    st.caption(
        f"**Model:** `{CHAT_MODEL}`\n\n"
        f"**Index:** {len(retriever.chunks):,} chunks\n\n"
        f"**Retrieval:** BM25 + dense embeddings"
    )

    if st.button("Clear conversation", use_container_width=True, key="sb_clear"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# ---- Main area ----
st.title("ClaimPilot")
st.caption(
    "Enterprise AI for insurance claims. Policy Q&A, claim triage, and evals - "
    "all grounded in real policy documents."
)

tab_qa, tab_triage, tab_evals = st.tabs(["Policy Q&A", "Claim Triage", "Evals"])


# ---- Helper: render a source card ----
def render_source_card(index, source, show_scores, show_snippet, snippet_chars=400):
    plan_name = html.escape(str(source.get("plan_name", "?")))
    page = html.escape(str(source.get("page", "?")))
    doc_type = html.escape(str(source.get("doc_type", "?")))
    section = html.escape(str(source.get("section", "?")))
    score_val = source.get("score", "?")

    score_line = (
        "<div class='source-score'>score " + html.escape(str(score_val)) + "</div>"
        if show_scores
        else ""
    )

    snippet_html = ""
    if show_snippet:
        text = (source.get("text") or "").strip()
        if text:
            if len(text) > snippet_chars:
                shown = text[:snippet_chars].rstrip() + "..."
            else:
                shown = text
            shown = html.escape(shown)
            shown = shown.replace("\n", "<br>")
            snippet_html = "<div class='source-snippet'>" + shown + "</div>"
        else:
            snippet_html = (
                "<div class='source-missing'>(chunk text not available)</div>"
            )

    card = (
        "<div class='source-card'>"
        + "<div class='source-header'>[" + str(index) + "] "
        + plan_name + " &middot; p" + page + "</div>"
        + "<div class='source-meta'>" + doc_type + " &middot; " + section + "</div>"
        + score_line
        + snippet_html
        + "</div>"
    )

    st.markdown(card, unsafe_allow_html=True)


# ============================================================
# TAB 1: Policy Q&A
# ============================================================
with tab_qa:
    st.caption(
        "Grounded answers from policy documents, with citations. "
        "Every answer links back to a source page."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources") and show_sources:
                with st.expander(
                    f"Sources ({len(msg['sources'])} chunks)", expanded=False
                ):
                    for i, s in enumerate(msg["sources"], 1):
                        render_source_card(i, s, show_scores, show_snippets)

    question = None

    if st.session_state.get("pending_question"):
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if not question:
        question = st.chat_input(
            "Ask about waiting periods, exclusions, claim procedures..."
        )

    if not st.session_state.messages and not question:
        st.divider()
        st.markdown("#### Try one of these questions")
        st.caption("Click any question to run it instantly.")

        sample_questions = [
            "What is the waiting period for pre-existing diseases?",
            "How long is the waiting period for maternity cover?",
            "What is the grace period for renewal?",
            "Is AYUSH treatment covered?",
            "What is the room rent limit?",
            "Are pre-existing diseases excluded?",
            "What documents are required for claim settlement?",
            "What is a pre-existing disease?",
        ]

        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        filters = {}
        if st.session_state.plan_filter:
            filters["plan_name"] = st.session_state.plan_filter
        if selected_doc_type != "All types":
            filters["doc_type"] = selected_doc_type

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                try:
                    result = ask(
                        question,
                        top_k=top_k,
                        filters=filters if filters else None,
                    )
                    answer_text = result["answer"]
                    sources = result.get("sources", [])
                    warning = result.get("warning")
                except Exception as e:
                    answer_text = (
                        f"**Error:** `{type(e).__name__}` — {e}\n\n"
                        "This may be a rate limit or an API issue. Try again in a moment."
                    )
                    sources = []
                    warning = "ui_error"

            st.markdown(answer_text)

            if warning:
                warning_messages = {
                    "no_chunks_retrieved": (
                        "**No matching passages found.** "
                        "Try rephrasing your question, or switch the **Plan** "
                        "filter to **All plans** to widen the search."
                    ),
                    "reasoning_only": (
                        "The model ran out of tokens while thinking and "
                        "didn't write a final answer. Try again, or ask a "
                        "shorter question."
                    ),
                    "hit_token_limit": (
                        "The answer was cut off because it hit the token "
                        "limit. Try a more specific question."
                    ),
                    "content_filter": (
                        "The response was filtered by content moderation. "
                        "Try rephrasing the question."
                    ),
                    "empty_content": (
                        "The model returned an empty response. "
                        "Try again in a moment."
                    ),
                    "api_error": (
                        "**API error.** This is usually a rate limit or "
                        "network issue. Wait a moment and try again."
                    ),
                    "no_choices": (
                        "The model returned no choices. Try again in a moment."
                    ),
                    "ui_error": (
                        "Something went wrong while generating the answer."
                    ),
                }
                message = warning_messages.get(warning, f"Unknown warning: `{warning}`")
                st.warning(message)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
            }
        )


# ============================================================
# TAB 2: Claim Triage
# ============================================================
with tab_triage:
    st.subheader("Triage a claim")
    st.caption(
        "Enter a claim ID (CLM-XXXX). The agent will search policies, "
        "check fraud signals, and return a decision with citations."
    )

    from agents import triage
    from tracer import get_trace

    col1, col2 = st.columns([3, 1])
    with col1:
        claim_id_input = st.text_input(
            "Claim ID", value="CLM-0001", key="triage_claim_id"
        )
    with col2:
        st.write("")
        st.write("")
        run_triage = st.button(
            "Run triage", use_container_width=True, key="btn_run_triage"
        )

    if run_triage and claim_id_input:
        with st.spinner(
            "Running agent on " + claim_id_input + "... this takes ~30-60s on the free model."
        ):
            try:
                result = triage(claim_id_input, verbose=False)
                d = result["decision"]

                decision_value = d["decision"]
                badge_class = {
                    "approve": "badge-approve",
                    "reject": "badge-reject",
                    "escalate": "badge-escalate",
                }.get(decision_value, "badge-escalate")

                st.markdown(
                    "<div class='decision-badge " + badge_class + "'>"
                    + decision_value.upper() + "</div>",
                    unsafe_allow_html=True,
                )

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Confidence", f"{d.get('confidence', 0):.2f}")
                col_b.metric("Cost", "$" + f"{result.get('cost_usd', 0):.5f}")
                col_c.metric(
                    "Duration", f"{result.get('duration_ms', 0) / 1000:.1f}s"
                )
                col_d.metric("Steps", result.get("steps", 0))

                st.markdown("**Rationale**")
                st.info(d.get("rationale", ""))

                if d.get("escalation_reason"):
                    st.warning("Escalation reason: " + str(d["escalation_reason"]))

                cits = d.get("citations", [])
                if cits:
                    with st.expander(
                        "Citations (" + str(len(cits)) + ")", expanded=True
                    ):
                        for c in cits:
                            st.markdown(
                                "- `" + str(c.get("chunk_id", "?")) + "` — "
                                "**" + str(c.get("source_pdf", "?")) + "** p"
                                + str(c.get("page", "?")) + " ("
                                + str(c.get("section", "?")) + ")"
                            )

                trace_id = result.get("trace_id")
                if trace_id:
                    with st.expander(
                        "Full trace: `" + str(trace_id) + "`", expanded=False
                    ):
                        tr = get_trace(trace_id)
                        if tr:
                            st.json(tr)
                        else:
                            st.write("Trace not found.")

            except FileNotFoundError:
                st.error(
                    "Claim `" + claim_id_input + "` not found in `data/claims/`. "
                    "Check the ID."
                )
            except Exception as e:
                st.error("Error: `" + type(e).__name__ + "` — " + str(e))


# ============================================================
# TAB 3: Evals
# ============================================================
with tab_evals:
    st.subheader("Eval Harness")
    st.caption(
        "Run the 5-grader eval harness against the frozen dataset. "
        "Graders: decision match, citation recall, citation grounding, "
        "rationale quality (LLM-judge), cost/latency."
    )
    
    from run import run_eval, load_report, list_recent_runs, load_cases

    all_cases = load_cases()
    st.metric("Frozen dataset", str(len(all_cases)) + " cases")

    col1, col2 = st.columns([1, 1])
    with col1:
        limit = st.slider(
            "Cases to run",
            1,
            max(1, len(all_cases)),
            3,
            1,
            key="eval_limit",
        )
    with col2:
        st.write("")
        st.write("")
        run_evals = st.button(
            "Run " + str(limit) + " cases",
            use_container_width=True,
            key="btn_run_evals",
        )

    if run_evals:
        progress = st.progress(0.0, text="Starting...")
        status_line = st.empty()
        results_holder = []

        def on_progress(done, total, claim_id, passed):
            progress.progress(
                done / total,
                text="Running case " + str(done) + "/" + str(total),
            )
            flag = "PASS" if passed else "FAIL"
            status_line.write(
                "[" + flag + "] " + claim_id + " (" + str(done) + "/" + str(total) + ")"
            )

        with st.spinner(
            "Running " + str(limit) + " eval cases... this takes ~30-60s per case on the free model."
        ):
            try:
                summary, results = run_eval(
                    limit=limit, progress_callback=on_progress
                )
                results_holder = results
                progress.progress(1.0, text="Done")
                st.success("Completed " + str(summary["total"]) + " cases.")
            except Exception as e:
                st.error("Eval failed: `" + type(e).__name__ + "` — " + str(e))
                summary = None

        if summary:
            st.markdown("### Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Decision acc", f"{summary['decision_acc']:.1%}")
            m2.metric("Citation recall", f"{summary['citation_recall']:.1%}")
            m3.metric("Grounding", f"{summary['citation_grounding']:.1%}")
            m4.metric("Rationale avg", f"{summary['rationale_avg']:.2f} / 5")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Cost avg", "$" + f"{summary['cost_avg']:.5f}")
            m6.metric("Cost total", "$" + f"{summary['cost_total']:.4f}")
            m7.metric("Latency p50", f"{summary['latency_p50'] / 1000:.1f}s")
            m8.metric("Latency p95", f"{summary['latency_p95'] / 1000:.1f}s")

            fails = [
                r for r in results_holder
                if not r.get("graders", {}).get("decision", {}).get("passed")
            ]
            if fails:
                st.markdown("### Failures (" + str(len(fails)) + ")")
                for r in fails[:10]:
                    st.markdown(
                        "- **`" + str(r["claim_id"]) + "`** expected `"
                        + str(r["expected"]) + "`, got `"
                        + str(r.get("actual")) + "` — trace `"
                        + str(r.get("trace_id", "?")) + "`"
                    )

    st.divider()

    st.markdown("### Latest report (`evals/report.md`)")
    report_md = load_report()
    if report_md:
        with st.expander("View full report", expanded=False):
            st.markdown(report_md)
    else:
        st.info("No report yet. Run evals above to generate one.")

    st.markdown("### Recent runs")
    runs = list_recent_runs(limit=5)
    if runs:
        for r in runs:
            st.write("- `" + r.name + "` (" + str(r.stat().st_size) + " bytes)")
    else:
        st.write("No runs yet.")
