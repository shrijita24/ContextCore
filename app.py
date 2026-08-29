"""
ContextCore — Live Demo
A persistent personal context layer that grounds AI conversations in
verified user truth, and flags when AI contradicts it.

Runs against the real schema.py, chunker.py, vector_store.py, rag_pipeline.py,
and contradiction.py in src/.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from schema import TruthDocument
from chunker import chunk_truth_document

st.set_page_config(page_title="ContextCore — Live Demo", page_icon="🧠", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "truth_document.json")

# ---------------------------------------------------------------------------
# Styling — accent colour matches the existing brand (pink/red), just applied
# consistently: card borders, pill badges, section headers, verdict states.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .cc-hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255,75,110,0.15), rgba(255,75,110,0.03));
        border: 1px solid rgba(255,75,110,0.25);
        margin-bottom: 1.5rem;
    }
    .cc-hero h1 { margin: 0 0 0.35rem 0; font-size: 2rem; }
    .cc-hero p { margin: 0; opacity: 0.85; font-size: 1rem; line-height: 1.5; }

    .cc-pill-row { margin-top: 0.9rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .cc-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(255,75,110,0.12);
        border: 1px solid rgba(255,75,110,0.3);
        color: #ff8fa3;
    }

    .cc-card {
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        background: rgba(255,255,255,0.02);
    }
    .cc-card .cc-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.55;
        margin-bottom: 0.3rem;
    }

    .cc-verdict-ok {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.4);
        color: #7be3a5;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .cc-verdict-bad {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        background: rgba(255, 75, 110, 0.14);
        border: 1px solid rgba(255, 75, 110, 0.45);
        color: #ff8fa3;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .cc-section-title { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="cc-hero">
  <h1>🧠 ContextCore</h1>
  <p>A persistent personal context layer that grounds AI conversations in verified
  user truth — and flags when AI contradicts it.</p>
  <div class="cc-pill-row">
    <span class="cc-pill">LangChain</span>
    <span class="cc-pill">ChromaDB</span>
    <span class="cc-pill">Groq</span>
    <span class="cc-pill">RAG</span>
    <span class="cc-pill">LLM-as-Judge</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📍 Project status / roadmap", expanded=False):
    st.markdown("""
| Stage | Status |
|---|---|
| Architecture design | ✅ Done |
| Truth document schema (Pydantic validation) | ✅ Done |
| RAG chunker | ✅ Done |
| Vector store (ChromaDB + local embeddings) | ✅ Done |
| Semantic search over truth document | ✅ Done |
| LangChain retrieval chain (full conversational RAG) | ✅ Done |
| Contradiction detection (LLM-as-judge) | ✅ Done |
| Multi-user support | 🔭 Planned |
""")

# ---- Load & validate truth document ----
try:
    doc = TruthDocument.from_json(DATA_PATH)
    st.success(f"✅ Truth document loaded and validated — owner: {doc.identity.name}")
except Exception as e:
    st.error(f"Failed to load/validate truth document: {e}")
    st.stop()

chunks = chunk_truth_document(doc)


def has_key() -> bool:
    return bool(
        os.environ.get("GROQ_API_KEY")
        or (st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None)
    )


def ensure_key_in_env():
    os.environ.setdefault("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))


def no_key_warning(feature: str):
    st.warning(
        f"🔑 No Groq API key found. Add `GROQ_API_KEY` to this app's Streamlit Cloud "
        f"secrets (Settings → Secrets) to enable {feature}. "
        f"Free key at https://console.groq.com — no credit card required."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_doc, tab_chunks, tab_ask, tab_check = st.tabs([
    "📄 Truth Document",
    "🧩 RAG Chunker",
    "💬 Ask AI",
    "🔎 Contradiction Check",
])

with tab_doc:
    st.markdown('<div class="cc-section-title"><h3>Truth Document</h3></div>', unsafe_allow_html=True)
    st.caption("The single verified source of truth everything else in this app is grounded in.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div class="cc-card">
  <div class="cc-label">Identity</div>
  <b>{doc.identity.name}</b> — {doc.identity.role}<br/>
  {doc.identity.institution}, {doc.identity.location}<br/>
  CGPA {doc.identity.cgpa}
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="cc-card">
  <div class="cc-label">Immediate Goal</div>
  {doc.goals.immediate}
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="cc-card">
  <div class="cc-label">Current Focus</div>
  {doc.ongoing.current_focus}
</div>
""", unsafe_allow_html=True)

        if doc.goals.tracks:
            tracks_html = " ".join(f'<span class="cc-pill">{t}</span>' for t in doc.goals.tracks)
            st.markdown(f"""
<div class="cc-card">
  <div class="cc-label">Target Tracks</div>
  <div class="cc-pill-row">{tracks_html}</div>
</div>
""", unsafe_allow_html=True)

with tab_chunks:
    st.markdown('<div class="cc-section-title"><h3>RAG Chunker Output</h3></div>', unsafe_allow_html=True)
    st.caption(f"Generated {len(chunks)} retrieval-ready chunks from the truth document.")

    for c in chunks:
        st.markdown(f"""
<div class="cc-card">
  <div class="cc-label">{c.section}</div>
  {c.text}
</div>
""", unsafe_allow_html=True)

with tab_ask:
    st.markdown('<div class="cc-section-title"><h3>Conversational RAG (grounded Q&A)</h3></div>', unsafe_allow_html=True)
    st.caption("Ask a question — LangChain retrieves the relevant truth-document chunks via ChromaDB and generates an answer constrained to them.")

    if not has_key():
        no_key_warning("live conversational RAG")
    else:
        ensure_key_in_env()
        from vector_store import index_chunks
        from rag_pipeline import answer as rag_answer

        query = st.text_input("Ask something about the truth document", placeholder="e.g. What is her CGPA and where does she study?")
        run = st.button("💬 Ask", type="primary")

        if run and query.strip():
            with st.spinner("Indexing + retrieving + generating..."):
                try:
                    index_chunks(chunks)
                    result = rag_answer(query, top_k=3)

                    st.markdown("**Answer**")
                    st.markdown(f'<div class="cc-card">{result["answer"]}</div>', unsafe_allow_html=True)

                    with st.expander("📎 Grounded in these chunks"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}** &nbsp;·&nbsp; section: `{src.metadata.get('section')}`")
                            st.write(src.page_content)
                except Exception as e:
                    st.error(f"RAG pipeline failed: {e}")

with tab_check:
    st.markdown('<div class="cc-section-title"><h3>Contradiction Detection</h3></div>', unsafe_allow_html=True)
    st.caption("Paste a statement an AI assistant made about you — this checks it against the truth document and flags conflicts.")

    if not has_key():
        no_key_warning("contradiction detection")
    else:
        ensure_key_in_env()
        from vector_store import index_chunks as _index_chunks_cd
        from contradiction import check_contradiction

        statement = st.text_area(
            "Statement to check",
            placeholder="e.g. She studies at IIT Bombay and has a CGPA of 7.1.",
            height=80,
        )
        check = st.button("🔎 Check for contradictions", type="primary")

        if check and statement.strip():
            with st.spinner("Retrieving relevant facts + judging..."):
                try:
                    _index_chunks_cd(chunks)
                    result = check_contradiction(statement, top_k=3)

                    if result["contradicts"]:
                        st.markdown(
                            f'<div class="cc-verdict-bad">⚠️ Contradiction detected · confidence {result["confidence"]:.0%}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="cc-verdict-ok">✓ Consistent with truth document · confidence {result["confidence"]:.0%}</div>',
                            unsafe_allow_html=True,
                        )

                    st.write(result["explanation"])

                    with st.expander("📎 Evidence used for this verdict"):
                        for i, ev in enumerate(result["evidence"], 1):
                            st.markdown(f"**Source {i}** &nbsp;·&nbsp; section: `{ev.metadata.get('section')}`")
                            st.write(ev.page_content)
                except Exception as e:
                    st.error(f"Contradiction check failed: {e}")

st.divider()
st.caption("Built by [Shrijita Bhattacharyya](https://github.com/shrijita24) · [Repo](https://github.com/shrijita24/ContextCore)")
