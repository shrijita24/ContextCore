"""
ContextCore — Live Demo
A persistent personal context layer that grounds AI conversations in
verified user truth, and flags when AI contradicts it.

Runs against the real schema.py, chunker.py, and vector_store.py in src/.
"""

import os
import sys
import json

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from schema import TruthDocument
from chunker import chunk_truth_document

st.set_page_config(page_title="ContextCore — Live Demo", page_icon="🧠", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "truth_document.json")

st.title("🧠 ContextCore")
st.caption("A persistent personal context layer that grounds AI conversations in verified user truth — and flags when AI contradicts it.")

with st.expander("📍 Project status / roadmap", expanded=False):
    st.markdown("""
| Stage | Status |
|---|---|
| Architecture design | ✅ Done |
| Truth document schema (Pydantic validation) | ✅ Done |
| RAG chunker | ✅ Done |
| Vector store (ChromaDB + OpenAI embeddings) | ✅ Done |
| Semantic search over truth document | ✅ Done |
| LangChain retrieval chain (full conversational RAG) | ✅ Done |
| Contradiction detection (LLM-as-judge) | 🚧 In progress |
| Multi-user support | 🔭 Planned |
""")

# ---- Load & validate truth document ----
try:
    doc = TruthDocument.from_json(DATA_PATH)
    st.success(f"✅ Truth document loaded and validated — owner: {doc.identity.name}")
except Exception as e:
    st.error(f"Failed to load/validate truth document: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Truth Document")
    with st.container(border=True):
        st.markdown(f"**{doc.identity.name}** — {doc.identity.role}")
        st.markdown(f"{doc.identity.institution}, {doc.identity.location} · CGPA {doc.identity.cgpa}")
        st.markdown(f"**Immediate goal:** {doc.goals.immediate}")
        st.markdown(f"**Current focus:** {doc.ongoing.current_focus}")

    st.subheader("2. RAG Chunker Output")
    chunks = chunk_truth_document(doc)
    st.caption(f"Generated {len(chunks)} retrieval-ready chunks from the truth document")

    for c in chunks:
        with st.container(border=True):
            st.markdown(f"**`{c.section}`**")
            st.write(c.text)

with col2:
    st.subheader("3. Conversational RAG (grounded Q&A)")
    st.caption("Ask a question — LangChain retrieves the relevant truth-document chunks via ChromaDB and generates an answer constrained to them.")

    has_key = bool(os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else os.environ.get("GROQ_API_KEY"))

    if not has_key:
        st.warning(
            "🔑 No Groq API key found. Add `GROQ_API_KEY` to this app's Streamlit Cloud "
            "secrets (Settings → Secrets) to enable live conversational RAG. "
            "Free key at https://console.groq.com — no credit card required."
        )
    else:
        os.environ.setdefault("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
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
                    with st.container(border=True):
                        st.write(result["answer"])

                    with st.expander("📎 Grounded in these chunks"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {i}** &nbsp;·&nbsp; section: `{src.metadata.get('section')}`")
                            st.write(src.page_content)
                except Exception as e:
                    st.error(f"RAG pipeline failed: {e}")

st.divider()
st.caption("Built by [Shrijita Bhattacharyya](https://github.com/shrijita24) · [Repo](https://github.com/shrijita24/ContextCore) · Contradiction detection coming next.")
