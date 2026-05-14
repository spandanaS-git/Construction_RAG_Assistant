"""
app.py — Phase 8: Streamlit UI

Construction Knowledge Assistant — interactive RAG chatbot UI.
Loads the pipeline on startup; all subsequent queries are served from
the cached vector store (fully offline after build_index.py runs once).

Run:   streamlit run app.py
"""

import streamlit as st

from config import EMBEDDING_MODEL, LLM_MODEL, LLM_BACKEND
from pipeline import ask_question

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Knowledge Assistant",
    page_icon="🏗️",
    layout="wide",
)

# ── Source document list (for sidebar display) ─────────────────────────────────
SOURCE_DOCUMENTS = [
    "gtm_construction_safety_manual.pdf",
    "txdot_bridge_inspection_manual.pdf",
    "wsdot_construction_manual.pdf",
    "montana_dot_project_report.pdf",
    "osha_construction_safety_guide.pdf",
]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown(
    f"""
**Construction Knowledge Assistant**

A Retrieval-Augmented Generation (RAG) prototype for query-based
search across construction and safety documents.

---

**Active Configuration**

| Setting | Value |
|---|---|
| Embeddings | `{EMBEDDING_MODEL}` |
| LLM | `{LLM_MODEL}` |
| Backend | `{LLM_BACKEND}` |

---

**Source Documents**

"""
    + "\n".join(f"- `{doc}`" for doc in SOURCE_DOCUMENTS)
    + """

---

> **Note:** First run requires internet to download models.
> All subsequent runs are fully offline.
"""
)

# ──────────────────────────────────────────────────────────────────────────────
# Main Page
# ──────────────────────────────────────────────────────────────────────────────
st.title("Construction Knowledge Assistant 🏗️")
st.caption(
    f"Powered by: **{EMBEDDING_MODEL}** embeddings | **{LLM_MODEL}** | "
    f"Backend: **{LLM_BACKEND}**"
)

st.divider()

# ── Query input ────────────────────────────────────────────────────────────────
query = st.text_input(
    label="Ask a question about the construction documents:",
    placeholder="e.g. What PPE is required on a construction site?",
    key="query_input",
)

ask_button = st.button("Ask", type="primary", use_container_width=False)

# ── Query handling ─────────────────────────────────────────────────────────────
if ask_button:
    if not query or not query.strip():
        st.warning("⚠️ Please enter a question before clicking Ask.")
    else:
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                result = ask_question(query)

                # ── Answer display ─────────────────────────────────────────
                if result["answer"].startswith("Error:"):
                    st.error(f"Pipeline error: {result['answer']}")
                else:
                    st.success(result["answer"])

                # ── Source chunks display ──────────────────────────────────
                if result["chunks"]:
                    st.subheader("📄 Retrieved Source Chunks")
                    for i, (chunk_text, source_info) in enumerate(
                        zip(result["chunks"], result["sources"]), start=1
                    ):
                        label = (
                            f"Rank {i} — Source: {source_info['file']} "
                            f"— Page {source_info['page']}"
                        )
                        with st.expander(label):
                            st.markdown(chunk_text)

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")

# ── Footer hint ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "💡 Tip: Try asking about bridge inspection steps, PPE requirements, "
    "or safety audit procedures."
)
