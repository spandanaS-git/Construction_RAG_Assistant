# -*- coding: utf-8 -*-
"""
pipeline.py — Phase 7 (Upgraded): Full RAG Orchestration

Wires together the upgraded retrieval and generation modules into a single
ask_question() function. The following are all loaded ONCE at module
import time (singleton pattern) to avoid repeated disk/network calls:

  _vector_store      — FAISS or Pinecone connection
  _chunk_corpus      — all Document chunks (for BM25 index)
  _retriever         — hybrid (BM25 + dense) or pure-dense fallback
  _reranker          — cross-encoder (loaded on first call)

Query flow:
  ask_question(query)
    │
    ├─ 1. Hybrid retrieve (BM25 + dense, RRF fusion) → TOP_K candidates
    ├─ 2. Re-rank with cross-encoder → RERANKER_TOP_N final docs
    ├─ 3. Generate answer (HuggingFace / Ollama / OpenAI)
    └─ 4. Return {"answer", "sources", "chunks"}

Streamlit app.py continues to call ask_question() with the same
interface as before — no changes needed to app.py.
"""

from typing import Any, Dict, List

from langchain.schema import Document

from config import print_config_summary, TOP_K, RERANKER_TOP_N
from embedder import load_vector_store, load_chunks
from generator import generate_answer
from hybrid_retriever import build_hybrid_retriever, retrieve
from reranker import rerank

# ══════════════════════════════════════════════════════════════════════════════
# Module-level singletons — loaded once at import
# ══════════════════════════════════════════════════════════════════════════════

# 1. Vector store (FAISS or Pinecone)
_vector_store = None
try:
    _vector_store = load_vector_store()
except FileNotFoundError as _e:
    print(f"[WARN] Vector store not found: {_e}")
    print("   Run: python build_index.py")
except Exception as _e:
    print(f"[WARN] Failed to load vector store: {_e}")

# 2. Chunk corpus for BM25
# Strategy (graceful degradation):
#   a) Try loading chunks.pkl (created by build_index.py with new code)
#   b) If not found + FAISS backend: extract chunks from FAISS docstore
#   c) If neither: BM25 unavailable, fall back to pure dense retrieval
_chunk_corpus: List[Document] = []

_chunks_from_file = load_chunks()
if _chunks_from_file is not None:
    _chunk_corpus = _chunks_from_file
    print(f"[INFO] BM25 corpus: {len(_chunk_corpus)} chunks (from chunks.pkl)")
elif _vector_store is not None:
    from config import VECTOR_BACKEND
    if VECTOR_BACKEND == "faiss":
        try:
            # Extract docs directly from FAISS in-memory docstore
            # This works with the existing index.faiss without any rebuild
            _chunk_corpus = list(_vector_store.docstore._dict.values())
            print(
                f"[INFO] BM25 corpus: {len(_chunk_corpus)} chunks "
                "(extracted from FAISS docstore — run build_index.py to persist)"
            )
        except Exception as _ex:
            print(f"[WARN] Could not extract chunks from FAISS docstore: {_ex}")
    if not _chunk_corpus:
        print(
            "[WARN] No chunk corpus found for BM25. "
            "Hybrid search will fall back to dense-only retrieval.\n"
            "   Fix: Run `python build_index.py` to persist chunks.pkl"
        )

# 3. Hybrid retriever (BM25 + dense, or pure dense if BM25 unavailable)
_retriever = None
if _vector_store is not None:
    _retriever = build_hybrid_retriever(
        _chunk_corpus,
        _vector_store,
        top_k=TOP_K,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def ask_question(query: str) -> Dict[str, Any]:
    """
    Full RAG pipeline: hybrid retrieve → re-rank → generate.

    This function never raises — all exceptions are caught and returned
    as error strings to keep both the Streamlit UI and FastAPI stable.

    Args:
        query: Natural-language question from the user.

    Returns:
        dict with exactly these keys:
            "answer"  (str)        — generated answer
            "sources" (list[dict]) — [{"file": str, "page": int}, ...]
            "chunks"  (list[str])  — raw chunk texts for UI display
    """
    if not query or not query.strip():
        return {"answer": "Please enter a question.", "sources": [], "chunks": []}

    if _retriever is None or _vector_store is None:
        return {
            "answer": (
                "Error: Vector store is not available. "
                "Run `python build_index.py` first."
            ),
            "sources": [],
            "chunks":  [],
        }

    try:
        # ── Step 1: Hybrid retrieval (BM25 + dense, RRF) ──────────────────
        candidate_docs: List[Document] = retrieve(query, _retriever)

        # ── Step 2: Cross-encoder re-ranking ──────────────────────────────
        final_docs: List[Document] = rerank(query, candidate_docs, top_n=RERANKER_TOP_N)

        # ── Step 3: Generate answer ────────────────────────────────────────
        answer: str = generate_answer(query, final_docs)

        # ── Step 4: Build return payload ───────────────────────────────────
        sources = [
            {
                "file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page",   0),
            }
            for doc in final_docs
        ]
        chunks = [doc.page_content for doc in final_docs]

        return {"answer": answer, "sources": sources, "chunks": chunks}

    except Exception as exc:
        return {"answer": f"Error: {exc}", "sources": [], "chunks": []}


# ── CLI smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_config_summary()

    test_queries = [
        "What are the inspection steps for a concrete bridge?",
        "What PPE is required on a construction site?",
        "What fall protection is required above 6 feet?",
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"\n{'='*65}")
        print(f"  Query {i}: {query}")
        print("=" * 65)
        result = ask_question(query)
        print(f"\n[ANSWER]:\n{result['answer']}")
        print(f"\n[SOURCES]:")
        for src in result["sources"]:
            print(f"   - {src['file']}  --  page {src['page']}")
        print(f"\n[CHUNKS] ({len(result['chunks'])} total):")
        for j, chunk in enumerate(result["chunks"], start=1):
            preview = chunk[:200].replace("\n", " ")
            print(f"   [{j}] {preview}...")
