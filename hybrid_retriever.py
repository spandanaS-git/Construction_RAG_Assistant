# -*- coding: utf-8 -*-
"""
hybrid_retriever.py — Hybrid Search: BM25 (sparse) + Dense (vector)

Combines sparse keyword retrieval (BM25) with dense semantic retrieval
using LangChain's EnsembleRetriever, which merges results via
Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Dense retrieval is great at semantic similarity: "fall protection" ≈
    "personal fall arrest systems"
  - BM25 is great at exact-match: "OSHA 1926.502", regulation codes,
    proper nouns, part numbers
  - RRF fusion captures the best of both signals

Architecture:
    Query
      ├── BM25Retriever (TF-IDF sparse)  ──┐
      └── VectorStore.as_retriever (dense) ──┤ EnsembleRetriever (RRF)
                                            └── top-K fused results

Config knobs (all in .env):
  HYBRID_SEARCH_ENABLED  true | false
  HYBRID_BM25_WEIGHT     0.0–1.0 (default 0.4)
  HYBRID_DENSE_WEIGHT    0.0–1.0 (default 0.6)
  TOP_K                  candidate pool size
"""

from typing import List

from langchain.schema import Document

from config import (
    HYBRID_SEARCH_ENABLED,
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    TOP_K,
)


def build_hybrid_retriever(
    chunks: List[Document],
    vector_store,
    top_k: int = TOP_K,
):
    """
    Build a hybrid retriever combining BM25 and dense vector retrieval.

    If HYBRID_SEARCH_ENABLED=false, falls back to a pure dense retriever.
    If chunks list is empty, also falls back to pure dense retrieval.

    Args:
        chunks:       Full list of Document chunks (corpus for BM25).
                      Must be the same chunks used to build the vector store.
        vector_store: A loaded FAISS or PineconeVectorStore object.
        top_k:        Number of candidates to retrieve from each sub-retriever.
                      EnsembleRetriever pools both lists before RRF fusion.

    Returns:
        EnsembleRetriever (hybrid) or VectorStoreRetriever (fallback).
    """
    if not HYBRID_SEARCH_ENABLED:
        print("[RETRIEVER] Hybrid search DISABLED — using dense-only retriever.")
        return _build_dense_retriever(vector_store, top_k)

    if not chunks:
        print(
            "[RETRIEVER] No chunk corpus available for BM25 — "
            "falling back to dense-only retrieval.\n"
            "  Tip: Run `python build_index.py` to persist chunks.pkl."
        )
        return _build_dense_retriever(vector_store, top_k)

    # ── BM25 sparse retriever ──────────────────────────────────────────────
    from langchain_community.retrievers import BM25Retriever

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k
    print(f"[RETRIEVER] BM25 index built over {len(chunks)} chunks (k={top_k})")

    # ── Dense vector retriever ─────────────────────────────────────────────
    dense_retriever = _build_dense_retriever(vector_store, top_k)

    # ── EnsembleRetriever: Reciprocal Rank Fusion ──────────────────────────
    from langchain.retrievers import EnsembleRetriever

    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[HYBRID_BM25_WEIGHT, HYBRID_DENSE_WEIGHT],
    )

    print(
        f"[RETRIEVER] Hybrid retriever ready: "
        f"BM25(w={HYBRID_BM25_WEIGHT}) + Dense(w={HYBRID_DENSE_WEIGHT}), "
        f"top_k={top_k}"
    )
    return hybrid


def _build_dense_retriever(vector_store, top_k: int):
    """
    Build a pure dense vector retriever from the active vector store.

    Uses similarity search (compatible with both FAISS and Pinecone).
    MMR is not used here to keep Pinecone compatibility; the re-ranker
    in reranker.py provides diversity control at a higher quality level.

    Args:
        vector_store: FAISS or PineconeVectorStore.
        top_k:        Number of results to retrieve.

    Returns:
        VectorStoreRetriever.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def retrieve(query: str, retriever) -> List[Document]:
    """
    Run a query through the hybrid (or dense-fallback) retriever.

    Args:
        query:     Natural-language question from the user.
        retriever: Result of build_hybrid_retriever().

    Returns:
        List[Document]: Retrieved chunks (before re-ranking).
    """
    return retriever.invoke(query)
