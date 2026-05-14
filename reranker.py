# -*- coding: utf-8 -*-
"""
reranker.py — Cross-Encoder Re-ranking

Takes the candidate pool retrieved by the hybrid retriever and re-scores
every (query, chunk) pair using a cross-encoder model. Cross-encoders
jointly process the query and document together (unlike bi-encoders that
embed them separately), producing much more accurate relevance scores.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on the MS MARCO passage ranking dataset
  - ~80 MB download, runs on CPU in ~10ms per query (3-10 docs)
  - Free, no API key required

Flow:
    Hybrid retriever returns top-K docs (e.g. 5)
        │
        ▼
    Cross-encoder scores each (query, doc) pair → float relevance score
        │
        ▼
    Sort by score descending → return top-N (e.g. 3) to LLM

Config knobs (all in .env):
  RERANKER_ENABLED   true | false
  RERANKER_MODEL     HuggingFace cross-encoder model name
  RERANKER_TOP_N     final number of docs passed to the LLM
"""

from typing import List

from langchain.schema import Document

from config import RERANKER_ENABLED, RERANKER_MODEL, RERANKER_TOP_N

# ── Singleton cross-encoder (loaded once, reused across all queries) ───────────
_cross_encoder = None


def _load_cross_encoder():
    """
    Load and cache the cross-encoder model.

    Uses sentence-transformers CrossEncoder which is already installed
    as a dependency of sentence-transformers (no extra package needed).
    Loads to CPU automatically.
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"[RERANKER] Loading cross-encoder: {RERANKER_MODEL} ...")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        print(f"[OK] Cross-encoder ready: {RERANKER_MODEL}")
    return _cross_encoder


def rerank(
    query: str,
    docs: List[Document],
    top_n: int = RERANKER_TOP_N,
) -> List[Document]:
    """
    Re-rank a list of retrieved documents using a cross-encoder.

    If RERANKER_ENABLED=false, returns docs[:top_n] without scoring
    (preserves the order from the hybrid retriever / RRF fusion).

    Args:
        query:  The original user question.
        docs:   Candidate documents from the hybrid retriever.
        top_n:  Number of top-ranked documents to return (must be ≤ len(docs)).

    Returns:
        List[Document]: Top-N documents sorted by cross-encoder score,
                        descending. Each doc retains its original metadata.
    """
    if not docs:
        return []

    # Clamp top_n to available docs
    top_n = min(top_n, len(docs))

    if not RERANKER_ENABLED:
        # No reranking — just truncate to top_n
        return docs[:top_n]

    model = _load_cross_encoder()

    # Build (query, passage) pairs for scoring
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)  # returns numpy array of floats

    # Zip scores with docs, sort descending, take top_n
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in scored[:top_n]]

    # Debug log: show scores
    print(f"[RERANKER] Scored {len(docs)} docs → kept top {top_n}:")
    for i, (score, doc) in enumerate(scored[:top_n]):
        src = doc.metadata.get("source", "?")
        pg  = doc.metadata.get("page",   "?")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i+1}] score={score:.4f}  {src} p{pg}  \"{preview}...\"")

    return top_docs


if __name__ == "__main__":
    # Quick smoke test with dummy docs
    from langchain.schema import Document as Doc

    test_docs = [
        Doc(page_content="Workers must wear hard hats and steel-toed boots at all times.",
            metadata={"source": "osha.pdf", "page": 3}),
        Doc(page_content="Bridge deck inspections require non-destructive testing methods.",
            metadata={"source": "txdot.pdf", "page": 12}),
        Doc(page_content="Personal fall arrest systems must be worn above 6 feet.",
            metadata={"source": "osha.pdf", "page": 7}),
    ]
    result = rerank("What PPE is required on a construction site?", test_docs, top_n=2)
    print(f"\n[TEST] Top-2 after reranking:")
    for doc in result:
        print(f"  - {doc.metadata['source']} p{doc.metadata['page']}: {doc.page_content[:100]}")
