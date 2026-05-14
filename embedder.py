# -*- coding: utf-8 -*-
"""
embedder.py — Phase 4 (Upgraded): Embeddings & Vector Store

Supports two embedding backends and two vector store backends — all
controlled exclusively via config.py / .env. No hardcoded values.

Embedding backends:
  huggingface (default) → all-MiniLM-L6-v2 on CPU, fully offline
  openai                → text-embedding-3-small via OpenAI API

Vector store backends:
  faiss    (default) → local flat-file index, zero cloud dependency
  pinecone           → cloud serverless index, supports multi-instance

Additional utilities:
  save_chunks(chunks) → persists chunk corpus to disk for BM25 re-use
  load_chunks()       → reloads chunk corpus without re-embedding
"""

import os
import pickle
import warnings

# Suppress TensorFlow / Keras warnings before any heavy imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings.*")

from config import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    VECTOR_BACKEND,
    VECTOR_STORE_DIR,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    get_active_pinecone_index_name,
)

# ── Path for persisted chunk corpus (used by BM25 at startup) ─────────────────
CHUNKS_FILE: Path = VECTOR_STORE_DIR / "chunks.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

def _get_embeddings():
    """
    Return the active embedding model based on EMBEDDING_BACKEND.

    huggingface → HuggingFaceEmbeddings (CPU, L2-normalized cosine)
    openai      → OpenAIEmbeddings via langchain-openai

    Returns:
        Embedding object compatible with LangChain vectorstore APIs.
    """
    if EMBEDDING_BACKEND == "openai":
        from langchain_openai import OpenAIEmbeddings
        print(f"[EMBED] Using OpenAI embeddings: {EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )

    # Default: huggingface
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print(f"[EMBED] Using HuggingFace embeddings: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK PERSISTENCE (for BM25 corpus re-use)
# ══════════════════════════════════════════════════════════════════════════════

def save_chunks(chunks: List[Document]) -> None:
    """
    Persist the full chunk corpus to disk so BM25 can reload it at startup
    without re-ingesting and re-chunking all PDFs.

    Saved to: VECTOR_STORE_DIR/chunks.pkl

    Args:
        chunks: All Document chunks produced by chunker.chunk_documents().
    """
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[OK] Chunk corpus saved: {len(chunks)} chunks to {CHUNKS_FILE.resolve()}")


def load_chunks() -> Optional[List[Document]]:
    """
    Load the persisted chunk corpus from disk.

    Returns:
        List[Document] if chunks.pkl exists, else None.
    """
    if not CHUNKS_FILE.exists():
        return None
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"[OK] Chunk corpus loaded: {len(chunks)} chunks from {CHUNKS_FILE.resolve()}")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# FAISS BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def _build_faiss_store(chunks: List[Document]):
    """Build and persist a FAISS vector store from chunks."""
    from langchain_community.vectorstores import FAISS

    embeddings = _get_embeddings()
    print(f"[FAISS] Building index over {len(chunks)} chunks ...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"[OK] FAISS index saved to: {VECTOR_STORE_DIR.resolve()}")
    return vector_store


def _load_faiss_store():
    """Load an existing FAISS vector store from disk."""
    from langchain_community.vectorstores import FAISS

    index_file = VECTOR_STORE_DIR / "index.faiss"
    if not VECTOR_STORE_DIR.exists() or not index_file.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run: python build_index.py"
        )
    embeddings = _get_embeddings()
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[OK] FAISS index loaded: {vector_store.index.ntotal} vectors")
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
# PINECONE BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def _build_pinecone_store(chunks: List[Document]):
    """
    Upsert all chunks into the Pinecone cloud index.

    Creates the index if it does not exist yet, using the dimension
    inferred from the first embedding call. Index name is determined
    by the active EMBEDDING_BACKEND (local=384-dim, openai=1536-dim).
    """
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    index_name = get_active_pinecone_index_name()
    print(f"[PINECONE] Target index: '{index_name}'")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        # Probe one embedding to get the output dimension
        embeddings = _get_embeddings()
        sample_vec = embeddings.embed_query("dimension probe")
        dim = len(sample_vec)
        print(f"[PINECONE] Creating index '{index_name}' ({dim}-dim, cosine) ...")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[OK] Pinecone index '{index_name}' created.")
    else:
        print(f"[PINECONE] Index '{index_name}' already exists - upserting chunks.")
        embeddings = _get_embeddings()

    print(f"[PINECONE] Upserting {len(chunks)} chunks (this may take a few minutes) ...")
    vector_store = PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY,
    )
    print(f"[OK] Pinecone upsert complete, index '{index_name}'")
    return vector_store


def _load_pinecone_store():
    """Connect to an existing Pinecone index (no local files needed)."""
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    index_name = get_active_pinecone_index_name()
    embeddings = _get_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", "unknown")

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    print(f"[OK] Pinecone index '{index_name}' connected: {total} vectors")
    return vector_store


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — called by build_index.py and pipeline.py
# ══════════════════════════════════════════════════════════════════════════════

def build_vector_store(chunks: List[Document]):
    """
    Embed all chunks and persist to the active vector store backend.

    Also saves a chunks.pkl file alongside the index so BM25 can
    reload the corpus at inference time without re-chunking.

    Args:
        chunks: List of chunked Document objects (with source + page metadata).

    Returns:
        FAISS | PineconeVectorStore: The built vector store object.
    """
    print(f"\n[EMBED] Backend: {EMBEDDING_BACKEND}  |  Vector store: {VECTOR_BACKEND}")

    # Always persist chunk corpus for BM25 re-use
    save_chunks(chunks)

    if VECTOR_BACKEND == "pinecone":
        return _build_pinecone_store(chunks)
    else:
        return _build_faiss_store(chunks)


def load_vector_store():
    """
    Load a previously built vector store from the active backend.

    Returns:
        FAISS | PineconeVectorStore: Ready for similarity_search().

    Raises:
        FileNotFoundError: FAISS index files missing.
        Exception: Pinecone connection / auth failure.
    """
    print(f"[EMBED] Loading vector store (backend: {VECTOR_BACKEND}) ...")

    if VECTOR_BACKEND == "pinecone":
        return _load_pinecone_store()
    else:
        return _load_faiss_store()


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingest import load_documents
    from chunker import chunk_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    vs = build_vector_store(chunks)
    print(f"\n[INFO] Reloading to verify ...")
    vs2 = load_vector_store()
    print(f"[OK] Reload successful.")

    # Verify BM25 corpus
    loaded = load_chunks()
    if loaded:
        print(f"[OK] BM25 corpus intact: {len(loaded)} chunks")
