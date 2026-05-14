# -*- coding: utf-8 -*-
"""
build_index.py — One-Shot Index Builder

Run this script ONCE (with internet access) to:
  1. Download and cache all models locally (HuggingFace cache)
  2. Load all PDF documents from data/
  3. Chunk documents into overlapping text segments
  4. Embed chunks and save the FAISS vector index to disk

After this runs successfully, ALL subsequent runs are fully offline.

# FIRST RUN (internet required, one time only):
#   python build_index.py
#   This downloads and caches all models locally.
#
# ALL SUBSEQUENT RUNS (fully offline):
#   streamlit run app.py
"""

import sys

from config import print_config_summary
from ingest import load_documents
from chunker import chunk_documents
from embedder import build_vector_store


def main() -> None:
    """
    Execute the full index-building pipeline in sequence.

    Steps:
        1. Print config summary
        2. Load PDF documents
        3. Chunk documents
        4. Build and save FAISS vector store

    Any step failure prints a specific error message and exits cleanly.
    """
    print("\n" + "=" * 60)
    print("  Construction Knowledge Assistant — Build Index")
    print("=" * 60)

    # ── Step 1: Config summary ─────────────────────────────────────────────
    try:
        print_config_summary()
    except Exception as exc:
        print(f"\n❌ STEP 1 FAILED (Config): {exc}")
        sys.exit(1)

    # ── Step 2: Load documents ─────────────────────────────────────────────
    print("\n[STEP 2] Loading PDF documents ...")
    try:
        docs = load_documents()
        if not docs:
            print("[FAIL] STEP 2 FAILED: No documents loaded. Check the data/ directory.")
            sys.exit(1)
        print(f"[OK] STEP 2 DONE -- {len(docs)} pages loaded.")
    except Exception as exc:
        print(f"\n[FAIL] STEP 2 FAILED (Document Loading): {exc}")
        sys.exit(1)

    # ── Step 3: Chunk documents ────────────────────────────────────────────
    print("\n[STEP 3] Chunking documents ...")
    try:
        chunks = chunk_documents(docs)
        print(f"[OK] STEP 3 DONE -- {len(chunks)} chunks created.")
    except RuntimeError as exc:
        print(f"\n[FAIL] STEP 3 FAILED (Chunking / Metadata validation): {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[FAIL] STEP 3 FAILED (Chunking): {exc}")
        sys.exit(1)

    # ── Step 4: Build vector store ─────────────────────────────────────────
    print("\n[STEP 4] Building FAISS vector store (this may take a few minutes) ...")
    try:
        build_vector_store(chunks)
        print("[OK] STEP 4 DONE -- FAISS index built and saved.")
    except Exception as exc:
        print(f"\n[FAIL] STEP 4 FAILED (Embedding / FAISS): {exc}")
        sys.exit(1)

    # ── Done ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] Index built successfully.")
    print("   Run: streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
