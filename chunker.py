# -*- coding: utf-8 -*-
"""
chunker.py — Phase 3: Text Chunking

Splits loaded Document pages into overlapping chunks using
RecursiveCharacterTextSplitter. Uses split_documents() (not split_text())
to guarantee that source + page metadata is preserved on every chunk.
"""

from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split a list of Documents into overlapping text chunks.

    Uses RecursiveCharacterTextSplitter.split_documents() which automatically
    copies metadata from the parent Document to every child chunk.
    NEVER uses split_text() — that method strips all metadata.

    After splitting, validates that every chunk retains:
        - metadata["source"]: original PDF filename
        - metadata["page"]:   integer page number

    Args:
        docs: List of Document objects returned by ingest.load_documents().

    Returns:
        List[Document]: Chunked Documents with full metadata intact.

    Raises:
        RuntimeError: If any chunk is missing required metadata fields.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    # CRITICAL: always split_documents(), never split_text()
    chunks = splitter.split_documents(docs)

    # ── Metadata integrity check ───────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        if "source" not in chunk.metadata or "page" not in chunk.metadata:
            raise RuntimeError(
                f"Chunk at index {i} is missing required metadata. "
                f"Present metadata keys: {list(chunk.metadata.keys())}"
            )

    # ── Validation stats ───────────────────────────────────────────────────
    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print(f"\n[CHUNK] Chunking complete")
    print(f"   Chunk size    : {CHUNK_SIZE} chars  |  Overlap: {CHUNK_OVERLAP} chars")
    print(f"   Total chunks  : {len(chunks)}")
    print(f"   Min length    : {min(lengths)} chars")
    print(f"   Max length    : {max(lengths)} chars")
    print(f"   Avg length    : {avg_len:.1f} chars")

    # First chunk
    first = chunks[0]
    print(f"\n[CHUNK 0] metadata: {first.metadata}")
    print(f"   Content      : {first.page_content[:200]}...")

    # Last chunk
    last = chunks[-1]
    print(f"\n[CHUNK -1] metadata: {last.metadata}")
    print(f"   Content      : {last.page_content[:200]}...\n")

    return chunks


if __name__ == "__main__":
    from ingest import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"[OK] Total chunks ready for embedding: {len(chunks)}")
