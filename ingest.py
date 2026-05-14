# -*- coding: utf-8 -*-
"""
ingest.py — Phase 2: Document Ingestion

Scans DATA_DIR for all PDF files and loads them into LangChain Document
objects using PyPDFLoader. Each page is kept as a separate Document to
preserve page-level metadata (source filename + page number) through
the entire pipeline.
"""

import warnings
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

from config import DATA_DIR


def load_documents() -> List[Document]:
    """
    Discover and load all PDF files from DATA_DIR.

    Each page becomes a separate LangChain Document with metadata:
        - source: filename only (not full path)
        - page: page number (int)

    Files that fail to load are skipped with a warning; the pipeline continues.

    Returns:
        List[Document]: All successfully loaded pages from all PDFs.
    """
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    total_found = len(pdf_files)
    print(f"\n[DATA] Data directory  : {DATA_DIR.resolve()}")
    print(f"[INFO] PDF files found : {total_found}")

    if total_found == 0:
        print("[WARN] No PDF files found. Add PDFs to the data/ directory.")
        return []

    all_documents: List[Document] = []
    skipped: List[str] = []
    per_file_counts: dict = {}

    for pdf_path in pdf_files:
        filename = pdf_path.name
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Normalise metadata: keep only filename (not full path) + page
            for page in pages:
                page.metadata["source"] = filename
                if "page" not in page.metadata:
                    page.metadata["page"] = 0

            per_file_counts[filename] = len(pages)
            all_documents.extend(pages)

        except Exception as exc:
            warnings.warn(f"⚠️  Could not load '{filename}': {exc}")
            skipped.append(f"{filename}: {exc}")

    # ── Per-file summary table ─────────────────────────────────────────────
    print(f"\n  {'File':<50} {'Pages':>6}")
    print("  " + "-" * 58)
    for fname, count in per_file_counts.items():
        print(f"  {fname:<50} {count:>6}")
    print("  " + "-" * 58)
    print(f"  {'TOTAL PAGES LOADED':<50} {len(all_documents):>6}\n")

    if skipped:
        print("[WARN] Skipped files:")
        for s in skipped:
            print(f"   - {s}")

    # ── Preview first 300 chars of the first page ──────────────────────────
    if all_documents:
        preview = all_documents[0].page_content[:300].replace("\n", " ")
        first_source = all_documents[0].metadata.get("source", "unknown")
        print(f"[PREVIEW] First 300 chars of '{first_source}', page 1:")
        print(f"   {preview}\n")

    return all_documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"[OK] Total documents loaded: {len(docs)}")
