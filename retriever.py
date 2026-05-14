# -*- coding: utf-8 -*-
"""
retriever.py — Phase 5: Retrieval System

Accepts a natural-language query and returns the top-K most relevant
document chunks from the FAISS vector store, with source + page metadata.
"""

from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from config import TOP_K
from embedder import load_vector_store


def retrieve(query: str, vector_store: FAISS) -> List[Document]:
    """
    Perform similarity search over the FAISS vector store.

    Args:
        query:        Natural-language question from the user.
        vector_store: A loaded FAISS vector store object.

    Returns:
        List[Document]: The top-K most relevant chunks, each with
                        metadata containing 'source' and 'page'.
    """
    results = vector_store.similarity_search(query, k=TOP_K)
    return results


def _print_results(results: List[Document], query: str) -> None:
    """
    Pretty-print retrieval results in a boxed format for CLI validation.

    Args:
        results: Retrieved Document objects.
        query:   The original query string (for display context).
    """
    bar = "-" * 52
    print(f"\n[QUERY] \"{query}\"")
    print(f"   Returning top {TOP_K} result(s):\n")
    for rank, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  +-- Rank {rank} {bar}")
        print(f"  |  Source : {source}")
        print(f"  |  Page   : {page}")
        print(f"  |  Preview: {preview}...")
        print(f"  +{'-' * (len(bar) + 10)}")
    print()


if __name__ == "__main__":
    TEST_QUERY = "What are the safety requirements for bridge inspection?"

    print("Loading vector store ...")
    vs = load_vector_store()
    results = retrieve(TEST_QUERY, vs)
    _print_results(results, TEST_QUERY)
