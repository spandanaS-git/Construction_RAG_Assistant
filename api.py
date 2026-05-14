# -*- coding: utf-8 -*-
"""
api.py — FastAPI Inference Layer (v2)

Exposes the RAG pipeline as a production-ready HTTP API with:
  - Streaming responses via Server-Sent Events (SSE)
  - Sync JSON responses for simple clients
  - Liveness + readiness health check
  - Auto-generated OpenAPI docs at /docs

Endpoints:
  GET  /health        → system status (vector store, backends, features)
  POST /query         → streaming SSE (text/event-stream)
  POST /query/sync    → full JSON response (non-streaming)
  GET  /docs          → FastAPI OpenAPI UI (built-in)

Run locally:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Test streaming:
  curl -N -X POST http://localhost:8000/query \\
    -H "Content-Type: application/json" \\
    -d '{"question": "What PPE is required on a construction site?"}'

Test sync:
  curl -X POST http://localhost:8000/query/sync \\
    -H "Content-Type: application/json" \\
    -d '{"question": "What PPE is required on a construction site?"}' | python -m json.tool
"""

import asyncio
from typing import AsyncIterator, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── App init ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Construction Safety RAG API",
    description=(
        "Retrieval-Augmented Generation API for construction safety documents. "
        "Supports hybrid search (BM25 + dense), cross-encoder re-ranking, "
        "and streaming token responses."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="What PPE is required on a construction site?",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Candidate pool size (before re-ranking).",
    )


class SourceRef(BaseModel):
    file: str = Field(..., example="osha_construction_safety_guide.pdf")
    page: int = Field(..., example=5)


class QueryResponse(BaseModel):
    """Response schema for the non-streaming /query/sync endpoint."""
    answer:  str            = Field(..., example="Workers must wear hard hats...")
    sources: List[SourceRef] = Field(default_factory=list)
    chunks:  List[str]       = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response schema for the /health liveness check."""
    status:                str  = Field(..., example="ok")
    vector_store_loaded:   bool
    llm_backend:           str
    embedding_backend:     str
    vector_backend:        str
    hybrid_search_enabled: bool
    reranker_enabled:      bool
    active_index:          str


# ══════════════════════════════════════════════════════════════════════════════
# Startup: import pipeline (loads vector store, BM25, reranker singletons)
# ══════════════════════════════════════════════════════════════════════════════

# Importing pipeline triggers module-level loading of all singletons.
# This happens once when uvicorn starts up.
from pipeline import ask_question, _vector_store, _retriever


@app.on_event("startup")
async def startup_event():
    from config import (
        LLM_BACKEND,
        EMBEDDING_BACKEND,
        VECTOR_BACKEND,
        HYBRID_SEARCH_ENABLED,
        RERANKER_ENABLED,
        get_active_pinecone_index_name,
    )
    index = (
        get_active_pinecone_index_name() if VECTOR_BACKEND == "pinecone"
        else str(__import__("config").VECTOR_STORE_DIR)
    )
    status = "ok" if _vector_store is not None else "degraded (no vector store)"
    print(
        f"\n[API] Startup complete — status: {status}\n"
        f"  LLM: {LLM_BACKEND}  |  Embeddings: {EMBEDDING_BACKEND}  |  "
        f"VectorDB: {VECTOR_BACKEND}\n"
        f"  Hybrid: {HYBRID_SEARCH_ENABLED}  |  Reranker: {RERANKER_ENABLED}\n"
        f"  Index: {index}\n"
        f"  Docs:  http://localhost:8000/docs\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness check",
    tags=["System"],
)
async def health():
    """
    Returns the current system status including which backends are active
    and whether the vector store is loaded.
    """
    from config import (
        LLM_BACKEND,
        EMBEDDING_BACKEND,
        VECTOR_BACKEND,
        HYBRID_SEARCH_ENABLED,
        RERANKER_ENABLED,
        get_active_pinecone_index_name,
    )

    active_index = (
        get_active_pinecone_index_name()
        if VECTOR_BACKEND == "pinecone"
        else "faiss (local)"
    )

    return HealthResponse(
        status="ok" if _vector_store is not None else "degraded",
        vector_store_loaded=(_vector_store is not None),
        llm_backend=LLM_BACKEND,
        embedding_backend=EMBEDDING_BACKEND,
        vector_backend=VECTOR_BACKEND,
        hybrid_search_enabled=HYBRID_SEARCH_ENABLED,
        reranker_enabled=RERANKER_ENABLED,
        active_index=active_index,
    )


async def _sse_stream(question: str) -> AsyncIterator[str]:
    """
    Internal async generator that drives the SSE stream.

    Runs retrieval + re-ranking synchronously (they are CPU/network bound),
    then streams each token from the generator.

    SSE format:
        data: <token>\n\n
        ...
        data: [DONE]\n\n
    """
    from config import RERANKER_TOP_N
    from hybrid_retriever import retrieve
    from reranker import rerank
    from generator import generate_answer_stream

    if _retriever is None or _vector_store is None:
        yield "data: [ERROR] Vector store not loaded. Run build_index.py first.\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        # Step 1: Retrieve candidates (synchronous)
        candidate_docs = retrieve(question, _retriever)

        # Step 2: Re-rank (synchronous, CPU cross-encoder)
        final_docs = rerank(question, candidate_docs, top_n=RERANKER_TOP_N)

        # Step 3: Stream tokens
        async for token in generate_answer_stream(question, final_docs):
            # Escape newlines in SSE data fields (SSE spec requirement)
            escaped = token.replace("\n", " ")
            yield f"data: {escaped}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"


@app.post(
    "/query",
    summary="Streaming RAG query (SSE)",
    response_description="Server-Sent Events stream of answer tokens",
    tags=["RAG"],
)
async def query_stream(request: QueryRequest):
    """
    Stream the RAG answer token-by-token via Server-Sent Events.

    Connect with an EventSource client or curl:

        curl -N -X POST http://localhost:8000/query \\
          -H "Content-Type: application/json" \\
          -d '{"question": "What PPE is required?"}'

    Each event is in the format: `data: <token>\\n\\n`
    The stream ends with: `data: [DONE]\\n\\n`
    """
    if _vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded. Run `python build_index.py` first.",
        )

    return StreamingResponse(
        _sse_stream(request.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Connection":        "keep-alive",
        },
    )


@app.post(
    "/query/sync",
    response_model=QueryResponse,
    summary="Synchronous RAG query (JSON)",
    tags=["RAG"],
)
async def query_sync(request: QueryRequest):
    """
    Return the full RAG answer as a single JSON response (non-streaming).

    Use this endpoint for simple clients that don't support SSE,
    or for testing via the /docs UI.
    """
    if _vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded. Run `python build_index.py` first.",
        )

    # Run the pipeline (synchronous, runs in the event loop thread)
    result = ask_question(request.question)

    if result["answer"].startswith("Error:"):
        raise HTTPException(status_code=500, detail=result["answer"])

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceRef(**s) for s in result["sources"]],
        chunks=result["chunks"],
    )


# ── Entry point for direct execution ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
