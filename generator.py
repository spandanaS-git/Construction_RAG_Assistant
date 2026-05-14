# -*- coding: utf-8 -*-
"""
generator.py — Phase 6 (Upgraded): LLM Integration with Backend Router

Generates an answer from retrieved context chunks using the active LLM backend.
The backend is selected at import time from config.LLM_BACKEND.

Supported backends (all swappable via .env):
  - huggingface (default): google/flan-t5-base, fully offline after first run
  - ollama: any local model (mistral, llama3, etc.) via Ollama REST API
  - openai: GPT-3.5 / GPT-4 via OpenAI API key

New in this version:
  - generate_answer_stream(): async generator for token-by-token streaming
  - Improved prompt templates: grounded expert (OpenAI) + compact (Flan-T5)
  - Existing generate_answer() unchanged — Streamlit UI keeps working
"""

import asyncio
from typing import AsyncIterator, List

from langchain.schema import Document

from config import LLM_BACKEND, LLM_MODEL, OPENAI_API_KEY, OLLAMA_BASE_URL

# ── Prompt templates ───────────────────────────────────────────────────────────

# For large-context models (OpenAI, Ollama): grounded expert with citations
PROMPT_TEMPLATE_EXPERT = """\
You are a Construction Safety Expert with deep knowledge of OSHA regulations,
bridge inspection standards, PPE requirements, and construction project documentation.

STRICT INSTRUCTIONS:
1. Answer using ONLY the information found in the Context below.
2. Cite the source document and page number for every factual claim, \
e.g. [osha_guide.pdf, p.12].
3. If the context does not contain sufficient information, respond with:
   "The available documents do not address this question."
4. Never invent regulation numbers, statistics, or technical specifications.
5. Use precise technical language appropriate for construction professionals.

Context:
{context}

Question: {question}

Answer:"""

# For small-context models (Flan-T5, 512-token limit): minimal overhead
PROMPT_TEMPLATE_COMPACT = """\
Construction safety context:
{context}

Based ONLY on the above, answer: {question}
If not found in context, say: "Not found in documents."
Answer:"""

# ── Token budget constants for flan-t5-base ────────────────────────────────────
_MAX_INPUT_TOKENS:       int = 480   # leave 32 tokens safety margin
_ANSWER_HEADROOM:        int = 150   # reserved for generated answer
_PROMPT_SKELETON_TOKENS: int = 30    # approximate overhead of COMPACT template

# ── Module-level model singletons (loaded ONCE, not per call) ─────────────────
_hf_tokenizer  = None
_hf_model      = None
_ollama_llm    = None
_openai_llm    = None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_huggingface():
    """Load and cache the HuggingFace tokenizer and seq2seq model to CPU."""
    global _hf_tokenizer, _hf_model
    if _hf_tokenizer is None or _hf_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        print(f"[LLM] Loading HuggingFace model: {LLM_MODEL} (CPU) ...")
        try:
            _hf_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            _hf_model     = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
            _hf_model.to("cpu")
            _hf_model.eval()
            print(f"[OK] Model loaded: {LLM_MODEL}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load HuggingFace model '{LLM_MODEL}': {exc}")


def _load_ollama():
    """Instantiate and cache the Ollama LangChain LLM client."""
    global _ollama_llm
    if _ollama_llm is None:
        try:
            from langchain_community.llms import Ollama
            _ollama_llm = Ollama(base_url=OLLAMA_BASE_URL, model=LLM_MODEL)
            print(f"[OK] Ollama client ready: model={LLM_MODEL}, url={OLLAMA_BASE_URL}")
        except Exception as exc:
            raise RuntimeError(f"Failed to connect to Ollama: {exc}")


def _load_openai_sync():
    """Instantiate and cache the OpenAI LangChain LLM client (sync calls)."""
    global _openai_llm
    if _openai_llm is None:
        try:
            from langchain_openai import ChatOpenAI
            _openai_llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                model=LLM_MODEL,
                temperature=0,
                max_tokens=500,
            )
            print(f"[OK] OpenAI chat client ready: model={LLM_MODEL}")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise OpenAI client: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN BUDGET (HuggingFace backend only)
# ══════════════════════════════════════════════════════════════════════════════

def _assemble_context_within_budget(
    context_docs: List[Document], query: str
) -> str:
    """
    Assemble context chunks that fit inside Flan-T5's 512-token window.

    Strategy:
      1. Measure fixed cost: prompt skeleton + question tokens.
      2. Divide remaining budget equally among chunks.
      3. Truncate each chunk to its token share (never blindly cut).

    Args:
        context_docs: Retrieved Document objects from the retriever.
        query:        The user query (needed to measure its token cost).

    Returns:
        str: Token-safe concatenated context string.
    """
    if _hf_tokenizer is None:
        # Fallback (non-HF backend): plain join
        return "\n---\n".join(doc.page_content for doc in context_docs)

    query_tokens = len(_hf_tokenizer.encode(query, add_special_tokens=False))
    fixed_cost   = _PROMPT_SKELETON_TOKENS + query_tokens
    context_budget = _MAX_INPUT_TOKENS - fixed_cost

    if context_budget <= 0:
        print("[WARN] Query too long to leave room for context.")
        return ""

    n_chunks = max(len(context_docs), 1)
    per_chunk_budget = context_budget // n_chunks
    separator_cost   = 3

    trimmed_chunks: List[str] = []
    for doc in context_docs:
        token_ids = _hf_tokenizer.encode(doc.page_content, add_special_tokens=False)
        allowed   = max(per_chunk_budget - separator_cost, 10)
        if len(token_ids) > allowed:
            token_ids  = token_ids[:allowed]
            chunk_text = _hf_tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            chunk_text = doc.page_content
        trimmed_chunks.append(chunk_text)

    assembled    = "\n---\n".join(trimmed_chunks)
    total_tokens = len(_hf_tokenizer.encode(assembled, add_special_tokens=False))
    print(
        f"[TOKEN BUDGET] query={query_tokens}tok  context={total_tokens}tok  "
        f"budget={context_budget}tok  chunks={n_chunks}x{per_chunk_budget}tok"
    )
    return assembled


# ══════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS GENERATION (used by Streamlit + /query/sync endpoint)
# ══════════════════════════════════════════════════════════════════════════════

def generate_answer(query: str, context_docs: List[Document]) -> str:
    """
    Generate a natural-language answer from retrieved context chunks.

    Selects the prompt template based on LLM_BACKEND:
      - huggingface → compact template (preserves token budget)
      - openai/ollama → expert template (grounded with citation instruction)

    Args:
        query:        User's original question.
        context_docs: Retrieved Document objects from retriever / reranker.

    Returns:
        str: Generated answer, or an error message string on failure.
    """
    use_expert_prompt = LLM_BACKEND in ("openai", "ollama")

    if LLM_BACKEND == "huggingface":
        _load_huggingface()
        context = _assemble_context_within_budget(context_docs, query)
        prompt  = PROMPT_TEMPLATE_COMPACT.format(context=context, question=query)
    else:
        context = "\n---\n".join(doc.page_content for doc in context_docs)
        prompt  = PROMPT_TEMPLATE_EXPERT.format(context=context, question=query)

    try:
        # ── HuggingFace ──────────────────────────────────────────────────────
        if LLM_BACKEND == "huggingface":
            import torch

            inputs = _hf_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=False,
                max_length=None,
            )
            n_input_tokens = inputs["input_ids"].shape[-1]
            if n_input_tokens > 512:
                print(
                    f"[WARN] Prompt is {n_input_tokens} tokens — "
                    "exceeds 512 limit. Forcing tokenizer truncation."
                )
                inputs = _hf_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )

            with torch.no_grad():
                outputs = _hf_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                )
            answer = _hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()

        # ── Ollama ───────────────────────────────────────────────────────────
        elif LLM_BACKEND == "ollama":
            _load_ollama()
            return _ollama_llm(prompt).strip()

        # ── OpenAI ───────────────────────────────────────────────────────────
        elif LLM_BACKEND == "openai":
            _load_openai_sync()
            from langchain.schema import HumanMessage
            response = _openai_llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()

        else:
            return f"Answer could not be generated: unknown backend '{LLM_BACKEND}'"

    except Exception as exc:
        return f"Answer could not be generated: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING GENERATION (used by FastAPI /query SSE endpoint)
# ══════════════════════════════════════════════════════════════════════════════

async def generate_answer_stream(
    query: str, context_docs: List[Document]
) -> AsyncIterator[str]:
    """
    Async generator that yields answer tokens for Server-Sent Events (SSE).

    OpenAI backend: real token streaming via openai.AsyncOpenAI.
    Ollama backend: real token streaming via httpx async calls.
    HuggingFace:    pseudo-streaming — generates full answer then
                    yields word-by-word (Flan-T5 is seq2seq, no native
                    token streaming without custom decoding hooks).

    Args:
        query:        User's question.
        context_docs: Retrieved + re-ranked Document objects.

    Yields:
        str: Individual tokens (or words for HF) forming the answer.
    """
    context = "\n---\n".join(doc.page_content for doc in context_docs)

    # ── OpenAI streaming ─────────────────────────────────────────────────────
    if LLM_BACKEND == "openai":
        from openai import AsyncOpenAI

        prompt = PROMPT_TEMPLATE_EXPERT.format(context=context, question=query)
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        try:
            stream = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=500,
                temperature=0,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
        except Exception as exc:
            yield f"[ERROR] OpenAI streaming failed: {exc}"
        return

    # ── Ollama streaming ──────────────────────────────────────────────────────
    if LLM_BACKEND == "ollama":
        import json
        import httpx

        prompt = PROMPT_TEMPLATE_EXPERT.format(context=context, question=query)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            data  = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done", False):
                                break
        except Exception as exc:
            yield f"[ERROR] Ollama streaming failed: {exc}"
        return

    # ── HuggingFace pseudo-streaming ──────────────────────────────────────────
    # Flan-T5 is a seq2seq model: generate the full answer synchronously,
    # then yield word-by-word to simulate a streaming experience.
    full_answer = generate_answer(query, context_docs)
    words = full_answer.split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.025)   # 25ms cadence ≈ natural typing speed


# ── Smoke test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from langchain.schema import Document as Doc

    mock_docs = [
        Doc(
            page_content="Workers must wear hard hats, high-visibility vests, "
                         "and steel-toed boots on all active construction sites.",
            metadata={"source": "osha_construction_safety_guide.pdf", "page": 5},
        )
    ]
    answer = generate_answer("What PPE is required on a construction site?", mock_docs)
    print(f"\n[SYNC] Generated answer:\n{answer}")
