# 🏗️ Construction Safety RAG Assistant — Run Guide

> **Two fully working modes.** Switch between them by editing 5 lines in `.env`.  
> Everything else — Streamlit UI, FastAPI, hybrid search, re-ranking — stays identical.

---

## Architecture Overview

```
PDFs (./data/)
    │
    ▼  ingest.py → chunker.py
Chunks (8938 total, ~434 chars each)
    │
    ├──────────────────────────────────────────────────┐
    ▼  [MODE 1] FAISS (local file)                    ▼  [MODE 2] Pinecone (cloud)
    │  all-MiniLM-L6-v2 (384-dim, free)               │  text-embedding-3-small (1536-dim, ~$0.02 once)
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼  hybrid_retriever.py
              BM25 (sparse) + Dense (vector)
              Reciprocal Rank Fusion → top-5 candidates
                       │
                       ▼  reranker.py
              CrossEncoder (ms-marco-MiniLM) → top-3 docs
                       │
    ┌──────────────────┴───────────────────────────────┐
    ▼  [MODE 1] Flan-T5-base (local, CPU, free)        ▼  [MODE 2] GPT-3.5-turbo (OpenAI API, ~$0.002/q)
    └──────────────────┬───────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Streamlit UI       FastAPI Server
        (port 8501)        (port 8000)
```

---

## Mode 1 — Local (Free, Zero Cost)

**Use this for**: Development, testing, working offline.

**Characteristics**:
- Embedding: `all-MiniLM-L6-v2` (HuggingFace, CPU, 384-dim)
- Vector DB: FAISS (local file, `./vector_store/`)
- LLM: `google/flan-t5-base` (HuggingFace, CPU, 512-token limit)
- Answers: Short (1-2 sentences, token-limited)
- Cost: **$0.00**

### Step 1 — Switch `.env` to Mode 1

Open `.env` and make the Mode 1 block active, Mode 2 commented out:

```env
# ── MODE 1: LOCAL (free, offline after first download) ────────
EMBEDDING_BACKEND=huggingface
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_BACKEND=huggingface
LLM_MODEL=google/flan-t5-base
VECTOR_BACKEND=faiss

# ── MODE 2: OPENAI + PINECONE (demo day, best quality) ────────
# EMBEDDING_BACKEND=openai
# EMBEDDING_MODEL=text-embedding-3-small
# LLM_BACKEND=openai
# LLM_MODEL=gpt-3.5-turbo
# VECTOR_BACKEND=pinecone
```

### Step 2 — Validate Config

```powershell
python -X utf8 config.py
```

✅ Expected output:
```
Embedding Backend : huggingface
LLM Backend       : huggingface
Vector Backend    : faiss
```

### Step 3 — Build Index (only needed once, or after switching from Mode 2)

> Skip this step if you've already built the FAISS index and haven't switched embedding models.

```powershell
python -X utf8 build_index.py
```

✅ Takes ~3-5 minutes. Creates `./vector_store/index.faiss` + `chunks.pkl`.

### Step 4 — Run the App(s)

**Streamlit UI** (open a new terminal):
```powershell
streamlit run app.py
```
Open: **http://localhost:8501**

**FastAPI** (open another terminal):
```powershell
python -X utf8 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open: **http://localhost:8000/docs**

---

## Mode 2 — OpenAI + Pinecone (Demo Day)

**Use this for**: Class demonstration, best answer quality, professor demo.

**Characteristics**:
- Embedding: `text-embedding-3-small` (OpenAI API, 1536-dim)
- Vector DB: Pinecone serverless (cloud, `cnst-openai` index)
- LLM: `gpt-3.5-turbo` (OpenAI API, 16k context)
- Answers: Full paragraphs with source citations
- Cost: ~$0.02 for index build (once) + ~$0.002 per query

### Prerequisites (one-time setup)

**Pinecone**:
1. Sign up free at [pinecone.io](https://www.pinecone.io)
2. Create API Key → copy it
3. Create index: name=`cnst-openai`, dimensions=`1536`, metric=`Cosine`, cloud=AWS, region=`us-east-1`
4. Create index: name=`cnst-local`, dimensions=`384`, metric=`Cosine` (for Mode 1 if needed later)

**OpenAI**:
1. Sign up at [platform.openai.com](https://platform.openai.com)
2. Go to API Keys → Create new secret key → copy it
3. Add a payment method under Billing ($5 minimum is plenty)

### Step 1 — Add Keys to `.env`

```env
OPENAI_API_KEY=sk-proj-your-full-key-here
PINECONE_API_KEY=your-pinecone-uuid-key-here
```

> ⚠️ Paste the full key on one line. No quotes. No spaces. Use the copy button — don't manually select.

### Step 2 — Switch `.env` to Mode 2

```env
# ── MODE 1: LOCAL (free, offline after first download) ────────
# EMBEDDING_BACKEND=huggingface
# EMBEDDING_MODEL=all-MiniLM-L6-v2
# LLM_BACKEND=huggingface
# LLM_MODEL=google/flan-t5-base
# VECTOR_BACKEND=faiss

# ── MODE 2: OPENAI + PINECONE (demo day, best quality) ────────
EMBEDDING_BACKEND=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_BACKEND=openai
LLM_MODEL=gpt-3.5-turbo
VECTOR_BACKEND=pinecone
```

### Step 3 — Validate Keys

```powershell
python -X utf8 check_keys.py
```

✅ Expected output:
```
=== OpenAI Key ===
  Length : 164 chars      ← must be 100+
  Valid  : True
  Preview: sk-proj-Sc...bAA

=== Pinecone Key ===
  Length : 73 chars
  Preview: pcsk_2s7...xQi
```

### Step 4 — Validate Config

```powershell
python -X utf8 config.py
```

✅ Expected output:
```
Embedding Backend : openai
LLM Backend       : openai
Vector Backend    : pinecone
Active Index      : cnst-openai
```

### Step 5 — Build Pinecone Index (one-time, ~5 min, costs ~$0.02)

```powershell
python -X utf8 build_index.py
```

✅ Embeds 8938 chunks via OpenAI and upserts to Pinecone.  
You can watch vector count go up live in the [Pinecone Console](https://app.pinecone.io).

> After this runs once, you **never need to run it again** unless you add new PDFs.
> Pinecone stores the vectors in the cloud permanently.

### Step 6 — Run the App(s)

**Streamlit UI**:
```powershell
streamlit run app.py
```
Open: **http://localhost:8501**

**FastAPI** (always use `-X utf8` on Windows):
```powershell
python -X utf8 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open: **http://localhost:8000/docs**

---

## Switching Between Modes

| Scenario | Action needed |
|---|---|
| **Mode 1 → Mode 2** | Comment/uncomment 5 lines in `.env` → run `build_index.py` once (if not done yet) → restart server |
| **Mode 2 → Mode 1** | Comment/uncomment 5 lines in `.env` → restart server (FAISS index already on disk) |
| **Adding new PDFs** | Drop PDF into `./data/` → run `build_index.py` in active mode → restart server |
| **After any `.env` change** | Always restart Streamlit and/or uvicorn to reload config |

> The FAISS index (`./vector_store/`) and Pinecone cloud index are **independent**.  
> Switching modes does **not** delete either index. Both can coexist.

---

## Testing Endpoints

### Health Check (Mode doesn't matter)
```powershell
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET | Select-Object -ExpandProperty Content
```

### Query via PowerShell
```powershell
$body = '{"question": "What PPE is required on a construction site?", "top_k": 5}'
Invoke-WebRequest -Uri "http://localhost:8000/query/sync" `
  -Method POST -ContentType "application/json" -Body $body `
  | Select-Object -ExpandProperty Content
```

### Query via Python script
```python
import httpx

r = httpx.post(
    "http://localhost:8000/query/sync",
    json={"question": "What fall protection is required above 6 feet?", "top_k": 5},
    timeout=120,
)
print(r.json()["answer"])
```

### Interactive browser testing
Open **http://localhost:8000/docs** → `POST /query/sync` → Try it out

---

## File Reference

| File | Purpose |
|---|---|
| `.env` | All configuration — the only file you edit |
| `config.py` | Reads `.env`, validates settings |
| `ingest.py` | Loads PDFs from `./data/` |
| `chunker.py` | Splits pages into 500-char overlapping chunks |
| `embedder.py` | Embeds chunks → FAISS or Pinecone |
| `hybrid_retriever.py` | BM25 + dense retrieval (RRF fusion) |
| `reranker.py` | CrossEncoder re-ranking |
| `generator.py` | LLM answer generation (sync + streaming) |
| `pipeline.py` | Orchestrates the full RAG pipeline |
| `api.py` | FastAPI server (`/health`, `/query`, `/query/sync`) |
| `app.py` | Streamlit UI |
| `build_index.py` | One-shot script to embed + index all PDFs |
| `check_keys.py` | Validates API keys from `.env` |
| `./data/` | Drop your PDFs here |
| `./vector_store/` | FAISS index files + `chunks.pkl` (BM25 corpus) |

---

## Cost Summary

| Item | Mode 1 | Mode 2 |
|---|---|---|
| Index build | $0.00 | ~$0.02 (one-time) |
| Per query (retrieval) | $0.00 | $0.00 (Pinecone free) |
| Per query (LLM) | $0.00 | ~$0.002 (GPT-3.5) |
| 100 demo queries | **$0.00** | **~$0.20** |
| Pinecone storage | $0.00 | $0.00 (free tier) |

---

*Last updated: April 2026 — Construction Safety RAG v2.0*
