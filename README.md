# 🏗️ Construction Safety RAG Assistant

A **Retrieval-Augmented Generation (RAG)** system built for construction safety knowledge — powered by hybrid search, cross-encoder re-ranking, and a dual-mode architecture that runs fully **free & offline** or with **OpenAI + Pinecone** for best quality.

---

## ✨ Features

- 🔍 **Hybrid Search** — BM25 (sparse) + Dense vector retrieval fused with Reciprocal Rank Fusion
- 🎯 **Cross-Encoder Re-ranking** — `ms-marco-MiniLM-L-6-v2` narrows top-5 candidates to top-3 most relevant docs
- 🔀 **Dual-Mode Architecture** — switch between fully local (free) and cloud (best quality) by editing 5 lines in `.env`
- 🖥️ **Streamlit UI** — interactive chat interface at `localhost:8501`
- ⚡ **FastAPI Server** — REST API with sync + streaming endpoints at `localhost:8000`
- 📄 **Multi-document** — ingests multiple construction safety PDFs and chunks them into 8,938 searchable segments

---

## 🏛️ Architecture

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

## 🔀 Mode Comparison

| Feature | Mode 1 — Local | Mode 2 — Cloud |
|---|---|---|
| Embedding | `all-MiniLM-L6-v2` (HuggingFace) | `text-embedding-3-small` (OpenAI) |
| Vector DB | FAISS (local file) | Pinecone (serverless cloud) |
| LLM | `google/flan-t5-base` (CPU) | `gpt-3.5-turbo` (OpenAI API) |
| Answer Quality | Short (1-2 sentences) | Full paragraphs with citations |
| Index Build Cost | **$0.00** | ~$0.02 (one-time) |
| Per Query Cost | **$0.00** | ~$0.002 |
| Use Case | Development / offline | Demo day / best quality |

---

## 📁 Project Structure

```
Construction_RAG_Assistant/
├── .env.example              # ← Copy to .env and fill in your API keys
├── .gitignore
├── requirements.txt
├── README.md
├── RUN_GUIDE.md              # Detailed run instructions
│
├── config.py                 # Reads .env, validates all settings
├── ingest.py                 # Loads PDFs from ./data/
├── chunker.py                # Splits pages into overlapping chunks
├── embedder.py               # Embeds chunks → FAISS or Pinecone
├── hybrid_retriever.py       # BM25 + dense retrieval (RRF fusion)
├── reranker.py               # CrossEncoder re-ranking
├── generator.py              # LLM answer generation (sync + streaming)
├── pipeline.py               # Orchestrates the full RAG pipeline
├── api.py                    # FastAPI server (/health, /query, /query/sync)
├── app.py                    # Streamlit UI
├── build_index.py            # One-shot: embed + index all PDFs
├── check_keys.py             # Validates API keys from .env
│
├── data/                     # Source PDF documents
└── vector_store/             # FAISS index + chunks.pkl (auto-generated)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/spandanaS-git/Construction_RAG_Assistant.git
cd Construction_RAG_Assistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Open .env and fill in your API keys (for Mode 2) or leave as-is for Mode 1
```

### 3. Build the Index

```bash
python build_index.py
```

### 4. Run the App

**Streamlit UI:**
```bash
streamlit run app.py
```
Open → **http://localhost:8501**

**FastAPI Server:**
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open → **http://localhost:8000/docs**

---

## 🔑 API Keys Setup

Copy `.env.example` to `.env` and fill in your keys:

```env
OPENAI_API_KEY=your-openai-api-key-here       # platform.openai.com/api-keys
PINECONE_API_KEY=your-pinecone-api-key-here   # app.pinecone.io → API Keys
```

> ⚠️ **Never commit your `.env` file.** It is already in `.gitignore`.  
> Only `.env.example` (with empty placeholders) is tracked by git.

---

## 💰 Cost Summary

| Item | Mode 1 (Local) | Mode 2 (Cloud) |
|---|---|---|
| Index build | $0.00 | ~$0.02 (one-time) |
| Per query (retrieval) | $0.00 | $0.00 (Pinecone free tier) |
| Per query (LLM) | $0.00 | ~$0.002 (GPT-3.5) |
| 100 demo queries | **$0.00** | **~$0.20** |

---

## 📚 Source Documents

The system is trained on the following construction safety and project management documents:

- OSHA Construction Safety Guide
- GTM Construction Safety Manual
- WSDOT Construction Manual
- TxDOT Bridge Inspection Manual
- Montana DOT Project Report

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embedding (Local) | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Embedding (Cloud) | OpenAI `text-embedding-3-small` |
| Vector Store (Local) | FAISS |
| Vector Store (Cloud) | Pinecone Serverless |
| Sparse Retrieval | BM25 (`rank_bm25`) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (Local) | `google/flan-t5-base` (HuggingFace) |
| LLM (Cloud) | OpenAI `gpt-3.5-turbo` |
| UI | Streamlit |
| API | FastAPI + Uvicorn |
| Config | `python-dotenv` |

---

## 📖 Detailed Run Guide

For full step-by-step instructions, mode switching, and API testing examples, see **[RUN_GUIDE.md](./RUN_GUIDE.md)**.

---

*Construction Safety RAG Assistant v2.0 — April 2026*
