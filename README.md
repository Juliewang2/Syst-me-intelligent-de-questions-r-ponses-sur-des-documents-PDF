# 📚 PDF Chat: A RAG Document Q&A System with Hybrid Retrieval

Upload PDFs and ask questions as if you were chatting with ChatGPT. Answers are based **only on the document content**, with citations showing the source document and page number.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-dense%20retrieval-4267B2)
![BM25](https://img.shields.io/badge/BM25-sparse%20retrieval-orange)
![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

**Evaluation results (a 449-page textbook and a 40-question test set):** compared with the pure vector retrieval baseline, **hit@4 improved from 0.81 to 0.94**, **MRR from 0.70 to 0.88**, **answer correctness from 0.78 to 0.93**, and **faithfulness from 0.96 to 1.00**. See [Evaluation](#evaluation).

---

## Table of Contents

1. [The System in One Sentence](#the-system-in-one-sentence)
2. [Features](#features)
3. [The Two Core Pipelines](#the-two-core-pipelines)
4. [Improvements over the Baseline](#improvements-over-the-baseline)
5. [Evaluation](#evaluation)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Quick Start](#quick-start)
9. [Configuration](#configuration)
10. [API](#api)
11. [Tests](#tests)
12. [Known Limitations and Future Work](#known-limitations-and-future-work)

---

## The System in One Sentence

An LLM is like a **well-read professor who has never read your particular document**: if you ask it directly, "What does Article 5 of my contract say?", it does not know and may make up an answer.

**Retrieval-Augmented Generation (RAG)** first sends a **librarian** into your PDF to find the most relevant passages, then gives those passages and your question to the professor so that it can **answer only from the provided material**.

This project implements that "librarian + professor" workflow and makes the librarian more accurate: **vector retrieval understands meaning, BM25 recognizes keywords, and a reranker model selects the best passages**.

---

## Features

- **Multiple PDF upload and management**: drag-and-drop upload; ask questions across all documents or target a specific document
- **Scanned PDFs**: automatically runs OCR on pages without a text layer (Chinese and English)
- **Hybrid retrieval**: combines vector retrieval and BM25 with RRF, then applies LLM reranking and relevance thresholds
- **Source citations**: every answer includes the document name, page range, and relevance score
- **Streaming output**: returns text progressively over SSE for a ChatGPT-like experience
- **Multi-turn conversations**: conversation history is stored in SQLite; follow-up questions are rewritten into complete questions before retrieval
- **Authentication**: registration, login, JWT authentication, and complete data isolation between users
- **Structured summaries**: uses the OpenAI Responses API and JSON Schema to produce summaries in a fixed format
- **Evaluation framework**: hit@k, MRR, faithfulness, correctness, refusal accuracy, and analysis by question type

---

## The Two Core Pipelines

The system essentially does two things: **store documents** and **answer questions**.

### Pipeline A: Upload a PDF (Putting a Book in the Library)

```
User uploads a PDF (login required)
  → ① Validate: extension, file header, size ≤ 25 MB              routers/upload.py
  → ② Save to data/uploads/
  → ③ Extract text page by page; pages with fewer than 20 characters use OCR
       (pypdf → PyPDF2 fallback; OCR uses pypdfium2 rendering + RapidOCR) services/pdf_loader.py
  → ④ Create cross-page chunks: concatenate the document before splitting;
       1,000 characters per chunk, 200-character overlap, with start/end pages
                                                                    services/chunking_service.py
  → ⑤ Convert each chunk into a 1,536-dimensional vector          services/embedding_service.py
  → ⑥ Store and persist it in the user's isolated FAISS index     services/vector_store.py
  → ⑦ Store document metadata in SQLite (user, pages, chunks, status)
                                                                    models.py
```

### Pipeline B: Ask a Question (Having the Professor Answer)

```
User enters a question
  → ① Read the current conversation history                   services/conversation_memory.py
  → ② Rewrite the question: "What about the third point?" → an independent question
                                                               rag_service._condense_question
  → ③ Hybrid retrieval                                      services/retriever.py
       ├─ Vector retrieval: top 12 (discard cosine similarity < 0.2)
                                                               services/vector_store.py
       ├─ BM25 keyword retrieval: top 12 (jieba Chinese tokenization)
                                                               services/keyword_search.py
       ├─ Fuse both rankings with RRF
       └─ LLM reranking: score each passage from 0–10, discard < 4, keep the top 4
                                                               services/reranker.py
  → ④ Build a prompt: answer only from the material + material + history + question
                                                               services/prompt_templates.py
  → ⑤ gpt-4o-mini generates the answer and sends it to the browser over SSE
                                                               routers/chat.py
  → ⑥ Store the answer and citations for future follow-up questions
```

The main orchestration logic is in `services/rag_service.py`; the retrieval pipeline is in `services/retriever.py`.

---

## Improvements over the Baseline

The baseline is a standard **pure vector retrieval RAG** system. The following seven shortcomings were identified and addressed:

| # | Baseline problem | Improvement | Key files |
|---|---|---|---|
| 1 | **Scanned PDFs cannot be read** because there is no OCR | Render pages with fewer than the threshold number of characters as images, then recognize them with RapidOCR (ONNX, Chinese and English, installable with pip) | `pdf_loader.py` |
| 2 | **Vector retrieval only** performs poorly for exact terms such as model numbers, names, and terminology | Combine vector retrieval and BM25, fuse them with RRF, and apply LLM reranking with structured scores | `retriever.py`, `keyword_search.py`, `reranker.py` |
| 3 | **No relevance threshold**: always returns four passages, even when all are irrelevant | Apply two thresholds: cosine similarity ≥ 0.2 and reranking score ≥ 4/10; explicitly tell the model when everything is filtered out | `retriever.py` |
| 4 | Document filtering was **"search first, filter later"**: only 16 results were retrieved before filtering, so the target document could produce no results | Maintain an isolated index for each user; search the entire index when filtering, with no additional cost for a flat index | `vector_store.py` |
| 5 | **Page-based chunking** cuts sentences across page boundaries | Concatenate the document before chunking, infer each chunk's start/end page from character offsets, and add Chinese punctuation as split points | `chunking_service.py` |
| 6 | **No authentication**: anyone could call the API and consume API credits | Add registration and login; store salted PBKDF2 password hashes; use HttpOnly + SameSite cookies for JWTs; isolate user data; support invite codes; reject the default secret in production | `auth_service.py`, `routers/auth.py` |
| 7 | **No evaluation** | Build an evaluation framework inspired by RAGAS: hit@k, MRR, LLM-judged faithfulness/correctness/refusal accuracy, question-type analysis, and an automatic page-evidence verification script | `evaluation/` |

**Additional issues discovered and fixed during evaluation:**

- **Chinese paths prevented vector-store persistence**: FAISS C++ file I/O could not open paths containing Chinese characters on Windows. The index is now serialized to bytes first, then written by Python with an atomic temporary-file replacement.
- **BM25 failed on short documents**: the classic Okapi IDF can be zero or negative for words appearing in more than half of the passages, making keyword retrieval fail completely for documents with only a few chunks. The Lucene/Elasticsearch IDF formula is used instead.
- **OpenAI requests had no timeout**: one request became stuck for 84 minutes. All calls now have timeouts; reranking times out after 20 seconds and automatically falls back to the non-reranked ordering.

---

## Evaluation

### Experimental Setup

| Item | Details |
|---|---|
| Document | *Understanding Machine Learning* (Shalev-Shwartz & Ben-David), 449 pages → 1,080 chunks |
| Test set | **40 questions**, six types: terminology (8), paraphrased concepts (8), cross-chapter (5), formulas (5), Chinese questions about an English book (6), and unanswerable questions (8) |
| Annotations | Each question includes a reference answer, answer pages, and source evidence; `check_dataset.py` verifies that the evidence appears on the annotated pages |
| Development set | An additional 11-question development set for debugging, with no overlapping knowledge points; the test set was not used for tuning |
| Models | gpt-4o-mini (generation, reranking, and judging), text-embedding-3-small; four passages per question |

### Overall Results

| Configuration | hit@4 | MRR | Faithfulness | Correctness | Refusal accuracy | Median retrieval time |
|---|---|---|---|---|---|---|
| Pure vector (baseline) | 0.812 | 0.703 | 0.963 | 0.784 | 1.00 | 0.16 s |
| Vector + BM25 | 0.875 | 0.719 | 0.975 | 0.825 | 1.00 | 0.15 s |
| **Vector + BM25 + reranking** | **0.938** | **0.880** | **1.000** | **0.928** | 1.00 | 1.51 s |

Compared with the baseline: hit@4 **+12.5 percentage points** (+15% relative), MRR **+25%**, and correctness **+18%**.

### By Question Type (hit@4 / correctness)

| Question type | Pure vector | Vector + BM25 | + Reranking |
|---|---|---|---|
| Terminology (8) | 0.62 / 0.69 | **1.00** / 0.88 | 1.00 / **1.00** |
| Concepts (8) | 0.75 / 0.72 | 0.75 / 0.75 | **0.88 / 0.91** |
| Formulas (5) | 0.80 / 0.80 | 0.80 / 0.80 | 0.80 / **0.96** |
| Chinese questions (6) | 1.00 / 0.83 | 0.83 / 0.80 | 1.00 / 0.80 |
| Cross-chapter (5) | 1.00 / 0.96 | 1.00 / 0.92 | 1.00 / 0.96 |
| Unanswerable (8) | – / 1.00 | – / 1.00 | – / 1.00 |

### Conclusions

- **BM25 provides the largest gain on terminology questions** (hit rate 0.62 → 1.00), because proper nouns are a weakness of vector retrieval.
- **BM25 does not help paraphrased concept questions; reranking does** (correctness 0.72 → 0.91).
- **For Chinese questions about English documents, BM25 introduces noise** (hit rate 1.00 → 0.83), which reranking corrects.
- **Better retrieval reduces hallucinations**: the baseline hallucinated on two questions because the retrieved material was insufficient; with reranking, the relevant passages moved to the top and faithfulness reached 1.00.
- **Trade-off**: reranking adds another LLM call, increasing retrieval time from 0.16 s to 1.5 s.

### Limitations

- Forty questions is still a small sample. The hit-rate improvement came from four questions where pure vector retrieval missed and reranking succeeded, with no regressions. The direction is consistent, but the hit-rate difference alone is not statistically significant. MRR and correctness show larger gaps.
- Correctness is scored by an LLM judge and can be misclassified. For example, it may treat "maximize variance" and "minimize reconstruction error" as different even though they are equivalent. The overall scores may therefore be low, but the effect is consistent across configurations, so comparisons remain useful.
- The "pure vector" configuration uses the improved chunking strategy, so the comparison measures retrieval-method differences rather than the effect of chunking.

### Reproduction

```bash
# 1) Verify dataset annotations (does not call the API)
venv\Scripts\python.exe evaluation\check_dataset.py --pdf textbook.pdf --dataset evaluation\dataset_ml_book_test.json
# 2) Run the evaluation (about 15 minutes; approximately $0.10 in gpt-4o-mini costs)
venv\Scripts\python.exe evaluation\run_eval.py --pdf textbook.pdf --dataset evaluation\dataset_ml_book_test.json
# Evaluate retrieval only (cheaper), or run selected configurations
venv\Scripts\python.exe evaluation\run_eval.py --pdf textbook.pdf --retrieval-only --configs vector,hybrid
```

Evaluation uses an independent temporary index that is deleted automatically, so application data is not affected. Detailed results are saved in `evaluation/results/`.

---

## Tech Stack

### Core Technologies

| Category | Technology |
|---|---|
| Language | Python 3.12 |
| Web backend | FastAPI, Uvicorn, Jinja2; SSE streaming |
| LLM orchestration | LangChain 0.3 (Document Loader, TextSplitter, VectorStore, Retriever, PromptTemplate, LCEL) |
| LLMs | OpenAI gpt-4o-mini, text-embedding-3-small; Responses API structured output |
| Vector retrieval | FAISS |
| Data storage | SQLite + SQLAlchemy 2.0 |
| Configuration/validation | Pydantic v2, pydantic-settings, python-dotenv |
| PDF parsing | pypdf, PyPDF2 |
| Frontend | Native HTML/CSS/JavaScript, marked.js, DOMPurify |
| Testing | pytest, FastAPI TestClient |

### Added Technologies

| Category | Technology | Purpose |
|---|---|---|
| Sparse retrieval | **rank-bm25** (Lucene IDF) | BM25 keyword retrieval |
| Chinese tokenization | **jieba** | Tokenize Chinese text for BM25 |
| Result fusion | **RRF (Reciprocal Rank Fusion)** | Combine vector and BM25 rankings |
| Reranking | **LLM Reranker** (LangChain `with_structured_output` + JSON Schema) | Score candidate passages from 0–10 |
| OCR | **RapidOCR** (ONNX Runtime), **pypdfium2** | Render scanned pages and recognize text |
| Authentication | **PyJWT**, PBKDF2-HMAC-SHA256 (`hashlib`), HttpOnly/SameSite cookies | Login and data isolation |
| Evaluation | **LLM-as-a-judge**, custom hit@k/MRR/Faithfulness/Correctness metrics | Quantify RAG quality |
| Engineering | `asyncio.to_thread`, request timeouts and fallbacks, lightweight migrations, FAISS byte serialization | Reliability and compatibility |

---

## Project Structure

```
pdf-chat/
├── main.py                    Application entry point: routes, pages, login redirects, startup checks
├── config.py                  All configuration options (loaded from .env)
├── database.py / models.py    SQLite connection and lightweight migrations; User / Document / Conversation / ChatMessage
├── schemas.py                 API request and response schemas
├── routers/                   [API layer]
│   ├── auth.py                Registration / login / logout / current user
│   ├── upload.py              Upload and ingestion pipeline
│   ├── documents.py           Document list / deletion / structured summaries
│   ├── chat.py                Q&A (regular + SSE streaming) and conversation management
│   └── health.py              Health check
├── services/                  [Business layer]
│   ├── pdf_loader.py          Text extraction + OCR
│   ├── chunking_service.py    Cross-page chunking
│   ├── embedding_service.py   Text → vectors
│   ├── vector_store.py        Per-user FAISS indices
│   ├── keyword_search.py      BM25 + jieba
│   ├── reranker.py            LLM reranking
│   ├── retriever.py           Hybrid retrieval pipeline (vector → BM25 → RRF → reranking → top-k)
│   ├── prompt_templates.py    Prompts for answers / rewriting / reranking / summaries
│   ├── conversation_memory.py Conversation memory
│   ├── auth_service.py        Password hashing, JWT, current-user dependency
│   └── rag_service.py         RAG orchestration
├── evaluation/
│   ├── run_eval.py            Evaluation script (configuration comparison and question-type statistics)
│   ├── check_dataset.py       Automatic page-annotation verification
│   ├── dataset_ml_book_test.json   40-question test set
│   └── dataset.example.json        11-question development set
├── templates/ static/         Frontend pages, including the login page
├── tests/                     35 tests (retrieval, authentication, OCR, chunking, and more; offline)
└── learn/rag_minimal.py       50-line minimal RAG prototype for learning
```

---

## Quick Start

Python 3.12 is required (the locked dependency versions do not yet have packages for Python 3.14), along with an OpenAI API key.

```bash
# 1. Create a virtual environment and install dependencies
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Configure
copy .env.example .env        # then set OPENAI_API_KEY and change SECRET_KEY to a random string

# 3. Start
venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000, register an account, upload a PDF, and start asking questions. API documentation is available at `/docs`; the **Authorize** button on that page can be used with the token returned by login.

> macOS/Linux: replace `venv\Scripts\python.exe` with `venv/bin/python`.

---

## Configuration

See `.env.example` for the complete list. Common options include:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 200 | Chunk size and overlap |
| `RETRIEVER_K` | 4 | Number of passages ultimately provided to the model |
| `RETRIEVAL_CANDIDATES` | 12 | Candidate count for each retrieval method |
| `HYBRID_SEARCH_ENABLED` | true | Whether to enable BM25 |
| `MIN_VECTOR_SIMILARITY` | 0.2 | Vector similarity threshold |
| `RERANK_ENABLED` / `RERANK_MIN_SCORE` | true / 4 | Whether to rerank and the minimum reranking score |
| `OCR_ENABLED` / `OCR_MIN_CHARS` | true / 20 | Whether to enable OCR and the character threshold that triggers it |
| `OPENAI_TIMEOUT_SECONDS` / `RERANK_TIMEOUT_SECONDS` | 60 / 20 | Request timeouts |
| `SECRET_KEY` | placeholder | JWT signing key; **must be changed before deployment** |
| `REGISTRATION_INVITE_CODE` | empty | If set, registration requires this invite code |

---

## API

All endpoints except `/api/health` and registration/login require authentication (via cookie or `Authorization: Bearer <token>`).

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/register` · `/login` · `/logout` | Register / log in / log out |
| GET | `/api/auth/me` | Current user |
| POST | `/api/upload` | Upload one or more PDFs |
| GET / DELETE | `/api/documents` · `/api/documents/{id}` | List, inspect, and delete documents |
| POST | `/api/documents/{id}/summary` | Generate a structured summary |
| GET / POST | `/api/conversations` | List and create conversations |
| GET / DELETE | `/api/conversations/{id}[/messages]` | Read messages and delete conversations |
| POST | `/api/chat` · `/api/chat/stream` | Ask a question (regular / SSE streaming) |
| GET | `/api/health` | Health check |

---

## Tests

```bash
venv\Scripts\python.exe -m pytest
```

There are 35 tests, and **no API key is required**: retrieval tests use an offline hash-based embedding model instead of OpenAI. Coverage includes:

- Hybrid retrieval, RRF fusion, similarity thresholds, reranking filters, and failure fallbacks
- Complete results when filtering by document, Chinese BM25, and index persistence under Chinese paths
- User data isolation, JWT authentication, and cookie authentication
- OCR for scanned pages and page mapping for cross-page chunks

---

## Known Limitations and Future Work

- **Reranking cost**: LLM reranking adds approximately 1.5 seconds of latency. A local cross-encoder such as `bge-reranker` could reduce latency and cost.
- **Learned sparse retrieval**: models such as SPLADE could replace or complement BM25 and be compared using the existing evaluation framework.
- **Evaluation reliability**: expand the test set, use a stronger judge model, and rewrite reference answers as lists of required points.
- **Formula extraction**: mathematical formulas extracted from PDFs can become garbled, affecting formula-related questions.
- **Deployment**: add per-user rate limiting and usage-cost tracking; for multi-instance deployments, migrate to a managed vector database such as pgvector or Qdrant.

---

## License

MIT
