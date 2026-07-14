# Architecture Deep Dive

This document expands on the high-level architecture summarized in the root `README.md`.

## Layered design

```
┌─────────────────────────────────────────────────────────────┐
│  Presentation layer                                          │
│  templates/*.html + static/css + static/js                   │
│  (server-rendered Jinja2 pages, vanilla JS, SSE client)       │
└───────────────────────────┬───────────────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼───────────────────────────────────┐
│  API layer (FastAPI routers)                                  │
│  routers/health.py  routers/upload.py                         │
│  routers/documents.py  routers/chat.py                        │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│  Service layer (business logic, framework-agnostic)           │
│  pdf_loader → chunking_service → embedding_service             │
│  → vector_store → retriever → rag_service                     │
│  conversation_memory, prompt_templates                        │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│  Persistence layer                                             │
│  SQLite (documents, conversations, chat_messages)              │
│  FAISS local index (data/vector_store)                        │
│  Filesystem (data/uploads)                                    │
└─────────────────────────────────────────────────────────────┘
```

## Request lifecycle: uploading a PDF

1. `POST /api/upload` receives one or more `multipart/form-data` files.
2. `routers/upload.py` validates extension, magic bytes, and size.
3. The file is streamed to disk under `data/uploads/<uuid>.pdf`.
4. `services/pdf_loader.py` extracts text per page (LangChain `PyPDFLoader`,
   falling back to `pypdf` then `PyPDF2` if needed).
5. `services/chunking_service.py` splits each page's text into overlapping
   chunks using `RecursiveCharacterTextSplitter`, attaching
   `document_id` / `document_name` / `page` / `chunk_index` metadata.
6. `services/embedding_service.py` provides a cached `OpenAIEmbeddings` client.
7. `services/vector_store.py` embeds the chunks and adds them to the shared
   FAISS index, then persists the index to disk.
8. A `Document` row is written to SQLite with status `ready` (or `failed`
   with an error message if any step raised).

## Request lifecycle: asking a question (streaming)

1. `POST /api/chat/stream` receives `{ conversation_id?, document_id?, message }`.
2. `services/conversation_memory.py` loads or creates a `Conversation` and
   fetches the last N turns as LangChain `BaseMessage` objects.
3. `services/rag_service.py`:
   a. If there is prior history, a small LLM call rewrites the follow-up
      question into a standalone question (`CONDENSE_QUESTION_PROMPT`).
   b. `services/retriever.py` runs a FAISS similarity search (optionally
      filtered to one `document_id`) and returns the top-k chunks.
   c. Chunks are formatted into a numbered context block.
   d. `RAG_PROMPT` (system + history placeholder + question) is filled in
      and sent to `ChatOpenAI` with `streaming=True`.
   e. Tokens are yielded as they arrive from the LLM.
4. The router wraps each pipeline event in a Server-Sent Event: `start`,
   `sources`, `token` (repeated), `error` (if any), then `close`.
5. Once streaming completes, the full assistant answer and its sources are
   persisted as a `ChatMessage` row.

## Why a single shared FAISS index instead of one-per-document?

A single index keeps "search across everything I've uploaded" queries fast
(one FAISS call instead of N), while `document_id` metadata + FAISS's
post-filtering still allows scoping a query to exactly one document. This
mirrors how most production RAG systems structure a single-tenant knowledge
base with document-level access control expressed as metadata filters.

## Structured output demo

`routers/documents.py`'s `POST /api/documents/{id}/summary` endpoint calls
`services/rag_service.generate_structured_summary`, which uses the **OpenAI
Python SDK directly** (not LangChain) to call the Responses API with
`text.format.type = "json_schema"` and `strict = True`. This guarantees the
model's output parses cleanly into `DocumentSummaryResponse` every time,
demonstrating structured output independent of LangChain's own output
parsers.

## Extensibility points

- Swap FAISS for another LangChain-supported vector store (Chroma, Pinecone,
  Weaviate, pgvector) by rewriting `services/vector_store.py`'s persistence
  functions; `services/retriever.py` and everything above it is unaffected.
- Swap SQLite for Postgres by changing `DATABASE_URL`; SQLAlchemy handles
  the rest (drop the SQLite-only `connect_args` in `database.py`).
- Add authentication by inserting a FastAPI dependency into the routers
  that currently only depend on `get_db`.
