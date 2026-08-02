# 📚 PDF Chat - AI-Powered RAG Application

**PDF Chat** is a production-ready, ChatGPT-style application that lets you upload PDF
documents and have grounded, cited conversations with them using **Retrieval-Augmented
Generation (RAG)**. It's built with **FastAPI**, **LangChain**, **FAISS**, and the
**OpenAI API** (including the latest **Responses API** with structured output), and ships
with a polished, responsive, dark-mode-ready web UI.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-vector--search-4267B2)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Concepts Explained](#concepts-explained)
   - [What is LangChain?](#what-is-langchain)
   - [What is RAG?](#what-is-rag-retrieval-augmented-generation)
   - [Embeddings](#embeddings)
   - [Chunking](#chunking)
   - [Vector Databases](#vector-databases)
   - [FAISS](#faiss)
   - [Prompt Templates](#prompt-templates)
   - [Conversation Memory](#conversation-memory)
3. [Architecture](#architecture)
4. [Folder Structure](#folder-structure)
5. [Installation Guide](#installation-guide)
6. [Visual Studio Code Setup](#visual-studio-code-setup)
7. [Running the Application](#running-the-application)
8. [Swagger & ReDoc API Docs](#swagger--redoc-api-docs)
9. [Example Questions](#example-questions)
10. [Screenshots](#screenshots)
11. [Deployment Guide](#deployment-guide)
12. [Troubleshooting](#troubleshooting)
13. [Future Improvements](#future-improvements)
14. [License](#license)

---

## What This Project Does

1. **Upload** one or more PDF documents through a drag-and-drop web UI or the API.
2. The app **extracts text**, **splits it into chunks**, and converts each chunk into a
   vector **embedding**.
3. Embeddings are stored in a local **FAISS** vector index.
4. When you ask a question, the app **retrieves** the most relevant chunks, injects them
   into a **prompt template** alongside your **conversation history**, and asks an OpenAI
   chat model to answer - **streaming** the response token-by-token, just like ChatGPT.
5. Every answer includes **source citations** (document name + page number) so you can
   verify where the information came from.

---

## Concepts Explained

### What is LangChain?

[LangChain](https://www.langchain.com/) is an open-source framework for building
applications powered by large language models (LLMs). Instead of writing raw API calls to
OpenAI (or any other model provider) by hand, LangChain gives you composable building
blocks:

- **Document loaders** - read files (PDFs, web pages, databases) into a standard
  `Document` object with `page_content` and `metadata`.
- **Text splitters** - break long documents into smaller chunks suitable for embedding.
- **Embeddings models** - wrappers around embedding APIs (OpenAI, HuggingFace, etc.).
- **Vector stores** - a unified interface over FAISS, Chroma, Pinecone, and others.
- **Retrievers** - an abstraction for "given a query, return relevant documents."
- **Prompt templates** - reusable, parameterized prompts.
- **Chains / LCEL (LangChain Expression Language)** - the `|` pipe operator lets you
  compose retrievers, prompts, models, and parsers into a single runnable pipeline, e.g.:

  ```python
  chain = RAG_PROMPT | llm | StrOutputParser()
  answer = chain.invoke({"context": context, "question": question, "chat_history": history})
  ```

This project uses LangChain for document loading (`PyPDFLoader`), text splitting
(`RecursiveCharacterTextSplitter`), embeddings (`OpenAIEmbeddings`), the vector store
(`FAISS`), a custom `BaseRetriever`, prompt templates (`ChatPromptTemplate`), and LCEL
chains (see `services/rag_service.py`).

### What is RAG (Retrieval-Augmented Generation)?

LLMs only "know" what was in their training data - they have no knowledge of your private
PDF. **RAG** solves this by:

1. **Retrieving** the most relevant pieces of your document(s) for a given question
   (via semantic/vector search), and
2. **Augmenting** the LLM's prompt with that retrieved text before asking it to
   **generate** an answer.

This grounds the model's response in your actual data, dramatically reducing
hallucination and enabling the model to answer questions about content it has never seen
during training. The full pipeline implemented here is:

```
PDF Upload -> Text Extraction -> Chunking -> Embeddings -> FAISS -> Retriever -> LLM -> Final Answer
```

### Embeddings

An **embedding** is a numerical vector (e.g. 1536 floating-point numbers for
`text-embedding-3-small`) that represents the _meaning_ of a piece of text. Texts with
similar meaning end up with vectors that are close together in that high-dimensional
space (measured via cosine similarity or L2 distance). This project embeds every chunk of
every uploaded PDF using OpenAI's embedding model (`services/embedding_service.py`), and
embeds each user question the same way at query time so they can be compared.

### Chunking

LLMs and embedding models have finite context windows, and retrieval works better on
focused, topically coherent passages rather than entire documents. **Chunking** splits
extracted PDF text into overlapping windows (default: 1000 characters with 200 characters
of overlap) using LangChain's `RecursiveCharacterTextSplitter`, which tries to split on
paragraph boundaries first, then sentences, then words - so chunks stay semantically
coherent. See `services/chunking_service.py`.

### Vector Databases

A **vector database** (or vector store) is a database optimized for storing embeddings
and performing fast **similarity search** - "find the k vectors closest to this query
vector." Unlike a traditional keyword search (which matches exact words), vector search
matches _meaning_, so a question like "How much does it cost?" can retrieve a chunk that
says "The subscription is priced at $49/month" even though no words overlap.

### FAISS

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is an
open-source library for efficient similarity search over dense vectors. This project uses
LangChain's `FAISS` vector store wrapper to:

- Build an index from document chunk embeddings (`FAISS.from_documents` /
  `add_documents`)
- Persist the index to disk (`save_local` / `load_local`) under `data/vector_store/` so
  it survives restarts
- Run `similarity_search` queries, optionally filtered by `document_id` metadata so you
  can scope a question to a single uploaded file

See `services/vector_store.py`.

### Prompt Templates

A **prompt template** is a reusable, parameterized prompt string with placeholders (e.g.
`{context}`, `{question}`) that get filled in at runtime. LangChain's
`ChatPromptTemplate` also supports a `MessagesPlaceholder` for injecting an entire chat
history as structured messages rather than a flat string. This project defines three
templates in `services/prompt_templates.py`:

- `RAG_PROMPT` - the main answer-generation prompt (system instructions + retrieved
  context + chat history + question)
- `CONDENSE_QUESTION_PROMPT` - rewrites a follow-up question ("What about page 3?") into
  a standalone question using the conversation history, before retrieval
- A structured-output prompt used for AI-generated document summaries

### Conversation Memory

Unlike a single request/response API call, a chat experience needs to remember prior
turns so follow-up questions make sense. This project implements **persistent**
conversation memory backed by SQLite (`services/conversation_memory.py`) rather than an
in-memory-only object: every user and assistant message is saved to the `chat_messages`
table, associated with a `Conversation` row. On each new question, the last N messages
(configurable via `MAX_HISTORY_MESSAGES`) are loaded and converted into LangChain
`HumanMessage` / `AIMessage` objects, which are fed into the prompt template's
`chat_history` placeholder - giving the model full context of the conversation so far,
and surviving server restarts since it's stored in the database.

---

## Architecture

```
┌────────────┐      ┌──────────────┐      ┌────────────────┐      ┌───────────────┐
│  Browser    │◄────►│   FastAPI    │◄────►│  Service layer  │◄────►│   OpenAI API   │
│  (HTML/JS)  │ SSE  │   routers/*  │      │  services/*     │      │ (Chat + Embed  │
└────────────┘      └──────┬───────┘      └───────┬────────┘      │  + Responses)  │
                            │                       │               └───────────────┘
                     ┌──────▼───────┐        ┌──────▼────────┐
                     │   SQLite     │        │  FAISS index   │
                     │ (documents,  │        │ (data/vector_  │
                     │ conversations,│        │  store/)       │
                     │ chat_messages)│        └────────────────┘
                     └──────────────┘
```

**RAG pipeline:**

```
PDF Upload
   │
   ▼
Text Extraction (services/pdf_loader.py - pypdf / PyPDF2 / LangChain PyPDFLoader)
   │
   ▼
Chunking (services/chunking_service.py - RecursiveCharacterTextSplitter)
   │
   ▼
Embeddings (services/embedding_service.py - OpenAIEmbeddings)
   │
   ▼
FAISS (services/vector_store.py - persisted local index)
   │
   ▼
Retriever (services/retriever.py - BaseRetriever, top-k similarity search)
   │
   ▼
LLM (services/rag_service.py - ChatOpenAI, streamed via LCEL)
   │
   ▼
Final Answer (with source citations, saved to conversation history)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a deeper, layer-by-layer breakdown
including full request lifecycles.

---

## Folder Structure

```
pdf-chat/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── .env.example
├── pytest.ini
├── main.py # FastAPI app entrypoint
├── config.py # Pydantic Settings (env-driven config)
├── database.py # SQLAlchemy engine/session
├── models.py # ORM models: Document, Conversation, ChatMessage
├── schemas.py # Pydantic v2 request/response schemas
├── routers/
│   ├── chat.py # /api/chat, /api/chat/stream, /api/conversations
│   ├── upload.py # /api/upload
│   ├── documents.py # /api/documents (list/get/delete/summary)
│   └── health.py # /api/health
├── services/
│   ├── pdf_loader.py # PDF text extraction
│   ├── chunking_service.py # Text splitting
│   ├── embedding_service.py # OpenAI embeddings client
│   ├── vector_store.py # FAISS persistence & search
│   ├── retriever.py # LangChain BaseRetriever wrapper
│   ├── rag_service.py # LCEL RAG chain, streaming, structured output
│   ├── conversation_memory.py # DB-backed chat history
│   └── prompt_templates.py # ChatPromptTemplate definitions
├── templates/
│   ├── index.html # Landing page
│   ├── chat.html # Chat UI
│   └── upload.html # Upload UI
├── static/
│   ├── css/style.css # Full design system (light + dark)
│   └── js/
│       ├── chat.js # SSE streaming, markdown rendering, sidebar
│       ├── upload.js # Drag-and-drop upload + progress
│       └── theme.js # Dark mode toggle
├── tests/
│   ├── test_health.py
│   └── test_chunking.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── screenshots/
└── data/ # created at runtime (gitignored)
    ├── uploads/
    ├── vector_store/
    └── pdf_chat.db
```

---

## Installation Guide

### 1. Python Installation

This project requires **Python 3.12 or newer**.

- **Windows**: Download from [python.org/downloads](https://www.python.org/downloads/)
  and make sure to check _"Add Python to PATH"_ during installation.
- **macOS**: `brew install python@3.12` (requires [Homebrew](https://brew.sh/)), or
  download from python.org.
- **Linux (Debian/Ubuntu)**: `sudo apt update && sudo apt install python3.12 python3.12-venv`

Verify your installation:

```bash
python3 --version
# Python 3.12.x
```

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-chat.git
cd pdf-chat
```

### 3. Create a Virtual Environment

A virtual environment keeps this project's dependencies isolated from your system Python.

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

You'll know it worked when your terminal prompt is prefixed with `(venv)`.

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Your OpenAI API Key

Copy the example environment file:

```bash
cp .env.example .env # macOS/Linux
copy .env.example .env # Windows
```

Open `.env` in your editor and set:

```env
OPENAI_API_KEY=sk-your-real-key-here
```

You can generate a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
The app will still boot without a key (so you can view the UI), but upload/chat requests
that call OpenAI will fail until it's set.

### 6. Initialize and Run

The SQLite database and FAISS index are created automatically on first run - no manual
migration step needed. Just start the server (see [Running the Application](#running-the-application)).

---

## Visual Studio Code Setup

This repository ships with a `.vscode/` folder pre-configured for a smooth VS Code
experience:

1. **Install the Python extension**: open the Extensions panel (`Ctrl+Shift+X` /
   `Cmd+Shift+X`) and install `ms-python.python` (VS Code will also suggest it
   automatically via `.vscode/extensions.json`).
2. **Open the folder**: `File -> Open Folder...` and select the `pdf-chat` directory.
3. **Select the interpreter**: press `Ctrl+Shift+P` / `Cmd+Shift+P`, run
   `Python: Select Interpreter`, and choose the one inside `./venv`.
4. **Run the server with the debugger**: open the "Run and Debug" panel (`Ctrl+Shift+D`)
   and choose **"FastAPI: Uvicorn (main:app)"** from the dropdown, then press `F5`. This
   launches Uvicorn with `--reload` and loads environment variables from `.env`
   automatically (see `.vscode/launch.json`).
5. **Run tests from the Testing panel**: the Python extension will auto-discover
   `tests/test_*.py` via `pytest.ini`. Click the flask/beaker icon in the Activity Bar to
   run/debug individual tests.
6. **Integrated terminal**: you can always just run `uvicorn main:app --reload` directly
   in VS Code's integrated terminal (`` Ctrl+` ``) instead of using the debugger.

---

## Running the Application

With your virtual environment activated and dependencies installed:

```bash
uvicorn main:app --reload
```

You should see log output ending with something like:

```
INFO | pdf_chat | Starting PDF Chat (env=development)
INFO | pdf_chat | Startup complete. Database and storage directories are ready.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Then open your browser to:

| URL                              | Description                       |
| -------------------------------- | --------------------------------- |
| http://127.0.0.1:8000            | Landing page                      |
| http://127.0.0.1:8000/upload     | Upload PDF documents              |
| http://127.0.0.1:8000/chat       | Chat with your documents          |
| http://127.0.0.1:8000/docs       | Swagger UI (interactive API docs) |
| http://127.0.0.1:8000/redoc      | ReDoc (reference-style API docs)  |
| http://127.0.0.1:8000/api/health | Health check (JSON)               |

You can also run it directly via Python (uses the `if __name__ == "__main__"` block in
`main.py`):

```bash
python main.py
```

---

## Swagger & ReDoc

FastAPI automatically generates OpenAPI documentation from the routers' type hints and
docstrings:

- **Swagger UI** (`/docs`) - an interactive playground where you can expand each
  endpoint, fill in parameters, and execute real requests directly from the browser.
  Great for testing `/api/upload`, `/api/chat`, and `/api/documents/{id}/summary` without
  writing any client code.
- **ReDoc** (`/redoc`) - a clean, read-only reference view of the same OpenAPI schema,
  better suited for sharing with API consumers as documentation.

Both are generated live from `schemas.py` and the route definitions in `routers/`, so
they always stay in sync with the actual API.

---

## Example Questions

Once you've uploaded a PDF, try asking things like:

- _"Summarize this document in 3 bullet points."_
- _"What are the key takeaways from this document?"_
- _"Are there any dates, deadlines, or dollar amounts mentioned?"_
- _"Who are the main people or organizations mentioned in this document?"_
- _"Explain the section about [topic] like I'm five."_
- _"What does this document say about [specific term]?"_
- Follow-ups work too, thanks to conversation memory: _"Can you go deeper on the second
  point?"_

---

## Screenshots

> Replace these placeholders with real screenshots once you've run the app locally.

**Landing page**

![Landing page](docs/screenshots/landing.png)

**Upload page (drag & drop)**

![Upload page](docs/screenshots/upload.png)

**Chat interface (streaming answer with citations)**

![Chat interface](docs/screenshots/chat.png)

**Dark mode**

![Dark mode](docs/screenshots/dark-mode.png)

**Swagger UI**

![Swagger UI](docs/screenshots/swagger.png)

---

## Deployment Guide

### Option A - Docker

Create a `Dockerfile` (not included by default, since this repo targets local/VS Code
development, but easy to add):

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/uploads data/vector_store

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t pdf-chat .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data pdf-chat
```

Mounting `./data` as a volume ensures your SQLite database and FAISS index persist across
container restarts.

### Option B - A traditional VM / bare metal server

1. Provision a server (e.g. Ubuntu 22.04+), install Python 3.12+.
2. Clone the repo, create a virtual environment, install `requirements.txt`.
3. Set environment variables (via `.env` or your process manager's env config).
4. Run behind a production ASGI setup:

   ```bash
   pip install gunicorn
   gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4 -b 0.0.0.0:8000
   ```

5. Put a reverse proxy (Nginx, Caddy) in front for TLS termination and static file
   caching.
6. Use a process supervisor (systemd, supervisord) to keep the app running and restart it
   on failure.

### Option C - Platform-as-a-Service (Render, Railway, Fly.io, etc.)

Most PaaS providers can build directly from `requirements.txt` and a start command:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set `OPENAI_API_KEY` (and any other `.env` values) as environment variables/secrets in
the platform's dashboard rather than committing `.env`. For persistence, either attach a
volume for `data/`, or swap SQLite/FAISS for managed services (Postgres + a hosted vector
DB) for a fully stateless deployment - see `docs/ARCHITECTURE.md#extensibility-points`.

### Production checklist

- [ ] Set a strong, unique `SECRET_KEY`
- [ ] Set `DEBUG=false` and `APP_ENV=production`
- [ ] Restrict `ALLOWED_ORIGINS` to your actual frontend domain(s) instead of `["*"]`
- [ ] Put the app behind HTTPS (via a reverse proxy or platform-managed TLS)
- [ ] Configure log aggregation (the app logs to stdout in a structured format)
- [ ] Set appropriate `MAX_UPLOAD_SIZE_MB` for your infrastructure
- [ ] Back up `data/pdf_chat.db` and `data/vector_store/` regularly if using local storage

---

## Troubleshooting

**`OPENAI_API_KEY is not set` warning on startup**
Copy `.env.example` to `.env` and fill in a real key. The app boots without one but chat
and upload requests that call OpenAI will fail with an authentication error.

**`ModuleNotFoundError` for a package like `langchain_openai`**
Make sure your virtual environment is activated (`(venv)` should be visible in your
prompt) and re-run `pip install -r requirements.txt`.

**PDF upload succeeds but status is `failed`**
Check the `error_message` field on the document (visible in the sidebar tooltip / via
`GET /api/documents/{id}`). Common causes:

- The PDF is a scanned image with no text layer (OCR is not included in this project -
  see [Future Improvements](#future-improvements)).
- The PDF is corrupted or password-protected.

**`RateLimitError` or `AuthenticationError` from OpenAI**
Verify your API key is correct and has available quota/billing set up on your OpenAI
account.

**Chat answers say "the context does not contain enough information"**
This is expected, grounded behavior - the model is instructed not to answer from outside
knowledge. Make sure you've selected the right document scope (or "All documents") in the
chat sidebar, and that the document actually contains the information you're asking
about.

**Streaming responses don't appear / hang in the browser**
Some corporate proxies or ad blockers buffer or block `text/event-stream` responses.
Try a different network, or check the browser console/network tab for the
`/api/chat/stream` request.

**Port 8000 already in use**
Run on a different port: `uvicorn main:app --reload --port 8001`.

**`sqlite3.OperationalError: database is locked`**
This can happen under heavy concurrent write load with SQLite. For production traffic,
switch `DATABASE_URL` to a Postgres connection string - SQLAlchemy handles the rest.

**Vector search returns irrelevant results**
Try lowering `CHUNK_SIZE` for more granular chunks, or increasing `RETRIEVER_K` to widen
the pool of retrieved chunks per question.

---

## Future Improvements

- **OCR support** for scanned/image-only PDFs (e.g. via `pytesseract` or a cloud OCR API)
- **Multi-user auth** (JWT/OAuth) with per-user document isolation
- **Hybrid search** (BM25 keyword search + vector search) for improved retrieval on exact
  terms, codes, and identifiers
- **Re-ranking** retrieved chunks with a cross-encoder before generation
- **Streaming summary generation** for the structured-output summary endpoint
- **Support for other file types** (.docx, .txt, .md, web URLs) via additional LangChain
  loaders
- **Swap to a managed vector database** (Pinecone, Weaviate, pgvector) for horizontal
  scale and multi-instance deployments
- **Usage/cost tracking dashboard** for OpenAI API spend per user or document
- **Automated evaluation** (e.g. RAGAS) to measure answer faithfulness and retrieval
  precision over a test question set
- **WebSocket-based chat** as an alternative to SSE for bidirectional features (e.g.
  typing indicators, cancel-generation)

---

## License

This project is licensed under the [MIT License](LICENSE).
