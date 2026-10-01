"""
main.py
--------
FastAPI application entrypoint for PDF Chat.

Run locally with:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000            (web UI)
    http://127.0.0.1:8000/docs       (Swagger UI)
    http://127.0.0.1:8000/redoc      (ReDoc)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from typing import Optional
from urllib.parse import quote

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import DEFAULT_SECRET_KEY, get_settings
from database import init_db
from models import User
from routers import auth, chat, documents, health, upload
from services.auth_service import get_optional_user

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("pdf_chat")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)
    if settings.secret_key == DEFAULT_SECRET_KEY:
        if settings.app_env == "production":
            raise RuntimeError(
                "SECRET_KEY is still the default value. Set a long random SECRET_KEY "
                "in .env before running in production - it signs login tokens."
            )
        logger.warning("SECRET_KEY is the default value. Change it before deploying.")
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. Upload/chat endpoints that call OpenAI will fail "
            "until you configure it in your .env file."
        )
    init_db()
    settings.ensure_directories()
    logger.info("Startup complete. Database and storage directories are ready.")
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "A production-ready AI PDF Chat application. Upload PDF documents, "
        "and ask questions answered via Retrieval-Augmented Generation (RAG) "
        "over your own documents, powered by LangChain, FAISS, and OpenAI."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    # Credentials (cookies) may only be shared with explicitly listed origins.
    allow_credentials="*" not in settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static files & templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- API routers ---
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(documents.router)
app.include_router(chat.router)


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.app_name})


def _page_for_user(request: Request, template: str, user: Optional[User]) -> Response:
    """Render an app page, or send anonymous visitors to the login page."""
    if user is None:
        return RedirectResponse(f"/login?next={quote(request.url.path)}", status_code=303)
    return templates.TemplateResponse(
        template, {"request": request, "app_name": settings.app_name, "user": user}
    )


@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "invite_required": bool(settings.registration_invite_code),
        },
    )


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page(request: Request, user: Optional[User] = Depends(get_optional_user)) -> Response:
    return _page_for_user(request, "upload.html", user)


@app.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_page(request: Request, user: Optional[User] = Depends(get_optional_user)) -> Response:
    return _page_for_user(request, "chat.html", user)


# ---------------------------------------------------------------------------
# Global error handling
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
