"""
database.py
-----------
SQLAlchemy engine, session factory, and declarative base for the
application's SQLite database. Provides both a FastAPI dependency
(`get_db`) and a context-manager (`db_session`) for use in services
and scripts.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import get_settings

settings = get_settings()

# SQLite requires this flag when accessed from multiple threads, which
# happens routinely under Uvicorn's threadpool for sync endpoints.
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""

    pass


def init_db() -> None:
    """Create all database tables. Safe to call multiple times."""
    import models  # noqa: F401  imported for side effect: registers models on Base

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Minimal forward-only migration: `create_all` never alters existing
    tables, so columns added to a model after a database was first
    created (e.g. `owner_id`) are appended here with ALTER TABLE.
    Pre-existing rows get NULL, which makes them invisible to every user."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue
            existing = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name not in existing:
                    col_type = column.type.compile(dialect=engine.dialect)
                    conn.execute(
                        text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}')
                    )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for use outside of FastAPI's dependency injection
    (e.g. inside background tasks or service-layer functions)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
