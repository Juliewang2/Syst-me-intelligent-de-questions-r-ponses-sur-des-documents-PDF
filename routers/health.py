"""
routers/health.py
-------------------
Simple health/readiness endpoint used by orchestrators (Docker,
Kubernetes, load balancers) and for quick manual sanity checks.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from config import get_settings
from schemas import HealthResponse

router = APIRouter(prefix="/api/health", tags=["Health"])
settings = get_settings()


@router.get("", response_model=HealthResponse, summary="Health check")
def health_check() -> HealthResponse:
    """Report application status and basic configuration diagnostics."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        openai_configured=bool(settings.openai_api_key),
        timestamp=datetime.now(timezone.utc),
    )
