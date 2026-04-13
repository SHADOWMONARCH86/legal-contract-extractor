"""
Health check endpoints.
"""

import platform
import sys

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check():
    """Returns service liveness status and runtime metadata."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "legal-contract-extractor",
            "version": "1.0.0",
            "python": sys.version,
            "platform": platform.system(),
        },
    )
