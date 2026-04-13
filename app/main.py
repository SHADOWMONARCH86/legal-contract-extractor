"""
Legal Contract Entity Extractor — FastAPI Application Entry Point
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import extraction, health
from app.utils.logger import get_logger
from fastapi.responses import RedirectResponse

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    logger.info("🚀 Starting Legal Contract Entity Extractor...")
    # Warm up NER model on startup to avoid cold-start latency on first request
    try:
        from app.services.ner_service import NERService
        NERService.get_instance()
        logger.info("✅ NER model loaded and ready.")
    except Exception as e:
        logger.error(f"❌ Failed to load NER model on startup: {e}")
    yield
    logger.info("🛑 Shutting down application.")


app = FastAPI(
    title="Legal Contract Entity Extractor",
    description=(
        "Production-grade NLP API for extracting structured entities "
        "(dates, party names, monetary values, termination clauses) from legal PDF documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} ({duration:.1f}ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check server logs."},
    )

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")
# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, tags=["Health"])
app.include_router(extraction.router, prefix="/api/v1", tags=["Extraction"])
