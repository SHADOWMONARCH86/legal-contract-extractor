"""
Document extraction endpoints.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.extraction import ExtractionResponse
from app.services.extraction_service import ExtractionService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract entities from a legal PDF",
    response_description="Structured JSON with extracted legal entities",
)
async def extract_entities(file: UploadFile = File(..., description="PDF document (scanned or digital)")):
    """
    Accepts a PDF (digital or scanned) and returns extracted legal entities:

    - **dates**: Contract dates, effective dates, expiry dates
    - **parties**: Contracting party names and roles
    - **monetary_values**: Financial figures, fees, penalties
    - **termination_clauses**: Detected termination / exit conditions
    """
    # ── Validate file type ───────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    # ── Read and size-check ──────────────────────────────────────────────────
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB} MB.",
        )

    if len(contents) < 4 or contents[:4] != b"%PDF":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file does not appear to be a valid PDF.",
        )

    # ── Write to a temp file and process ─────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        logger.info(f"Processing document: {file.filename} ({len(contents) / 1024:.1f} KB)")
        service = ExtractionService()
        result = service.process(tmp_path, original_filename=file.filename or "unknown.pdf")
        return result
    except ValueError as e:
        logger.warning(f"Validation error for {file.filename}: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Extraction failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Entity extraction failed. The document may be corrupt or unsupported.",
        )
    finally:
        tmp_path.unlink(missing_ok=True)
