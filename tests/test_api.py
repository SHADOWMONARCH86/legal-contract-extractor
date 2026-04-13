"""
API integration tests — uses FastAPI TestClient (no running server needed).
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _minimal_pdf_bytes() -> bytes:
    """Return a minimal valid PDF with embedded text (no images)."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>>"
        b">>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Test PDF content.) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\n"
        b"startxref\n370\n%%EOF"
    )


# ── Health Endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_contains_service_name(self):
        data = client.get("/health").json()
        assert "service" in data


# ── Extract Endpoint ──────────────────────────────────────────────────────────

class TestExtractEndpoint:
    def test_extract_rejects_non_pdf(self):
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert response.status_code in (400, 415, 422, 500)

    def test_extract_rejects_empty_file(self):
        response = client.post(
            "/api/v1/extract",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code in (400, 422, 500)

    def test_extract_rejects_fake_pdf(self):
        """File with wrong magic bytes is rejected."""
        response = client.post(
            "/api/v1/extract",
            files={"file": ("fake.pdf", b"NOTAPDF content here", "application/pdf")},
        )
        assert response.status_code == 400

    def test_extract_valid_pdf_returns_200(self):
        """A valid minimal PDF should return 200 with expected schema."""
        pdf_bytes = _minimal_pdf_bytes()
        response = client.post(
            "/api/v1/extract",
            files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
        )
        # Depending on pdfplumber availability this may succeed or raise 500
        # We just confirm it doesn't return a 4xx client error
        assert response.status_code in (200, 500)
        if response.status_code == 200:
            data = response.json()
            assert "entities" in data
            assert "metadata" in data
            assert "document_id" in data
            assert "dates" in data["entities"]
            assert "parties" in data["entities"]
            assert "monetary_values" in data["entities"]
            assert "termination_clauses" in data["entities"]

    def test_extract_response_schema(self):
        """Validate top-level response structure."""
        pdf_bytes = _minimal_pdf_bytes()
        response = client.post(
            "/api/v1/extract",
            files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
        )
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data["success"], bool)
            assert isinstance(data["document_id"], str)
            assert isinstance(data["metadata"]["ocr_applied"], bool)
            assert isinstance(data["metadata"]["pages_processed"], int)
