"""
Pydantic schemas for API request and response contracts.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DateEntity(BaseModel):
    value: str = Field(..., example="2024-01-15", description="Normalized date string (ISO-8601 when possible)")
    raw_text: str = Field(..., example="January 15, 2024", description="Original text as found in document")
    label: str = Field(..., example="EFFECTIVE_DATE", description="Date role/label")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.92)


class PartyEntity(BaseModel):
    name: str = Field(..., example="Acme Corporation", description="Extracted party name")
    role: Optional[str] = Field(None, example="VENDOR", description="Inferred role: VENDOR, CLIENT, EMPLOYER, etc.")
    raw_text: str = Field(..., example="Acme Corporation, a Delaware corporation")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.88)


class MonetaryEntity(BaseModel):
    amount: Optional[float] = Field(None, example=150000.0, description="Parsed numeric amount")
    currency: Optional[str] = Field(None, example="USD", description="ISO 4217 currency code")
    raw_text: str = Field(..., example="USD 150,000")
    context: Optional[str] = Field(None, example="annual service fee", description="Surrounding clause context")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.95)


class TerminationClause(BaseModel):
    text: str = Field(..., description="Full termination clause text")
    clause_type: str = Field(..., example="TERMINATION_FOR_CAUSE", description="Clause classification")
    notice_period: Optional[str] = Field(None, example="30 days", description="Required notice period if mentioned")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.85)


class ProcessingMetadata(BaseModel):
    filename: str
    file_size_bytes: int
    ocr_applied: bool = Field(..., description="Whether OCR was used to extract text")
    pages_processed: int
    processing_time_ms: float
    model_version: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class ExtractionResponse(BaseModel):
    success: bool = True
    document_id: str = Field(..., description="Unique processing ID for this request")
    metadata: ProcessingMetadata
    entities: "EntityBundle"

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "a3f9c12b-4e77-4a1a-bf3c-1234567890ab",
                "metadata": {
                    "filename": "service_agreement.pdf",
                    "file_size_bytes": 204800,
                    "ocr_applied": False,
                    "pages_processed": 12,
                    "processing_time_ms": 1423.5,
                    "model_version": "legal-ner-bert-v1",
                    "extracted_at": "2024-06-01T10:30:00Z",
                },
                "entities": {
                    "dates": [
                        {
                            "value": "2024-01-15",
                            "raw_text": "January 15, 2024",
                            "label": "EFFECTIVE_DATE",
                            "confidence": 0.92,
                        }
                    ],
                    "parties": [
                        {
                            "name": "Acme Corporation",
                            "role": "VENDOR",
                            "raw_text": "Acme Corporation, a Delaware corporation",
                            "confidence": 0.88,
                        }
                    ],
                    "monetary_values": [
                        {
                            "amount": 150000.0,
                            "currency": "USD",
                            "raw_text": "USD 150,000",
                            "context": "annual service fee",
                            "confidence": 0.95,
                        }
                    ],
                    "termination_clauses": [
                        {
                            "text": "Either party may terminate this Agreement upon 30 days written notice.",
                            "clause_type": "TERMINATION_FOR_CONVENIENCE",
                            "notice_period": "30 days",
                            "confidence": 0.85,
                        }
                    ],
                },
            }
        }


class EntityBundle(BaseModel):
    dates: List[DateEntity] = []
    parties: List[PartyEntity] = []
    monetary_values: List[MonetaryEntity] = []
    termination_clauses: List[TerminationClause] = []


ExtractionResponse.model_rebuild()
