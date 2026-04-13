# ⚖️ Legal Contract Entity Extractor

> **Production-grade NLP system** for automated extraction of structured entities from legal PDF contracts — handles both digital and scanned documents.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Folder Structure](#-folder-structure)
- [Tech Stack](#-tech-stack)
- [Pipeline Design](#-pipeline-design)
- [Extracted Entities](#-extracted-entities)
- [Setup — Local](#-setup--local-development)
- [Setup — Docker](#-setup--docker)
- [API Reference](#-api-reference)
- [Training a Custom Model](#-training-a-custom-ner-model)
- [Sample Output](#-sample-json-output)
- [Running Tests](#-running-tests)
- [MLflow Experiment Tracking](#-mlflow-experiment-tracking)
- [Roadmap](#-roadmap)

---

## 🎯 Overview

Legal teams review hundreds of contracts per week. This system automates the extraction of critical entities from legal PDFs — whether they are digitally generated or scanned — using a layered NLP pipeline:

| Entity Type | Examples |
|---|---|
| **Dates** | Effective date, expiry date, signing date |
| **Party Names** | Vendor, Client, Employer, Employee |
| **Monetary Values** | Fees, penalties, payment amounts |
| **Termination Clauses** | Notice periods, for-cause/convenience provisions |

All output is returned as clean, structured **JSON** via a REST API.

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           CLIENT (curl / UI / SDK)       │
                    └──────────────────┬──────────────────────┘
                                       │  POST /api/v1/extract
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │          FastAPI Application             │
                    │   ┌──────────────────────────────────┐  │
                    │   │      ExtractionService           │  │
                    │   │  (Pipeline Orchestrator)         │  │
                    │   └──────────┬───────────────────────┘  │
                    └─────────────┼───────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────────┐
          │                       │                           │
          ▼                       ▼                           ▼
  ┌───────────────┐     ┌─────────────────┐       ┌──────────────────┐
  │  PDFUtils     │     │   TextCleaner   │       │   NERService     │
  │               │     │                 │       │  (Singleton)     │
  │ pdfplumber    │     │ • Ligatures     │       │                  │
  │ (digital PDF) │     │ • OCR artefacts │       │ HuggingFace BERT │
  │               │     │ • Whitespace    │       │   ↓ (fallback)   │
  │     ↓ (if     │     │ • Unicode norm  │       │ spaCy NER        │
  │   scanned)    │     └─────────────────┘       │   ↓ (fallback)   │
  │               │                               │ Rule-based regex │
  │ OCRPipeline   │                               └────────┬─────────┘
  │ pdf2image +   │                                        │
  │ Tesseract OCR │                               ┌────────▼─────────┐
  └───────────────┘                               │  PostProcessor   │
                                                  │                  │
                                                  │ • Date ISO norm  │
                                                  │ • Currency parse │
                                                  │ • Clause classify│
                                                  │ • Deduplication  │
                                                  └────────┬─────────┘
                                                           │
                                                  ┌────────▼─────────┐
                                                  │  JSON Response   │
                                                  │  (Pydantic)      │
                                                  └──────────────────┘
```

---

## 📁 Folder Structure

```
legal-contract-extractor/
│
├── app/                          # FastAPI application
│   ├── main.py                   # App entrypoint, middleware, lifespan
│   ├── routes/
│   │   ├── extraction.py         # POST /api/v1/extract endpoint
│   │   └── health.py             # GET /health endpoint
│   ├── services/
│   │   ├── extraction_service.py # Pipeline orchestrator
│   │   ├── ner_service.py        # Singleton NER model wrapper (HF/spaCy/rules)
│   │   ├── text_cleaner.py       # OCR noise removal & text normalisation
│   │   └── postprocessor.py      # Validation, normalisation, classification
│   ├── schemas/
│   │   └── extraction.py         # Pydantic request/response models
│   ├── models/                   # (Reserved) Custom PyTorch/Pydantic models
│   └── utils/
│       ├── logger.py             # Structured logging (text/JSON)
│       └── pdf_utils.py          # Smart PDF text extraction with OCR fallback
│
├── ocr/
│   └── ocr_pipeline.py           # Tesseract OCR pipeline (pdf2image + preprocessing)
│
├── training/
│   ├── train.py                  # Fine-tune BERT NER on Doccano annotations
│   └── evaluate.py               # Evaluate model on held-out test set
│
├── config/
│   ├── config.yaml               # Non-secret configuration
│   └── settings.py               # Pydantic-settings (env var loading)
│
├── data/
│   ├── raw/                      # Input PDFs (not committed to git)
│   ├── processed/                # Cleaned text outputs
│   └── annotations/
│       └── sample_annotations.jsonl   # Doccano JSONL annotation format
│
├── notebooks/                    # Jupyter notebooks (EDA only, not production code)
│
├── tests/
│   ├── test_pipeline.py          # Unit tests: TextCleaner, PostProcessor
│   └── test_api.py               # Integration tests: FastAPI TestClient
│
├── models/                       # Fine-tuned model artifacts (gitignored)
├── Dockerfile                    # Multi-stage production Docker image
├── docker-compose.yml            # API + MLflow services
├── requirements.txt              # Pinned dependencies
├── pytest.ini                    # Test configuration
├── .env.example                  # Environment variable template
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI 0.111 | REST endpoints, async, auto-docs |
| NER (Primary) | HuggingFace Transformers (BERT) | Fine-tuned named entity recognition |
| NER (Fallback 1) | spaCy en_core_web_trf | Pre-trained English NER |
| NER (Fallback 2) | Regex rule engine | Zero-dependency fallback |
| PDF Extraction | pdfplumber | Digital PDF text extraction |
| OCR | Tesseract + pdf2image | Scanned PDF text recognition |
| Image Processing | OpenCV + Pillow | OCR preprocessing (denoise, binarise) |
| Experiment Tracking | MLflow | Training runs, metrics, model registry |
| Containerisation | Docker + Compose | Reproducible deployment |
| Validation | Pydantic v2 | Type-safe request/response schemas |
| Testing | pytest + httpx | Unit + integration tests |
| Logging | python-json-logger | Structured JSON logs for log aggregators |

---

## 🔄 Pipeline Design

```
PDF Input
    │
    ├─► [pdfplumber] ──► Enough text? ──YES──► Raw Text
    │                          │
    │                          NO
    │                          │
    └─► [pdf2image] ──► PIL Images ──► [OpenCV preprocess] ──► [Tesseract OCR] ──► Raw Text
                                         (deskew, denoise,
                                          binarise, threshold)
                                                │
                                                ▼
                                        [TextCleaner]
                                    (ligatures, line breaks,
                                     OCR artefacts, unicode)
                                                │
                                                ▼
                                        [NERService]
                                    HuggingFace BERT NER
                                    → chunked inference
                                    → entity routing
                                                │
                                                ▼
                                        [PostProcessor]
                                    Dates  → ISO-8601 normalise
                                    Money  → amount + currency parse
                                    Parties→ name clean + role infer
                                    Terms  → clause classify + notice
                                                │
                                                ▼
                                        Structured JSON
```

---

## 📦 Extracted Entities

### Dates
- Normalised to ISO-8601 (`YYYY-MM-DD`)
- Labelled: `EFFECTIVE_DATE`, `EXPIRY_DATE`, `TERMINATION_DATE`, `SIGNING_DATE`, `CONTRACT_DATE`

### Parties
- Cleaned organisation/person names
- Roles inferred from context: `VENDOR`, `CLIENT`, `EMPLOYER`, `EMPLOYEE`, `CONTRACTOR`, `LESSOR`, `LESSEE`

### Monetary Values
- Parsed to `amount` (float) + `currency` (ISO 4217)
- Supports symbols ($, €, £, ₹) and codes (USD, EUR, GBP, INR…)
- Handles multipliers (million, billion, thousand)
- Context snippet attached

### Termination Clauses
- Full clause text preserved
- Classified: `TERMINATION_FOR_CAUSE`, `TERMINATION_FOR_CONVENIENCE`, `TERMINATION_BY_MUTUAL_AGREEMENT`, `AUTOMATIC_TERMINATION`, `INSOLVENCY_TERMINATION`, `TERMINATION_GENERAL`
- Notice period extracted (e.g., "30 days")

---

## 💻 Setup — Local Development

### Prerequisites

- Python 3.11+
- Tesseract OCR: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Ubuntu)
- Poppler: `brew install poppler` or `apt-get install poppler-utils`

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/legal-contract-extractor.git
cd legal-contract-extractor

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy fallback model
python -m spacy download en_core_web_sm

# 5. Configure environment
cp .env.example .env
# Edit .env — set MODEL_PATH if you have a fine-tuned model

# 6. Run the API
uvicorn app.main:app --reload --port 8000

# 7. Open interactive docs
open http://localhost:8000/docs
```

---

## 🐳 Setup — Docker

```bash
# 1. Configure environment
cp .env.example .env

# 2. Build and start all services (API + MLflow)
docker-compose up --build

# API available at:  http://localhost:8000
# MLflow UI at:      http://localhost:5000
# API docs at:       http://localhost:8000/docs

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down
```

---

## 📡 API Reference

### `GET /health`

Health check — confirms service is running.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "service": "legal-contract-extractor",
  "version": "1.0.0",
  "python": "3.11.8",
  "platform": "Linux"
}
```

---

### `POST /api/v1/extract`

Extract entities from a legal PDF.

**Request**

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@/path/to/contract.pdf"
```

**Python Example**

```python
import httpx

with open("contract.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/extract",
        files={"file": ("contract.pdf", f, "application/pdf")},
    )

print(response.json())
```

**Constraints**
- Accepts: `application/pdf`
- Max file size: 50 MB
- Supports: digital PDFs and scanned PDFs (OCR applied automatically)

---

## 📄 Sample JSON Output

```json
{
  "success": true,
  "document_id": "a3f9c12b-4e77-4a1a-bf3c-1234567890ab",
  "metadata": {
    "filename": "service_agreement.pdf",
    "file_size_bytes": 204800,
    "ocr_applied": false,
    "pages_processed": 12,
    "processing_time_ms": 1423.5,
    "model_version": "spacy-en_core_web_sm",
    "extracted_at": "2024-06-01T10:30:00Z"
  },
  "entities": {
    "dates": [
      {
        "value": "2024-01-15",
        "raw_text": "January 15, 2024",
        "label": "EFFECTIVE_DATE",
        "confidence": 0.92
      },
      {
        "value": "2026-01-14",
        "raw_text": "January 14, 2026",
        "label": "EXPIRY_DATE",
        "confidence": 0.88
      }
    ],
    "parties": [
      {
        "name": "Acme Corporation, Inc.",
        "role": "VENDOR",
        "raw_text": "Acme Corporation, Inc. (the \"Vendor\")",
        "confidence": 0.91
      },
      {
        "name": "Beta Enterprises LLC",
        "role": "CLIENT",
        "raw_text": "Beta Enterprises LLC (the \"Client\")",
        "confidence": 0.89
      }
    ],
    "monetary_values": [
      {
        "amount": 150000.0,
        "currency": "USD",
        "raw_text": "USD 150,000",
        "context": "the total annual service fee shall be USD 150,000, payable quarterly",
        "confidence": 0.95
      }
    ],
    "termination_clauses": [
      {
        "text": "Either party may terminate this Agreement for convenience upon 30 days written notice to the other party.",
        "clause_type": "TERMINATION_FOR_CONVENIENCE",
        "notice_period": "30 days",
        "confidence": 0.87
      },
      {
        "text": "Either party may terminate this Agreement for cause in the event of a material breach that remains uncured for 15 days.",
        "clause_type": "TERMINATION_FOR_CAUSE",
        "notice_period": "15 days",
        "confidence": 0.85
      }
    ]
  }
}
```

---

## 🧠 Training a Custom NER Model

### Step 1 — Annotate Data with Doccano

[Doccano](https://github.com/doccano/doccano) is the recommended open-source annotation tool.

```bash
# Install and run Doccano locally
pip install doccano
doccano
# Open http://localhost:8000 → create project → import contract text → annotate
# Export as JSONL → save to data/annotations/train.jsonl + val.jsonl + test.jsonl
```

**Annotation Labels to Configure in Doccano:**
`DATE`, `PARTY`, `MONEY`, `TERMINATION`

### Step 2 — Fine-tune BERT

```bash
python training/train.py \
  --data_dir data/annotations \
  --output_dir models/legal-ner-bert-v1 \
  --base_model dslim/bert-base-NER \
  --epochs 5 \
  --batch_size 16 \
  --mlflow_uri http://localhost:5000
```

### Step 3 — Evaluate

```bash
python training/evaluate.py \
  --model_path models/legal-ner-bert-v1/final \
  --test_file data/annotations/test.jsonl \
  --output_dir reports/
```

### Step 4 — Wire Up the Fine-tuned Model

```bash
# In .env
MODEL_PATH=./models/legal-ner-bert-v1/final
```

The API will automatically load the fine-tuned model on next startup.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only pipeline unit tests
pytest tests/test_pipeline.py -v

# Run only API integration tests
pytest tests/test_api.py -v

# Run with coverage report
pip install pytest-cov
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

---

## 📊 MLflow Experiment Tracking

When `MLFLOW_TRACKING_URI` is set, training runs are automatically logged:

- Hyperparameters (learning rate, batch size, epochs)
- Metrics per epoch (precision, recall, F1, accuracy)
- Best model checkpoint artifact
- Training loss curves

Access the MLflow UI:

```
http://localhost:5000
```

---

## 🗺️ Roadmap

- [ ] Async NER inference with background tasks for large batches
- [ ] Clause-level confidence calibration
- [ ] Support for DOCX input format
- [ ] Multi-language contracts (multilingual BERT)
- [ ] Redis-based result caching for repeated documents
- [ ] Webhook callbacks for async processing
- [ ] Fine-grained role detection using dependency parsing
- [ ] Kubernetes Helm chart for production deployment
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] Admin dashboard (Streamlit or Next.js)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/add-clause-classifier`
3. Commit your changes: `git commit -m 'feat: add clause type classifier'`
4. Push to the branch: `git push origin feature/add-clause-classifier`
5. Open a Pull Request

Please ensure all tests pass before submitting a PR.
