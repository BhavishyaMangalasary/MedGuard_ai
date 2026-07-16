"""
api_server.py

HTTP API entrypoint for MedGuard.

Exposes the 4-agent pipeline as a REST API with:
- Versioned endpoints (/v1/analyze)
- Request ID tracing across all logs
- Structured JSON logging
- Hashed patient identifiers (PII-safe logs)
- Proper 429 handling with Retry-After header
- Health check endpoint for Cloud Run / load balancers

HOW TO RUN:
    uvicorn api_server:app --reload --port 8080

ENDPOINTS:
    GET  /          -- welcome and API info
    GET  /healthz   -- liveness probe for Cloud Run / load balancers
    POST /v1/analyze -- analyze a medication list, returns daily brief
    GET  /docs      -- auto-generated Swagger UI
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import datetime
import logging
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.logging_utils import configure_logging, hash_patient_identifier
from services.pipeline_service import (
    enforce_disclaimer,
    get_endpoint,
    get_patient_id_hash,
    get_request_id,
    run_pipeline,
    set_endpoint,
    set_patient_id_hash,
    set_request_id,
    validate_input,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedGuard API",
    version="1.0.0",
    description=(
        "AI-powered medication safety assistant. Analyzes a patient's full "
        "medication list using live FDA label data and multi-agent reasoning "
        "to flag drug interactions, build a daily schedule, and alert on "
        "upcoming refill gaps."
    ),
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    patient_name: str = Field(
        min_length=2,
        max_length=100,
        description="Name of the patient (e.g. 'Janet', 'my mom')",
        examples=["Janet"]
    )
    medication_text: str = Field(
        min_length=10,
        max_length=10000,
        description=(
            "Full medication list in plain text. Include drug name, dose, "
            "frequency, prescribing doctor, days supply, and last fill date."
        ),
        examples=[
            "From Dr. Patel: Warfarin 5mg once daily in the evening. "
            "From Dr. Nguyen: Ibuprofen 400mg as needed for joint pain."
        ]
    )


class AnalyzeResponse(BaseModel):
    patient_name: str
    brief: str
    generated_at: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a consistent JSON error response with request ID header."""
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=request_id)
    ).model_dump()
    headers = {"x-request-id": request_id}
    if extra_headers:
        headers.update(extra_headers)
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def _is_quota_exhausted_error(exc: Exception) -> bool:
    """Detect Gemini quota exhaustion errors from upstream exception text."""
    text = str(exc).lower()
    markers = ("resource_exhausted", "quota exceeded", "error code 429", "429")
    return any(marker in text for marker in markers)


def _extract_retry_after_seconds(exc: Exception) -> str | None:
    """
    Parse retry delay seconds from upstream quota error text.
    Returns as string for the Retry-After header, or None if not found.
    """
    text = str(exc)
    # e.g. "Please retry in 28.629168138s."
    retry_in_match = re.search(
        r"retry\s+in\s+([0-9]+(?:\.[0-9]+)?)s", text, re.IGNORECASE
    )
    if retry_in_match:
        return str(max(1, round(float(retry_in_match.group(1)))))
    # e.g. "'retryDelay': '28s'"
    retry_delay_match = re.search(
        r"retryDelay'?:\s*'([0-9]+)s'", text, re.IGNORECASE
    )
    if retry_delay_match:
        return retry_delay_match.group(1)
    return None


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    request_id = get_request_id()
    code_map = {400: "bad_request", 429: "quota_exhausted", 502: "upstream_error"}
    code = code_map.get(exc.status_code, "upstream_error")
    extra_headers = dict(exc.headers or {})
    return _error_response(
        status_code=exc.status_code,
        code=code,
        message=str(exc.detail),
        request_id=request_id,
        extra_headers=extra_headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    request_id = get_request_id()
    logger.warning(
        "request_schema_validation_failed",
        extra={"request_id": request_id, "endpoint": get_endpoint(), "status": 422},
    )
    message = f"Request schema validation failed: {exc.errors()[0]['msg']}"
    return _error_response(
        status_code=422, code="schema_validation_error",
        message=message, request_id=request_id,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    request_id = get_request_id()
    logger.exception(
        "unhandled_exception",
        extra={
            "request_id": request_id,
            "endpoint": get_endpoint(),
            "patient_id_hash": get_patient_id_hash(),
            "status": 500,
        },
    )
    return _error_response(
        status_code=500, code="internal_error",
        message="An internal server error occurred.",
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    Attach request ID and context variables for traceable structured logs.

    Every request gets a unique ID (from header or auto-generated).
    This ID flows through all log statements for the duration of the request,
    making it easy to trace a single request across multiple log lines.
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    endpoint = request.url.path
    set_request_id(request_id)
    set_endpoint(endpoint)
    started_at = time.perf_counter()

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": endpoint,
            "endpoint": endpoint,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": endpoint,
                "endpoint": endpoint,
                "latency_ms": round(elapsed_ms, 2),
                "status": 500,
            },
        )
        raise

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    response.headers["x-request-id"] = request_id
    logger.info(
        "request_finished",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": endpoint,
            "endpoint": endpoint,
            "latency_ms": round(elapsed_ms, 2),
            "status": response.status_code,
        },
    )
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> dict:
    """Welcome endpoint -- confirms the API is running."""
    return {
        "name": "MedGuard API",
        "version": "1.0.0",
        "description": (
            "AI-powered medication safety assistant using live FDA data "
            "and multi-agent reasoning."
        ),
        "endpoints": {
            "health": "GET /healthz",
            "analyze": "POST /v1/analyze",
            "docs": "GET /docs",
        },
    }


@app.get("/healthz")
async def healthz() -> dict:
    """
    Liveness probe for container and platform health checks.
    Used by Cloud Run, load balancers, and monitoring systems.
    Returns 200 OK when the service is running.
    """
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a medication list and return a plain-language daily safety brief.

    Runs the full 4-agent MedGuard pipeline:
    1. Intake Agent -- normalizes raw medication list
    2. Conflict & Risk Agent -- checks live FDA labels, flags interactions,
       uses Antigravity as fallback for drugs with no FDA label
    3. Scheduler & Gap Agent -- builds daily schedule, flags refill gaps
    4. Reporter Agent -- synthesizes into a plain-language daily brief

    Returns 429 with Retry-After header if Gemini quota is exhausted.
    Returns 502 if the pipeline produced no output.
    """
    # Hash patient name for PII-safe logging
    set_patient_id_hash(hash_patient_identifier(payload.patient_name))

    # Validate before hitting the expensive pipeline
    validation_error = validate_input(payload.patient_name, payload.medication_text)
    if validation_error:
        logger.warning(
            "validation_failed",
            extra={
                "request_id": get_request_id(),
                "endpoint": get_endpoint(),
                "patient_id_hash": get_patient_id_hash(),
                "status": 400,
            },
        )
        raise HTTPException(status_code=400, detail=validation_error)

    logger.info(
        "pipeline_invoked",
        extra={
            "request_id": get_request_id(),
            "endpoint": get_endpoint(),
            "patient_id_hash": get_patient_id_hash(),
        },
    )

    try:
        reporter_output = await run_pipeline(
            payload.patient_name, payload.medication_text
        )
    except Exception as exc:
        # Handle Gemini quota exhaustion with a clean 429 + Retry-After
        if _is_quota_exhausted_error(exc):
            retry_after = _extract_retry_after_seconds(exc)
            logger.warning(
                "quota_exhausted",
                extra={
                    "request_id": get_request_id(),
                    "endpoint": get_endpoint(),
                    "patient_id_hash": get_patient_id_hash(),
                    "status": 429,
                },
            )
            message = (
                "Model quota exhausted. Please retry shortly or "
                "use a higher quota API key."
            )
            headers = {"Retry-After": retry_after} if retry_after else None
            raise HTTPException(
                status_code=429, detail=message, headers=headers
            ) from exc
        raise

    if not reporter_output:
        logger.error(
            "pipeline_no_output",
            extra={
                "request_id": get_request_id(),
                "endpoint": get_endpoint(),
                "patient_id_hash": get_patient_id_hash(),
                "status": 502,
            },
        )
        raise HTTPException(
            status_code=502,
            detail="Pipeline produced no output. Verify API keys and model availability.",
        )

    return AnalyzeResponse(
        patient_name=payload.patient_name,
        brief=enforce_disclaimer(reporter_output),
        generated_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )