"""
services/logging_utils.py

Shared logging configuration and helpers for MedGuard entrypoints.

WHY THIS EXISTS:
Both the CLI (demo.py) and the API (api_server.py) need consistent
logging behavior. Centralizing it here means one change affects both.

JSON LOGGING:
In production, logs are rendered as JSON (one object per line) for
ingestion by SIEM tools like Splunk or Google Cloud Logging. In
development, set LOG_FORMAT=text for human-readable output.

PII SAFETY:
Patient names are never logged directly. hash_patient_identifier()
returns a non-reversible 16-char hash used in place of the real name
in all structured log fields. This means logs are safe to ship to
external monitoring systems without exposing patient data.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os


class JsonFormatter(logging.Formatter):
    """
    Render logs as one JSON object per line for SIEM ingestion.

    Structured fields (request_id, patient_id_hash, endpoint, etc.)
    are extracted from LogRecord extras and included in the JSON payload.
    This makes logs filterable and queryable in log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structured fields added via logger.info("msg", extra={...})
        structured_fields = (
            "request_id",
            "patient_id_hash",
            "endpoint",
            "latency_ms",
            "status",
            "method",
            "path",
        )
        for field_name in structured_fields:
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True)


def hash_patient_identifier(patient_name: str) -> str:
    """
    Return a non-reversible hash for patient identifier logging.

    Uses SHA-256 truncated to 16 hex chars. This is enough to correlate
    log entries for the same patient without exposing the real name.
    Never used for security -- only for PII-safe logging.
    """
    normalized = patient_name.strip().lower().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def configure_logging() -> None:
    """
    Configure root logging once for both CLI and API entrypoints.

    Environment variables:
    - LOG_LEVEL: DEBUG | INFO | WARNING (default: WARNING)
    - LOG_FORMAT: json | text (default: json)

    Call this once at application startup before any log statements.
    """
    log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
    log_format = os.environ.get("LOG_FORMAT", "json").lower()

    handler = logging.StreamHandler()
    if log_format == "text":
        # Human-readable format for local development
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    else:
        # JSON format for production log aggregation
        handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level, logging.WARNING))