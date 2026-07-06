"""
services/pipeline_service.py

Shared runtime logic for MedGuard pipeline execution.

WHY THIS EXISTS:
Both the CLI (demo.py) and the API (api_server.py) need to run the same
4-agent pipeline. Rather than duplicating the pipeline execution logic
in both files, it lives here and both entrypoints import it.

This ensures:
- Identical behavior between CLI and API modes
- One place to update if the pipeline changes
- Cleaner, testable code

CONTEXT VARIABLES:
request_id, endpoint, and patient_id_hash are stored as context variables
(not global variables) so they're safe in async/concurrent environments.
Each request gets its own context, preventing cross-request data leakage.

DISCLAIMER ENFORCEMENT:
enforce_disclaimer() is a code-level safety net -- it appends the
disclaimer if the Reporter Agent omits it. Defense in depth: the
disclaimer appears even if the LLM ignores the instruction.
"""

from __future__ import annotations

import contextvars
import logging

from google.adk.runners import InMemoryRunner
from google.genai import types

from agents.orchestrator import root_agent
from agents.reporter_agent import DISCLAIMER_TEXT

logger = logging.getLogger(__name__)

# Context variables for request tracing -- safe in async environments.
# Each request gets its own context, no cross-request data leakage.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
_endpoint_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "endpoint", default="-"
)
_patient_id_hash_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "patient_id_hash", default="-"
)


def set_request_id(request_id: str) -> None:
    """Store request ID in context for downstream log correlation."""
    _request_id_var.set(request_id)


def get_request_id() -> str:
    """Read context-local request ID."""
    return _request_id_var.get()


def set_endpoint(endpoint: str) -> None:
    """Store endpoint path in context for structured logging."""
    _endpoint_var.set(endpoint)


def get_endpoint() -> str:
    """Read context-local endpoint path."""
    return _endpoint_var.get()


def set_patient_id_hash(patient_id_hash: str) -> None:
    """Store hashed patient ID in context for PII-safe logging."""
    _patient_id_hash_var.set(patient_id_hash)


def get_patient_id_hash() -> str:
    """Read context-local hashed patient ID."""
    return _patient_id_hash_var.get()


def enforce_disclaimer(text: str) -> str:
    """
    Code-level disclaimer enforcement independent of LLM behavior.

    The Reporter Agent is instructed to include the disclaimer on every
    output. This function checks in code and appends it if missing.
    Defense-in-depth: the disclaimer appears even if the model ignores
    the instruction or a prompt change accidentally removes it.
    """
    if DISCLAIMER_TEXT.lower() not in text.lower():
        logger.warning(
            "Reporter Agent omitted disclaimer -- appending in code",
            extra={
                "request_id": get_request_id(),
                "endpoint": get_endpoint(),
                "patient_id_hash": get_patient_id_hash(),
            },
        )
        return text.strip() + f"\n\n{DISCLAIMER_TEXT}"
    return text


def validate_input(patient_name: str, medication_text: str) -> str | None:
    """
    Validate inputs before sending to the pipeline.

    Returns an error message string if validation fails, None if valid.
    Catching bad input here prevents wasting API quota on empty or
    clearly invalid requests.
    """
    if not patient_name or len(patient_name.strip()) < 2:
        return "Patient name must be at least 2 characters."
    if len(patient_name) > 100:
        return "Patient name is too long (max 100 characters)."
    if not medication_text or len(medication_text.strip()) < 10:
        return "Medication list is too short. Please provide at least one medication."
    if len(medication_text) > 10000:
        return "Medication list is too long (max 10,000 characters)."
    return None


async def run_pipeline(patient_name: str, medication_text: str) -> str | None:
    """
    Run the full 4-agent MedGuard pipeline and return Reporter output.

    PIPELINE EXECUTION:
    ADK's InMemoryRunner manages the session and routes each event to
    the appropriate agent in sequence. We filter for events authored by
    reporter_agent since that's the only output the user should see.
    Intermediate agent outputs are internal pipeline state.

    Returns:
        Reporter Agent output text, or None if pipeline produced no output.
    """
    user_id = patient_name.lower().replace(" ", "_")

    logger.info(
        "Starting pipeline",
        extra={
            "request_id": get_request_id(),
            "endpoint": get_endpoint(),
            "patient_id_hash": get_patient_id_hash(),
        },
    )

    runner = InMemoryRunner(agent=root_agent, app_name="medguard")
    session = await runner.session_service.create_session(
        app_name="medguard",
        user_id=user_id,
    )

    # Prepend patient name so all agents refer to the patient correctly
    full_input = f"Patient name: {patient_name}\n\n{medication_text}"
    message = types.Content(role="user", parts=[types.Part(text=full_input)])

    reporter_output = None

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=message,
    ):
        # Only capture Reporter Agent output -- all other agents are internal
        if event.content and event.content.parts and event.author == "reporter_agent":
            for part in event.content.parts:
                if getattr(part, "text", None):
                    reporter_output = part.text
                    logger.info(
                        "Reporter Agent output received",
                        extra={
                            "request_id": get_request_id(),
                            "endpoint": get_endpoint(),
                            "patient_id_hash": get_patient_id_hash(),
                        },
                    )

    return reporter_output