"""
tools/safety_rules.py

Rule-based confidence calibration layer for MedGuard.

WHY THIS EXISTS:
The Conflict & Risk Agent assigns severity levels (critical/moderate/
informational) based on LLM reasoning over FDA label text. This is
generally accurate but has two failure modes:
1. The LLM may assign "moderate" to a Boxed Warning interaction when
   it should always be "critical"
2. The LLM may surface a flag with no citation -- meaning it generated
   a safety claim not grounded in the actual FDA label text

This module provides a deterministic post-processing layer that:
- Enforces severity escalation rules (certain keywords always = critical)
- Strips uncited flags before they reach the Reporter Agent
- Adds a confidence score to each flag based on evidence strength

DESIGN PRINCIPLE:
Rules are applied AFTER the LLM reasoning step, not instead of it.
The LLM catches the interactions -- these rules ensure they're
classified consistently regardless of how the LLM phrases its output.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Keywords that always escalate severity to CRITICAL.
# These map to FDA Boxed Warning language and explicit contraindications.
CRITICAL_ESCALATION_KEYWORDS = [
    "boxed warning",
    "black box warning",
    "fatal",
    "life-threatening",
    "contraindicated",
    "contraindication",
    "major or fatal bleeding",
    "lactic acidosis",
    "severe or fatal",
    "death",
    "can cause death",
    "risk of death",
]

# Keywords that indicate at least MODERATE severity
MODERATE_KEYWORDS = [
    "drug interactions",
    "interactions section",
    "warnings section",
    "may result in",
    "can lead to",
    "increases the risk",
    "increased risk",
    "monitor",
    "caution",
]

# Pattern for a valid FDA label citation
# Format: [Source: Drug name FDA label -- Section name]
CITATION_PATTERN = re.compile(
    r'\[Source:\s*.+?FDA label\s*--\s*.+?\]',
    re.IGNORECASE
)


def has_valid_citation(flag_text: str) -> bool:
    """
    Check if a flag contains a valid FDA label citation.

    This is the hallucination guard -- a flag without a citation cannot
    be traced to actual FDA label text and should not reach the user.
    """
    return bool(CITATION_PATTERN.search(flag_text))


def calibrate_severity(flag_text: str, assigned_severity: str) -> str:
    """
    Apply deterministic severity calibration rules to a flag.

    ESCALATION RULES (in priority order):
    1. Any Boxed Warning language → always CRITICAL
    2. Any explicit contraindication language → always CRITICAL
    3. Fatal/life-threatening language → always CRITICAL
    4. Drug interactions section mention → at least MODERATE

    Args:
        flag_text: Full text of the flag including explanation and citation
        assigned_severity: Severity level assigned by the LLM

    Returns:
        Calibrated severity string: "critical", "moderate", or "informational"
    """
    flag_lower = flag_text.lower()

    # Rule 1: Critical escalation -- always override the LLM's assignment
    for keyword in CRITICAL_ESCALATION_KEYWORDS:
        if keyword in flag_lower:
            if assigned_severity != "critical":
                logger.info(
                    "Severity escalated to critical: keyword='%s' original='%s'",
                    keyword, assigned_severity
                )
            return "critical"

    # Rule 2: Moderate floor -- these should never be informational
    for keyword in MODERATE_KEYWORDS:
        if keyword in flag_lower:
            if assigned_severity == "informational":
                logger.info(
                    "Severity escalated to moderate: keyword='%s' original='%s'",
                    keyword, assigned_severity
                )
                return "moderate"

    return assigned_severity


def calculate_confidence_score(flag_text: str, severity: str) -> float:
    """
    Calculate a confidence score (0.0 to 1.0) for a safety flag.

    Higher scores indicate stronger evidence grounding. Used to prioritize
    flags -- higher confidence flags surface first in Reporter output.

    SCORING FACTORS:
    - Has valid citation: +0.4 (most important -- grounded in FDA text)
    - Severity level: critical=+0.3, moderate=+0.2, informational=+0.1
    - Contains section name in citation: +0.2
    - Drug name specificity in citation: +0.1
    """
    score = 0.0

    # Citation presence is the strongest signal
    if has_valid_citation(flag_text):
        score += 0.4

    # Severity contribution
    severity_scores = {"critical": 0.3, "moderate": 0.2, "informational": 0.1}
    score += severity_scores.get(severity.lower(), 0.1)

    # Specific section reference adds confidence
    section_keywords = [
        "boxed warning", "drug interactions", "contraindications", "warnings"
    ]
    if any(kw in flag_text.lower() for kw in section_keywords):
        score += 0.2

    # Drug name specificity in citation
    if re.search(r'\[Source:\s*\w+', flag_text, re.IGNORECASE):
        score += 0.1

    return min(score, 1.0)


def process_flags(flags: list[dict]) -> list[dict]:
    """
    Apply the full safety rules pipeline to a list of flags.

    PIPELINE:
    1. Strip flags with no valid citation (hallucination guard)
    2. Calibrate severity using deterministic rules
    3. Add confidence score to each flag
    4. Sort by confidence score (highest first)

    Args:
        flags: List of flag dicts with keys: text, severity, drugs_involved

    Returns:
        Processed list with calibrated severity, confidence scores,
        sorted by confidence descending. Uncited flags removed.
    """
    processed = []
    stripped_count = 0

    for flag in flags:
        text = flag.get("text", "")
        severity = flag.get("severity", "informational")

        # HALLUCINATION GUARD: strip flags with no valid citation
        if not has_valid_citation(text):
            logger.warning(
                "Flag stripped -- no valid citation found: '%s'",
                text[:100]
            )
            stripped_count += 1
            continue

        # Apply severity calibration
        calibrated_severity = calibrate_severity(text, severity)

        # Calculate confidence score
        confidence = calculate_confidence_score(text, calibrated_severity)

        processed.append({
            **flag,
            "severity": calibrated_severity,
            "confidence_score": confidence,
        })

    if stripped_count > 0:
        logger.warning(
            "Hallucination guard: stripped %d uncited flags", stripped_count
        )

    # Sort by confidence score descending
    processed.sort(key=lambda f: f.get("confidence_score", 0), reverse=True)

    return processed


def get_severity_emoji(severity: str) -> str:
    """
    Return the confidence level emoji for a given severity.
    Centralized here so it's consistent across the codebase.
    """
    mapping = {
        "critical": "⚠️ CRITICAL",
        "moderate": "🔶 MODERATE",
        "informational": "ℹ️ INFORMATIONAL",
    }
    return mapping.get(severity.lower(), "ℹ️ INFORMATIONAL")