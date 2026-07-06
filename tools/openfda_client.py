"""
tools/openfda_client.py

Wraps the openFDA Drug Label API (https://api.fda.gov/drug/label.json).

WHY THIS EXISTS:
The NIH's RxNav Drug-Drug Interaction API -- the most widely used free
source for precomputed drug interaction data -- was discontinued in
January 2024. What remains is raw FDA label text per drug. This module
fetches that text live so the Conflict & Risk Agent can reason over it.

DESIGN DECISIONS:
- No caching: labels are fetched live on every run to ensure freshness.
  FDA labels update weekly; a cache would risk stale safety data.
- Graceful degradation: if a label isn't found, we return a structured
  DrugLabel with found=False rather than raising an exception. The agent
  can then explicitly note the gap rather than silently missing it.
- Multiple search strategies: openFDA indexes drugs inconsistently.
  We try four combinations per drug to maximize hit rate.
- Rate limiting: 0.3s between requests to be a well-behaved consumer
  of a free public API.
- Retry logic: transient failures are retried with exponential backoff.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
REQUEST_TIMEOUT = 20
RATE_LIMIT_SECONDS = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


@dataclass
class DrugLabel:
    """
    Normalized, agent-friendly view of one openFDA label record.

    Agents receive this as structured data rather than raw API JSON.
    to_context_block() renders it as plain text the LLM can reason over.
    """
    query: str
    found: bool
    brand_name: str | None = None
    generic_name: str | None = None
    rxcui: list[str] = field(default_factory=list)
    drug_interactions: str | None = None
    warnings: str | None = None
    contraindications: str | None = None
    boxed_warning: str | None = None
    error: str | None = None

    def to_context_block(self) -> str:
        """
        Render label data as plain text the LLM agent reasons over.

        Boxed warnings surface first -- they represent the FDA's most
        serious safety designation and must be treated as highest priority.
        Truncation limits prevent context window overflow while preserving
        the most clinically relevant opening sections.
        """
        if not self.found:
            return f"## {self.query}\nNo FDA label found. {self.error or ''}".strip()

        name = self.brand_name or self.generic_name or self.query
        parts = [f"## {name} (queried as '{self.query}')"]

        # Boxed warnings first -- highest severity
        if self.boxed_warning:
            parts.append(f"BOXED WARNING: {self.boxed_warning[:1200]}")

        # Drug interactions -- primary source for cross-drug reasoning
        if self.drug_interactions:
            parts.append(f"Drug Interactions: {self.drug_interactions[:1200]}")

        # Contraindications -- absolute restrictions
        if self.contraindications:
            parts.append(f"Contraindications: {self.contraindications[:800]}")

        # General warnings
        if self.warnings:
            parts.append(f"Warnings: {self.warnings[:800]}")

        if len(parts) == 1:
            parts.append("(Label found but no structured safety sections present.)")

        return "\n\n".join(parts)


def _first(values, default=None):
    """Return first element of a list, or default if empty/None.
    openFDA returns all fields as lists even when there's only one value."""
    if isinstance(values, list) and values:
        return values[0]
    return default


def _join(values, default=None):
    """Join a list of strings into one paragraph.
    openFDA sometimes splits a single label section across multiple list items."""
    if isinstance(values, list) and values:
        return " ".join(values)
    return default


def _query_once(field: str, drug_name: str, quoted: bool) -> dict | None:
    """
    Make one HTTP GET request to the openFDA label endpoint.

    quoted=True uses phrase search (exact match).
    quoted=False uses term search (broader, catches salt forms).
    Returns first result dict or None if 404/no results.
    """
    term = f'"{drug_name}"' if quoted else drug_name
    params = {"search": f"{field}:{term}", "limit": 1}

    logger.debug("openFDA query: field=%s drug=%s quoted=%s", field, drug_name, quoted)

    resp = requests.get(FDA_LABEL_URL, params=params, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 404:
        return None

    resp.raise_for_status()
    results = resp.json().get("results", [])
    return results[0] if results else None


def lookup_drug_label(drug_name: str) -> DrugLabel:
    """
    Look up a single drug's FDA label by name with retry logic.

    SEARCH ORDER (stops at first hit):
    1. Generic name, exact phrase
    2. Generic name, term search (catches salt forms)
    3. Brand name, exact phrase
    4. Brand name, term search

    Transient failures are retried with exponential backoff.
    Returns DrugLabel with found=False on all failures rather than raising.
    """
    drug_name = drug_name.strip()

    if not drug_name:
        return DrugLabel(query=drug_name, found=False, error="Empty drug name.")

    for field_name in ("openfda.generic_name", "openfda.brand_name"):
        for quoted in (True, False):
            for attempt in range(MAX_RETRIES):
                try:
                    record = _query_once(field_name, drug_name, quoted)

                    if record:
                        openfda = record.get("openfda", {})
                        logger.info(
                            "FDA label found for '%s' via %s (quoted=%s)",
                            drug_name, field_name, quoted
                        )
                        return DrugLabel(
                            query=drug_name,
                            found=True,
                            brand_name=_first(openfda.get("brand_name")),
                            generic_name=_first(openfda.get("generic_name")),
                            rxcui=openfda.get("rxcui", []),
                            drug_interactions=_join(record.get("drug_interactions")),
                            warnings=_join(record.get("warnings")),
                            contraindications=_join(record.get("contraindications")),
                            boxed_warning=_join(record.get("boxed_warning")),
                        )
                    else:
                        break  # No match for this strategy -- try next

                except requests.exceptions.Timeout:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning(
                        "openFDA timeout for '%s' (attempt %d/%d), retrying in %.1fs",
                        drug_name, attempt + 1, MAX_RETRIES, wait
                    )
                    time.sleep(wait)

                except requests.exceptions.RequestException as e:
                    wait = RETRY_BACKOFF ** attempt
                    logger.warning(
                        "openFDA error for '%s': %s (attempt %d/%d)",
                        drug_name, e, attempt + 1, MAX_RETRIES
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(wait)
                    else:
                        logger.error("All retries exhausted for '%s': %s", drug_name, e)
                        return DrugLabel(
                            query=drug_name,
                            found=False,
                            error=f"openFDA request failed after {MAX_RETRIES} attempts: {e}"
                        )

            time.sleep(RATE_LIMIT_SECONDS)

    logger.warning("No FDA label found for '%s' after all search strategies", drug_name)
    return DrugLabel(query=drug_name, found=False, error="No matching FDA label found.")


def lookup_multiple(drug_names: list[str]) -> list[DrugLabel]:
    """
    Look up FDA labels for a full medication list.
    Results preserve input order for agent correlation.
    """
    results = []
    for name in drug_names:
        results.append(lookup_drug_label(name))
        time.sleep(RATE_LIMIT_SECONDS)
    return results


def get_drug_safety_info(drug_names: list[str]) -> dict:
    """
    ADK tool: retrieve FDA label safety information for a medication list.

    This is the function the Conflict & Risk Agent calls as a tool.
    ADK auto-generates the tool schema from type hints and docstring.

    Args:
        drug_names: List of medication names e.g. ["warfarin", "ibuprofen"]

    Returns:
        dict with label text per drug, lookup failures, and fetch timestamp.
    """
    logger.info(
        "get_drug_safety_info called for %d drugs: %s", len(drug_names), drug_names
    )

    labels = lookup_multiple(drug_names)
    failures = [l.query for l in labels if not l.found]

    if failures:
        logger.warning("FDA label lookup failures: %s", failures)

    return {
        "labels": [l.to_context_block() for l in labels],
        "lookup_failures": failures,
        "source": "FDA openFDA Drug Label API (api.fda.gov) -- official label text, updated weekly",
        "fetched_live_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "freshness_note": (
            "Labels fetched live from api.fda.gov at the time of this run -- "
            "not cached. Reflects the current published FDA label."
        ),
    }


if __name__ == "__main__":
    # Quick smoke test -- run directly to verify live API connectivity
    import json
    logging.basicConfig(level=logging.INFO)
    result = get_drug_safety_info(["warfarin", "ibuprofen"])
    print(json.dumps(result, indent=2)[:3000])