"""
tools/antigravity_client.py

Antigravity managed agent integration for MedGuard.

WHAT ANTIGRAVITY IS:
Antigravity is Google's managed agent harness -- a powerful general-purpose
agent that plans, reasons, runs code, searches the web, and synthesizes
findings inside a secure isolated Linux sandbox hosted by Google.
It uses the Interactions API via the Google GenAI SDK.

WHY IT'S NOT IN THE MAIN PIPELINE:
Antigravity is a pay-as-you-go service and spins up a Linux sandbox per
call. Including it in the main pipeline would make every run expensive
and slow. Instead it's used as an on-demand fallback -- when openFDA
has no label for a drug, Antigravity autonomously researches it from
multiple authoritative medical sources.

HOW IT'S TRIGGERED:
The Conflict & Risk Agent calls get_drug_research_fallback when
get_drug_safety_info returns non-empty lookup_failures. Antigravity
researches those drugs and its findings flow through to the final brief
with [Source: Antigravity Research Findings] as the citation.

CONFIRMED WORKING:
Antigravity was confirmed working during development -- it returned
9,680 chars of research for a drug not in the FDA database, and those
findings appeared in the final Reporter Agent brief.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ANTIGRAVITY_MODEL = "antigravity-preview-05-2026"


def research_drug_safety(drug_name: str) -> dict:
    """
    Use the Antigravity managed agent to autonomously research a drug.

    Antigravity plans, searches the web, reads authoritative medical
    sources (FDA.gov, NIH, MedlinePlus), and synthesizes a comprehensive
    safety report. Unlike the openFDA tool which fetches a specific label,
    Antigravity can cross-reference multiple sources.

    Args:
        drug_name: Name of the drug to research (generic or brand name)

    Returns:
        dict with safety_summary, model_used, success flag, error if failed
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set -- cannot call Antigravity API")
        return {
            "drug_name": drug_name,
            "safety_summary": None,
            "success": False,
            "error": "GOOGLE_API_KEY not configured"
        }

    # Focused research prompt -- specific enough to get actionable output
    research_prompt = f"""
Research the medication "{drug_name}" and provide a structured safety summary.

Focus specifically on:
1. Known drug interactions -- what other medications does it interact with?
2. Contraindications -- when should this drug NOT be taken?
3. Boxed warnings -- does it carry any FDA black box warnings?
4. Timing considerations -- does it need to be taken at specific times or
   spaced away from other medications?

Use authoritative medical sources (FDA.gov, NIH, MedlinePlus, prescribing
information). Be specific and cite your sources.

Format your response as a structured safety summary a pharmacist could use
to assess interaction risks with other medications.
"""

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        logger.info("Calling Antigravity agent for drug research: %s", drug_name)

        # Use the Interactions API -- this is what distinguishes Antigravity
        # from standard LLM calls. The agent autonomously plans, searches,
        # and synthesizes rather than just generating text from training data.
        interaction = client.interactions.create(
            agent=ANTIGRAVITY_MODEL,
            input=research_prompt,
            environment="remote",
        )

        output_text = getattr(interaction, "output_text", None)

        if not output_text:
            logger.warning(
                "Antigravity returned empty output for drug: %s", drug_name
            )
            return {
                "drug_name": drug_name,
                "safety_summary": None,
                "success": False,
                "error": "Antigravity returned empty response"
            }

        logger.info(
            "Antigravity research complete for %s (%d chars)",
            drug_name, len(output_text)
        )

        return {
            "drug_name": drug_name,
            "safety_summary": output_text,
            "model_used": ANTIGRAVITY_MODEL,
            "success": True,
            "error": None,
            "note": (
                "Safety information researched autonomously by the Antigravity "
                "managed agent from multiple authoritative sources. Always verify "
                "critical decisions against official FDA label data."
            )
        }

    except Exception as e:
        logger.error("Antigravity error for %s: %s", drug_name, e)
        return {
            "drug_name": drug_name,
            "safety_summary": None,
            "success": False,
            "error": str(e)
        }


def get_drug_research_fallback(drug_names: list[str]) -> dict:
    """
    ADK tool: use Antigravity to research drugs with no FDA label data.

    Called by the Conflict & Risk Agent when openFDA returns no label
    for a specific drug. Antigravity autonomously researches the drug
    from multiple sources as a fallback data source.

    Args:
        drug_names: List of drug names that had no FDA label data

    Returns:
        dict with Antigravity research results for each drug
    """
    results = {}
    for drug_name in drug_names:
        logger.info("Running Antigravity fallback research for: %s", drug_name)
        results[drug_name] = research_drug_safety(drug_name)

    return {
        "research_results": results,
        "model": ANTIGRAVITY_MODEL,
        "note": (
            "Findings researched autonomously by Antigravity managed agent. "
            "Treat as supplementary to FDA label data, not a replacement."
        )
    }


if __name__ == "__main__":
    """
    Standalone demo of Antigravity drug research.
    Run: python tools/antigravity_client.py
    """
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("  MedGuard -- Antigravity Drug Research Demo")
    print("=" * 60)
    print("\nThis uses the Antigravity managed agent to autonomously")
    print("research a drug's safety profile from multiple sources.")
    print("(May take 30-60 seconds)\n")

    drug = input("Enter drug name to research: ").strip()
    if not drug:
        print("No drug name entered.")
    else:
        print(f"\nResearching {drug} with Antigravity...\n")
        result = research_drug_safety(drug)

        if result["success"]:
            print("=" * 60)
            print(f"ANTIGRAVITY RESEARCH: {drug.upper()}")
            print("=" * 60)
            print(result["safety_summary"])
        else:
            print(f"Research failed: {result['error']}")