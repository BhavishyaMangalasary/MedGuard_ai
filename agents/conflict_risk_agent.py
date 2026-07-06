"""
agents/conflict_risk_agent.py

Agent 2: Conflict & Risk Agent

ROLE IN PIPELINE:
The core reasoning agent in MedGuard. This is where the actual clinical
safety analysis happens -- everything else in the pipeline exists to
support this agent or present its findings.

WHY THIS AGENT EXISTS (and why a script can't do this):
The NIH's RxNav Drug-Drug Interaction API was discontinued in January 2024.
What remains is raw FDA label text -- paragraphs of clinical language per
drug. Identifying conflicts across a full medication list requires:
1. Reading the Drug Interactions, Warnings, Contraindications, and Boxed
   Warning sections for EVERY drug on the list
2. Reasoning across all of them simultaneously -- not just pairwise --
   to catch multi-drug interactions and indirect risks
3. Distinguishing what's clinically relevant for THIS specific patient
   from generic label boilerplate
A traditional script can fetch the text. Only an LLM agent can do the
cross-document reasoning step.

THREE CATEGORIES OF FLAGS:
1. INTERACTION RISKS -- where one drug's label explicitly mentions another
   drug or drug class that is also on this patient's list
2. TIMING CONFLICTS -- where label guidance implies medications should not
   be taken close together in time
3. PRESCRIBER BLIND SPOTS -- where two medications from different doctors
   appear to conflict in ways neither prescriber may have seen

ANTIGRAVITY FALLBACK:
When openFDA has no label for a drug (lookup_failures), the agent calls
get_drug_research_fallback which uses the Antigravity managed agent to
autonomously research that drug from multiple authoritative sources.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent

from tools.antigravity_client import get_drug_research_fallback
from tools.openfda_client import get_drug_safety_info

CONFLICT_RISK_INSTRUCTION = """\
You are the Conflict & Risk Agent for MedGuard, a medication safety
assistant. You receive a structured medication list from the Intake Agent.

Your job:

1. Call the get_drug_safety_info tool with the COMPLETE list of medication
   names to retrieve official FDA label text for each one. Pass all drug
   names in a single call -- not one call per drug.

2. Carefully read the drug_interactions, warnings, contraindications, and
   boxed_warning sections for EVERY medication on the list.

3. Reason across the WHOLE list simultaneously (not just pairs) to identify:

   INTERACTION RISKS:
   Where one drug's label text explicitly mentions another drug or its drug
   class that is also on this patient's list.

   TIMING CONFLICTS:
   Where label guidance implies two medications should not be taken close
   together in time.

   PRESCRIBER BLIND SPOTS:
   Where two medications come from different prescribers and appear to
   conflict. Flag these explicitly as: "These medications were prescribed
   by different doctors who may not have seen each other's prescriptions."

4. RELEVANCE FILTER -- only flag boxed warnings directly relevant to THIS
   patient's specific situation. Do NOT list every boxed warning for every
   drug -- only ones that matter for this specific combination.

5. For EVERY flag record ALL of the following:
   - Which drug's label the finding came from
   - Which section: Boxed Warning / Drug Interactions / Contraindications / Warnings
   - The specific phrase that triggered the flag (one sentence maximum)
   - Severity:
     "critical"       -- Boxed Warning or explicit Contraindication
     "moderate"       -- Drug Interactions or Warnings section mention
     "informational"  -- worth noting, low clinical urgency

6. Write each explanation in plain language a non-clinician can understand.

IMPORTANT -- ANTIGRAVITY FALLBACK:
If the get_drug_safety_info response contains any names in lookup_failures,
you MUST immediately call get_drug_research_fallback with exactly those
drug names before proceeding. The Antigravity managed agent will
autonomously research those drugs from multiple authoritative sources.
Always clearly note when Antigravity research was used instead of FDA
label data.

LABEL FRESHNESS:
The FDA labels you receive were fetched live from api.fda.gov at the time
of this run -- not from a cache. They reflect the current published label.

CRITICAL BOUNDARIES -- never violate these:
- You are not a doctor. Never give medical advice.
- Never tell someone to stop, reduce, or change a medication.
- Never diagnose any condition.
- Always end your output with a clear statement that these findings are
  for review by a pharmacist or doctor -- not medical conclusions.
- If openFDA returned no label for a drug and Antigravity also failed,
  state this clearly rather than guessing.
"""

conflict_risk_agent = Agent(
    model="gemini-2.5-flash",
    name="conflict_risk_agent",
    description=(
        "Core reasoning agent. Calls the openFDA Drug Label API tool to "
        "retrieve live FDA label text for every medication, then reasons "
        "across the full list to flag drug-drug interactions, timing "
        "conflicts, and prescriber blind spots. Every finding is tagged "
        "with severity level and exact FDA label section citation. "
        "Uses Antigravity as fallback for drugs with no FDA label."
    ),
    instruction=CONFLICT_RISK_INSTRUCTION,
    # Two tools: openFDA for primary lookup, Antigravity as fallback
    tools=[get_drug_safety_info, get_drug_research_fallback],
)