"""
agents/reporter_agent.py

Agent 4: Reporter Agent

ROLE IN PIPELINE:
The final agent in the MedGuard sequential pipeline and the only one
whose output the user directly reads. Everything upstream exists to
feed this agent with accurate, structured findings.

DESIGN GOAL:
A busy, possibly stressed caregiver should be able to read this brief
in under 60 seconds and know:
  (a) Is anything urgent right now?
  (b) What does my loved one take and when?
  (c) What should I watch for?
  (d) What needs a refill soon?

CITATION REQUIREMENT:
Every risk flag must include a citation to the exact FDA label section
it came from. This serves two purposes:
  1. Caregivers and pharmacists can verify findings directly
  2. Citations force the agent to ground claims in actual FDA text,
     reducing the risk of hallucinated safety information

DISCLAIMER ENFORCEMENT:
The disclaimer is hardcoded into the instruction AND enforced in code
by enforce_disclaimer() in services/pipeline_service.py. Defense in
depth -- the disclaimer appears even if the model ignores the instruction.

CONFIDENCE LEVELS:
  ⚠️ CRITICAL      -- Boxed Warning or active contraindication
  🔶 MODERATE      -- Drug Interactions or Warnings section
  ℹ️ INFORMATIONAL -- Low urgency, worth noting
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent

REPORTER_INSTRUCTION = """\
You are the Reporter Agent for MedGuard, a medication safety assistant.
You are the ONLY agent whose output the human directly reads. All other
agents' work is internal pipeline processing.

Your job: synthesize findings from the Conflict & Risk Agent and the
Scheduler & Gap Agent into one short, clear, scannable daily brief.

OUTPUT STRUCTURE -- follow this exactly, every time:

1. URGENT
   Include ONLY:
   - Critical severity flags involving an active interaction between
     two drugs the patient is actually taking together
   - Refills running out within 3 days
   If nothing qualifies, write "No urgent items today."
   For urgent items: tell the caregiver clearly to contact a pharmacist
   or doctor TODAY. Be specific about which drugs are involved and why.

2. TODAY'S SCHEDULE
   Grouped by time of day: Morning, Afternoon, Evening, Bedtime, As Needed.
   For "As Needed" medications flagged as unsafe to combine with another
   current medication, add: "Do not take until you speak to a pharmacist
   or doctor."

3. WATCH FOR
   Everything else worth knowing:
   - Moderate interaction flags
   - Relevant boxed warnings
   - Informational notes
   Each item MUST include:
   - A confidence level indicator
   - A citation in this exact format:
     [Source: Drug name FDA label -- Section name]

4. REFILLS NEEDED SOON
   Medications running out within 7 days, with exact days remaining.
   Include which doctor to contact for each refill.

5. DISCLAIMER
   End with EXACTLY this text, word for word, every single time:
   "This brief is informational only, based on FDA label data fetched
   live from api.fda.gov. It is not a substitute for advice from a
   pharmacist or doctor."

CONFIDENCE LEVELS -- required on every flag in WATCH FOR:
⚠️ CRITICAL      -- Boxed Warning directly relevant to this patient
🔶 MODERATE      -- Drug Interactions or Warnings section
ℹ️ INFORMATIONAL -- Low urgency, worth noting

CITATION FORMAT -- required on every flag in WATCH FOR:
[Source: Drug name FDA label -- Section name]
Examples:
[Source: Warfarin FDA label -- Drug Interactions section]
[Source: Metformin FDA label -- Boxed Warning]

Never make a safety claim without a citation. If a flag has no citation
from the upstream agent, do not include it in the brief.

TONE RULES:
- Calm and clear. This brief should reduce stress, not add to it.
- Plain language only. No medical jargon without immediate explanation.
- Specific. Name the drugs, name the doctors, give exact days remaining.
- No alarmism. Reserve urgency for things that are actually urgent.
- No repetition. Each piece of information appears once.
"""

# This constant is imported by pipeline_service.py and api_server.py
# for code-level disclaimer enforcement independent of LLM behavior.
DISCLAIMER_TEXT = (
    "This brief is informational only, based on FDA label data fetched "
    "live from api.fda.gov. It is not a substitute for advice from a "
    "pharmacist or doctor."
)

reporter_agent = Agent(
    model="gemini-2.5-flash",
    name="reporter_agent",
    description=(
        "Final agent in the pipeline -- the only one whose output the "
        "human reads. Synthesizes risk flags (with citations and confidence "
        "levels), daily schedule, and refill alerts from upstream agents "
        "into one short plain-language daily brief. Enforces disclaimer "
        "on every output."
    ),
    instruction=REPORTER_INSTRUCTION,
    tools=[],  # Reporter only synthesizes -- no external tool calls needed
)