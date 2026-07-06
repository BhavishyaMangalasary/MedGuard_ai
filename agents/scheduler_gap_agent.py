"""
agents/scheduler_gap_agent.py

Agent 3: Scheduler & Gap Agent

ROLE IN PIPELINE:
Third agent in the MedGuard sequential pipeline. Handles the logistics
side of medication management -- when to take each drug and when to
reorder.

WHY SEPARATE FROM THE CONFLICT & RISK AGENT:
Scheduling and refill tracking require a different kind of reasoning
than clinical text analysis:
- Scheduling is arithmetic + time constraint satisfaction
- Refill tracking is date arithmetic
- Clinical risk reasoning is cross-document inference

Combining these into one agent would create a prompt that tries to do
too many things, degrading performance on all of them. Separation also
makes each agent independently testable.

TIMING CONFLICT INTEGRATION:
This agent receives timing conflict flags from the Conflict & Risk Agent
via the shared session context (ADK passes each agent's output forward
automatically). When a timing conflict is flagged, the schedule must
reflect this constraint and explain why.

GRACEFUL HANDLING OF MISSING DATA:
Many caregivers won't know exact fill dates or days supply. Rather than
failing or guessing, this agent explicitly notes when refill tracking
isn't possible for a specific medication.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent

SCHEDULER_GAP_INSTRUCTION = """\
You are the Scheduler & Gap Agent for MedGuard, a medication safety
assistant. You receive:
  - A structured medication list from the Intake Agent
  - Any timing conflict flags from the Conflict & Risk Agent

Your job has two clearly separated parts:

PART 1 -- BUILD A DAILY SCHEDULE:

Convert each medication's frequency into specific times of day:
  - "once daily" → morning (unless label suggests otherwise e.g. Warfarin → evening)
  - "twice daily" → morning and evening
  - "three times daily" → morning, afternoon, and evening
  - "with meals" → morning (breakfast) and evening (dinner) for twice daily
  - "as needed" → list separately under As Needed
  - "at bedtime" → Bedtime slot

If a timing conflict flag says two medications must be spaced apart,
schedule them in different time slots and add a plain-language note
explaining why.

Group output by time of day:
  Morning | Afternoon | Evening | Bedtime | As Needed

If frequency is missing or ambiguous, say so explicitly rather than
guessing.

PART 2 -- FLAG REFILL GAPS:

For each medication where BOTH days_supply AND last_filled are known:
  - Calculate: run_out_date = last_filled + days_supply
  - If run_out_date is within 7 days, raise a refill alert with days remaining
  - Mention the prescribing doctor so the caregiver knows who to call

For each medication where days_supply OR last_filled is missing:
  - Note explicitly that refill tracking is not possible -- do not guess

Keep output structured and scannable. This is read by a busy caregiver.
"""

scheduler_gap_agent = Agent(
    model="gemini-2.5-flash",
    name="scheduler_gap_agent",
    description=(
        "Builds a practical daily medication schedule from the structured "
        "medication list, respecting any timing conflict flags from the "
        "Conflict & Risk Agent. Separately calculates refill gaps and flags "
        "medications running low within 7 days. Handles missing data "
        "gracefully -- notes when refill tracking isn't possible."
    ),
    instruction=SCHEDULER_GAP_INSTRUCTION,
    tools=[],  # No external tools needed -- pure reasoning over session context
)