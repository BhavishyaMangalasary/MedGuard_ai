"""
agents/intake_agent.py

Agent 1: Intake Agent

ROLE IN PIPELINE:
First agent in the MedGuard sequential pipeline. Receives raw user input
and produces structured medication records for downstream agents.

SINGLE RESPONSIBILITY:
This agent does ONE thing -- extract and normalize medication data from
free text. It does not assess safety, interactions, or risk. Keeping it
single-purpose means:
1. If downstream agents produce wrong output, we know it's not a parsing
   problem -- we can test the Intake Agent independently
2. The prompt stays focused and the agent performs consistently
3. Future improvements (e.g. OCR of pill bottle photos) slot in here
   without touching the clinical reasoning agents

DESIGN DECISION -- never ask for more info:
The agent is explicitly instructed to process whatever is given and mark
missing fields as unknown. A stressed caregiver shouldn't be interrogated
for information they may not have. The Conflict & Risk Agent handles
missing data gracefully downstream.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import Agent

INTAKE_INSTRUCTION = """\
You are the Intake Agent for MedGuard, a medication safety assistant.

Your ONLY job is to read whatever medication information the user provides
and convert it into a clean structured list immediately. You are the first
step in a pipeline -- other agents handle safety analysis downstream.

For each medication extract:
- name: the drug name exactly as given -- do not correct or guess
- dose: e.g. "10mg", "500mg" -- if not stated, use "unknown"
- frequency: e.g. "once daily", "twice a day" -- if not stated, use "unknown"
- prescriber: which doctor prescribed it -- if not stated, use "unknown"
- days_supply: how many days the current fill lasts -- if not stated, use "unknown"
- last_filled: when it was last picked up -- if not stated, use "unknown"

STRICT RULES -- follow these exactly:
- Process immediately with whatever information is given. Never ask for more.
- Never say information is missing before processing -- just mark it unknown.
- If input is ambiguous (e.g. "the blood pressure pill"), include it with
  name set to the description as given and a note that clarification is needed.
- Preserve the original text in a raw_input field so downstream agents and
  humans can always trace back to what was actually said.
- Do NOT add safety warnings, interaction notes, or medical commentary.
  That is explicitly handled by a different agent downstream.
- Always output the structured list right away -- never delay or ask questions.

Output a clean JSON array, one object per medication.
"""

# gemini-2.5-flash is used across all agents for consistency.
# It's fast enough for normalization tasks and shares quota efficiently
# across the 4-agent pipeline.
intake_agent = Agent(
    model="gemini-2.5-flash",
    name="intake_agent",
    description=(
        "Normalizes raw medication input (free text, lists, pharmacy notes) "
        "into structured records: name, dose, frequency, prescriber, "
        "days supply, last filled. Never asks for more information -- "
        "marks missing fields as unknown and processes immediately."
    ),
    instruction=INTAKE_INSTRUCTION,
    tools=[],  # Intake Agent needs no external tools -- pure text normalization
)