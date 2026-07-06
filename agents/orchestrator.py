"""
agents/orchestrator.py

Root orchestrator for the MedGuard multi-agent pipeline.

ARCHITECTURE:
MedGuard uses ADK's SequentialAgent to chain four specialist agents
into a single pipeline. Each agent has one clearly defined responsibility
and passes its output forward via the shared session context.

PIPELINE ORDER AND RATIONALE:
1. Intake Agent       -- must run first to normalize raw input into
                         structured data that downstream agents can parse
2. Conflict & Risk    -- needs structured med list from Intake, and its
                         timing conflict flags are needed by Scheduler
3. Scheduler & Gap    -- needs both med list (Intake) and timing flags
                         (Conflict & Risk)
4. Reporter           -- must run last, synthesizes all upstream output

WHY SEQUENTIAL OVER PARALLEL:
The agents have data dependencies. ADK's SequentialAgent handles this
automatically by appending each agent's response to the shared session
context, making all upstream output available to every downstream agent
without any manual orchestration code.

WHY FOUR AGENTS INSTEAD OF ONE:
A single monolithic agent attempting to normalize input, reason over
clinical text, do scheduling arithmetic, AND write a plain-language
summary in one pass tends to perform worse at each task. Separation means:
- Each agent can be tested and evaluated independently
- A failure in one stage is easy to localize
- Future capability additions slot into the appropriate agent

HOW TO RUN:
  Terminal CLI:    python demo.py
  Browser UI:      adk web  (then open http://127.0.0.1:8000)
  REST API:        uvicorn api_server:app --port 8080
"""

from __future__ import annotations

from google.adk.agents.sequential_agent import SequentialAgent

from agents.conflict_risk_agent import conflict_risk_agent
from agents.intake_agent import intake_agent
from agents.reporter_agent import reporter_agent
from agents.scheduler_gap_agent import scheduler_gap_agent

# root_agent is the ADK entry point -- this name is required for ADK
# to discover and run the pipeline via 'adk web' or 'adk run'
root_agent = SequentialAgent(
    name="medguard_pipeline",
    description=(
        "MedGuard: a 4-agent medication safety pipeline. "
        "Takes a patient's raw medication list and produces a daily brief "
        "covering drug interaction risks (with FDA label citations), a "
        "practical daily schedule, and refill alerts. "
        "Uses live FDA Drug Label API data -- never cached or stale. "
        "Track: Concierge Agents."
    ),
    sub_agents=[
        intake_agent,         # Step 1: normalize raw input
        conflict_risk_agent,  # Step 2: flag clinical risks using live FDA data
        scheduler_gap_agent,  # Step 3: build schedule, detect refill gaps
        reporter_agent,       # Step 4: synthesize into human-readable brief
    ],
)