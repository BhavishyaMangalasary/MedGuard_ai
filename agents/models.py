"""
agents/models.py

Shared data structures passed between agents in the MedGuard pipeline.

WHY THIS FILE EXISTS:
Each of the 4 agents needs to pass structured data to the next agent.
Rather than passing raw strings or dicts, we define explicit dataclasses
here so every agent speaks the same language. This makes the pipeline
easier to debug -- if something goes wrong, you can inspect exactly
what data each agent produced.

These are plain dataclasses (not Pydantic models) because they are
internal pipeline state, not user-facing API contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time


@dataclass
class Medication:
    """
    One medication as normalized by the Intake Agent.

    The Intake Agent converts raw user text into one Medication per drug.
    Fields that the user didn't provide are left as None -- downstream
    agents handle missing data explicitly rather than guessing.
    """
    name: str                           # drug name as the user typed it
    dose: str | None = None             # e.g. "10mg", "500mg"
    frequency: str | None = None        # e.g. "once daily", "twice a day"
    times: list[time] = field(default_factory=list)  # specific times of day
    prescriber: str | None = None       # which doctor prescribed it
    days_supply: int | None = None      # how many days the current fill lasts
    last_filled: str | None = None      # ISO date string of last fill
    raw_input: str | None = None        # original text -- preserved for traceability


@dataclass
class PatientProfile:
    """
    Container for one patient's full medication picture.

    Holds all medications for a single patient. The patient_id is
    derived from the patient name and used as the storage key in
    secure_storage.py.
    """
    patient_id: str
    display_name: str
    medications: list[Medication] = field(default_factory=list)


@dataclass
class RiskFlag:
    """
    One identified safety risk found by the Conflict & Risk Agent.

    Every flag must have a label_section -- this is the citation that
    traces the finding back to specific FDA label text. Flags without
    citations are stripped by the hallucination guard in safety_rules.py.
    """
    severity: str           # "critical" | "moderate" | "informational"
    category: str           # "interaction" | "timing_conflict" | "prescriber_blind_spot"
    drugs_involved: list[str]
    explanation: str        # plain language description for caregivers
    label_section: str      # e.g. "Warfarin FDA label -- Drug Interactions section"
    source: str = "FDA openFDA Drug Label API"


@dataclass
class ScheduleEntry:
    """One time slot in the daily medication schedule."""
    time_of_day: str            # "Morning" | "Afternoon" | "Evening" | "Bedtime" | "As Needed"
    medications: list[str]      # list of drug names for this slot
    notes: str | None = None    # timing notes e.g. "space 4 hours from Warfarin"


@dataclass
class RefillAlert:
    """A medication running low that needs a refill soon."""
    medication: str
    days_remaining: int | None  # None if days_supply or last_filled unknown
    message: str                # plain language alert for the caregiver


@dataclass
class DailyBrief:
    """
    Final output of the MedGuard pipeline -- what the human actually reads.

    The Reporter Agent fills this in and hands it to the user.
    All other agents' output is internal pipeline state.
    """
    patient_name: str
    generated_at: str           # UTC timestamp of when the brief was generated
    risk_flags: list[RiskFlag] = field(default_factory=list)
    schedule: list[ScheduleEntry] = field(default_factory=list)
    refill_alerts: list[RefillAlert] = field(default_factory=list)
    summary_text: str = ""      # the full plain-language brief text