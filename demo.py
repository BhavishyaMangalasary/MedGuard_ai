"""
demo.py

MedGuard - Medication Safety Assistant
Interactive CLI entry point for the MedGuard pipeline.

USAGE:
    python demo.py

WHAT THIS FILE DOES:
1. Collects patient name and medication list from the user
2. Checks for a saved encrypted record (load or enter new)
3. Validates input before sending to the pipeline
4. Runs the 4-agent ADK pipeline via services/pipeline_service.py
5. Enforces disclaimer as a code-level safety net
6. Displays only the Reporter Agent's final output
7. Offers to save the medication list encrypted for next time

DESIGN DECISION -- single entry point:
The actual pipeline logic lives in services/pipeline_service.py so it
can be shared with api_server.py. demo.py is purely the CLI interface.
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()

from services.logging_utils import configure_logging
from services.pipeline_service import (
    enforce_disclaimer,
    run_pipeline,
    validate_input,
)
from tools.secure_storage import (
    SecureStorageError,
    load_patient_record,
    record_exists,
    save_patient_record,
)

configure_logging()


def check_existing_record(patient_name: str) -> dict | None:
    """Check for and load a saved record for this patient."""
    if not record_exists(patient_name):
        return None
    try:
        patient_id = patient_name.lower().replace(" ", "_")
        return load_patient_record(patient_name, patient_id)
    except SecureStorageError:
        return None


def get_medication_input(patient_name: str) -> str:
    """Prompt user to type a new medication list."""
    print(f"\nPlease enter {patient_name}'s medication list below.")
    print("Include drug name, dose, frequency, which doctor")
    print("prescribed it, days supply, and last fill date if known.")
    print("\nType DONE on a new line when finished.\n")

    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "DONE":
                break
            lines.append(line)
        except EOFError:
            break

    return "\n".join(lines)


def ask_to_save(patient_name: str, medication_text: str) -> None:
    """Offer to save the medication list for future runs."""
    print("\nSave this medication list for next time? (yes/no)")
    answer = input("> ").strip().lower()

    if answer in ("yes", "y"):
        patient_id = patient_name.lower().replace(" ", "_")
        try:
            save_patient_record(
                patient_name=patient_name,
                record={
                    "patient_name": patient_name,
                    "medication_text": medication_text,
                },
                access_scope=[patient_id]
            )
        except SecureStorageError as e:
            print(f"⚠️  Could not save record: {e}")
    else:
        print("Record not saved.")


def get_user_input() -> tuple[str, str, bool]:
    """
    Collect patient name and medication list from the user.

    Returns:
        (patient_name, medication_text, is_from_saved_record)
        The third element tells the caller whether to offer saving.
    """
    print("\n" + "=" * 60)
    print("  Welcome to MedGuard - Medication Safety Assistant")
    print("=" * 60)
    print("\nWho are you managing medications for?\n")
    patient_name = input("> ").strip()

    if not patient_name:
        patient_name = "the patient"

    existing = check_existing_record(patient_name)

    if existing:
        print(f"\nFound a saved record for {patient_name}.")
        print("1. Load saved record")
        print("2. Enter a new list\n")
        choice = input("> ").strip()

        if choice == "1":
            print(f"\nLoading {patient_name}'s saved medication list...")
            medication_text = existing.get("medication_text", "")
            return patient_name, medication_text, True
        else:
            medication_text = get_medication_input(patient_name)
            return patient_name, medication_text, False
    else:
        medication_text = get_medication_input(patient_name)
        return patient_name, medication_text, False


async def main() -> None:
    patient_name, medication_text, is_from_saved_record = get_user_input()

    # Validate before hitting the API
    error = validate_input(patient_name, medication_text)
    if error:
        print(f"\n⚠️  {error}")
        sys.exit(1)

    print(f"\nAnalyzing {patient_name}'s medication list...\n")

    reporter_output = await run_pipeline(patient_name, medication_text)

    if not reporter_output:
        print("Something went wrong -- the pipeline did not produce output.")
        print("Please check your API key and try again.")
        sys.exit(1)

    # Enforce disclaimer in code as safety net
    reporter_output = enforce_disclaimer(reporter_output)

    # Display the final brief
    print("\n" + "=" * 60)
    print(f"  MEDGUARD DAILY BRIEF — {patient_name.upper()}")
    print("=" * 60 + "\n")
    print(reporter_output)

    # Only offer to save if this was new input
    if not is_from_saved_record:
        ask_to_save(patient_name, medication_text)


if __name__ == "__main__":
    asyncio.run(main())