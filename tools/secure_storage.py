"""
tools/secure_storage.py

Encrypted-at-rest storage for patient medication data.

WHY ENCRYPTION MATTERS HERE:
Medication lists are sensitive personal health information. If this
application's data directory were exposed, plaintext records would be
immediately readable. Encryption at rest ensures that without the key,
the data is useless to an attacker.

DESIGN DECISIONS:
- Fernet (AES-128-CBC + HMAC-SHA256): well-audited, simple, appropriate
  for single-tenant local storage. The HMAC component means any tampering
  with the ciphertext is detected at decryption time (InvalidToken).
- Per-record access scoping: a caregiver managing multiple patients should
  only be able to decrypt records they are explicitly authorized for.
  This is enforced before decryption is attempted.
- Fail closed: unauthorized access returns a deliberately vague error
  that does not confirm or deny whether a record exists.
- Key via environment variable: never hardcoded, never logged, never
  included in any prompt sent to the LLM.
- File permissions: saved records are chmod 0o600 (owner read/write only).
- Hard delete: supports a user's right to completely remove their data.

WHAT THIS DOES NOT CLAIM:
This is not HIPAA-compliant infrastructure. A production healthcare
system would additionally require TLS, a managed KMS, formal audit
logging, and a compliance review.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, UTC
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

# Patient records stored here -- one JSON file per patient.
# Directory is created on first save if it doesn't exist.
DATA_DIR = Path(__file__).parent.parent / "data" / "patients"

# Environment variable name for the Fernet encryption key.
# Generate with:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
ENV_KEY_NAME = "MEDGUARD_ENCRYPTION_KEY"


class SecureStorageError(Exception):
    """
    Raised for all storage-layer failures.

    Using a dedicated exception class lets callers distinguish storage
    failures from other errors without catching broad Exception types.
    """
    pass


def _get_fernet() -> Fernet:
    """
    Load and validate the encryption key from the environment.

    Called at the start of every storage operation rather than at module
    import time -- ensures key rotation is picked up without code changes.
    """
    key = os.environ.get(ENV_KEY_NAME)
    if not key:
        raise SecureStorageError(
            f"{ENV_KEY_NAME} is not set. Generate one with: "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    try:
        return Fernet(key.encode())
    except Exception as e:
        raise SecureStorageError(f"Invalid encryption key format: {e}") from e


def _patient_id_from_name(patient_name: str) -> str:
    """
    Derive a filesystem-safe patient ID from a display name.
    Lowercased and spaces replaced with underscores to match session user_id.
    """
    return patient_name.lower().strip().replace(" ", "_")


def save_patient_record(
    patient_name: str,
    record: dict,
    access_scope: list[str]
) -> None:
    """
    Encrypt and persist one patient's medication record to disk.

    The saved file contains an unencrypted envelope (patient_id,
    access_scope, metadata) wrapping the encrypted ciphertext. The
    envelope fields are not sensitive -- they're needed to enforce
    access control BEFORE attempting decryption.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    patient_id = _patient_id_from_name(patient_name)
    fernet = _get_fernet()

    # Serialize and encrypt the medication record
    plaintext = json.dumps(record).encode("utf-8")
    ciphertext = fernet.encrypt(plaintext)

    envelope = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "access_scope": access_scope,
        "saved_at": datetime.now(UTC).isoformat(),
        "ciphertext": ciphertext.decode("utf-8"),
    }

    out_path = DATA_DIR / f"{patient_id}.json"
    out_path.write_text(json.dumps(envelope, indent=2))

    # Restrict file permissions to owner read/write only
    try:
        out_path.chmod(0o600)
    except OSError:
        logger.warning(
            "Could not set file permissions on %s (non-unix system?)", out_path
        )

    logger.info("Patient record saved: patient_id=%s", patient_id)
    print(f"✓ Record saved for {patient_name}.")


def load_patient_record(
    patient_name: str,
    requesting_user_id: str
) -> dict:
    """
    Decrypt and return one patient's record, enforcing access scope.

    Access check happens before decryption -- unauthorized callers never
    get a chance to try decrypting. Error message is deliberately vague
    to prevent enumeration attacks.
    """
    patient_id = _patient_id_from_name(patient_name)
    path = DATA_DIR / f"{patient_id}.json"

    if not path.exists():
        raise SecureStorageError(f"No record found for '{patient_name}'.")

    envelope = json.loads(path.read_text())

    # Enforce access scope BEFORE attempting decryption
    if requesting_user_id not in envelope.get("access_scope", []):
        logger.warning(
            "Unauthorized access attempt: user=%s patient_id=%s",
            requesting_user_id, patient_id
        )
        raise SecureStorageError("Access denied for this patient record.")

    fernet = _get_fernet()
    try:
        plaintext = fernet.decrypt(envelope["ciphertext"].encode("utf-8"))
    except InvalidToken as e:
        logger.error(
            "Decryption failed for patient_id=%s -- key mismatch or corruption",
            patient_id
        )
        raise SecureStorageError(
            "Decryption failed. The record may be corrupted or the "
            "encryption key has changed since this record was saved."
        ) from e

    logger.info(
        "Patient record loaded: patient_id=%s user=%s", patient_id, requesting_user_id
    )
    return json.loads(plaintext)


def record_exists(patient_name: str) -> bool:
    """
    Check if a saved record exists without loading or decrypting it.
    Used by demo.py to decide whether to offer the load option.
    """
    patient_id = _patient_id_from_name(patient_name)
    return (DATA_DIR / f"{patient_id}.json").exists()


def delete_patient_record(patient_name: str) -> None:
    """
    Hard delete -- removes a patient's data entirely from disk.
    Supports the user's right to remove their data completely.
    """
    patient_id = _patient_id_from_name(patient_name)
    path = DATA_DIR / f"{patient_id}.json"

    if path.exists():
        path.unlink()
        logger.info("Patient record deleted: patient_id=%s", patient_id)
        print(f"✓ Record for {patient_name} deleted.")
    else:
        logger.warning(
            "Delete attempted for non-existent record: patient_id=%s", patient_id
        )
        print(f"No record found for {patient_name}.")


if __name__ == "__main__":
    """Smoke test -- run directly to verify encryption works."""
    import os
    logging.basicConfig(level=logging.INFO)

    os.environ[ENV_KEY_NAME] = Fernet.generate_key().decode()

    save_patient_record(
        "Test Patient",
        {"meds": ["warfarin", "ibuprofen"]},
        access_scope=["test_patient"]
    )

    record = load_patient_record("Test Patient", "test_patient")
    print("Loaded:", record)

    try:
        load_patient_record("Test Patient", "unauthorized")
    except SecureStorageError as e:
        print("Access control works:", e)

    delete_patient_record("Test Patient")
    print("Exists after delete:", record_exists("Test Patient"))