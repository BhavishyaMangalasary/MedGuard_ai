"""
tests/test_secure_storage.py

Unit tests for the secure storage module.

TESTING STRATEGY:
Tests run against a temporary directory and a fresh encryption key
generated per test -- never touching the real data directory or
the production key from .env. Each test is fully isolated.

HOW TO RUN:
    python -m pytest tests/ -v
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet

TEST_KEY = Fernet.generate_key().decode()


class TestSecureStorage(unittest.TestCase):

    def setUp(self):
        """
        Set up a fresh temporary directory and test encryption key
        before each test. Ensures complete test isolation.
        """
        self.temp_dir = tempfile.mkdtemp()

        self.env_patcher = patch.dict(
            os.environ,
            {"MEDGUARD_ENCRYPTION_KEY": TEST_KEY}
        )
        self.env_patcher.start()

        self.dir_patcher = patch(
            "tools.secure_storage.DATA_DIR",
            Path(self.temp_dir)
        )
        self.dir_patcher.start()

        from tools.secure_storage import (
            SecureStorageError,
            delete_patient_record,
            load_patient_record,
            record_exists,
            save_patient_record,
        )
        self.save = save_patient_record
        self.load = load_patient_record
        self.delete = delete_patient_record
        self.exists = record_exists
        self.SecureStorageError = SecureStorageError

    def tearDown(self):
        self.env_patcher.stop()
        self.dir_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_creates_encrypted_file(self):
        """Saving a record creates a file that is not plaintext."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        record_path = Path(self.temp_dir) / "janet.json"
        self.assertTrue(record_path.exists())
        raw_content = record_path.read_text()
        self.assertNotIn("warfarin", raw_content)

    def test_save_and_load_round_trip(self):
        """Data saved can be loaded back correctly."""
        original = {"meds": ["warfarin", "ibuprofen"]}
        self.save("Janet", original, access_scope=["janet"])
        loaded = self.load("Janet", "janet")
        self.assertEqual(loaded, original)

    def test_load_nonexistent_record_raises_error(self):
        """Loading a record that doesn't exist raises SecureStorageError."""
        with self.assertRaises(self.SecureStorageError):
            self.load("Nobody", "nobody")

    def test_access_control_blocks_unauthorized_user(self):
        """Unauthorized user cannot load another patient's record."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        with self.assertRaises(self.SecureStorageError) as ctx:
            self.load("Janet", "unauthorized_user")
        self.assertIn("Access denied", str(ctx.exception))

    def test_access_control_allows_authorized_user(self):
        """Authorized user in access_scope can load the record."""
        self.save(
            "Janet", {"meds": ["warfarin"]},
            access_scope=["janet", "caregiver_bob"]
        )
        record1 = self.load("Janet", "janet")
        record2 = self.load("Janet", "caregiver_bob")
        self.assertEqual(record1, record2)

    def test_record_exists_returns_true_when_saved(self):
        """record_exists returns True after a record is saved."""
        self.assertFalse(self.exists("Janet"))
        self.save("Janet", {"meds": []}, access_scope=["janet"])
        self.assertTrue(self.exists("Janet"))

    def test_record_exists_returns_false_when_not_saved(self):
        """record_exists returns False for a patient with no saved record."""
        self.assertFalse(self.exists("Nobody"))

    def test_delete_removes_record(self):
        """Deleting a record removes it from disk."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        self.delete("Janet")
        self.assertFalse(self.exists("Janet"))

    def test_delete_nonexistent_record_does_not_raise(self):
        """Deleting a record that doesn't exist does not raise an error."""
        try:
            self.delete("Nobody")
        except Exception as e:
            self.fail(f"delete_patient_record raised unexpectedly: {e}")

    def test_patient_name_case_insensitive(self):
        """Patient names are case-insensitive."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        record = self.load("janet", "janet")
        self.assertIsNotNone(record)

    def test_patient_name_with_spaces(self):
        """Patient names with spaces are handled correctly."""
        self.save("my mom", {"meds": ["metformin"]}, access_scope=["my_mom"])
        self.assertTrue(self.exists("my mom"))
        record = self.load("my mom", "my_mom")
        self.assertEqual(record["meds"], ["metformin"])

    def test_envelope_contains_unencrypted_metadata(self):
        """The storage envelope contains unencrypted metadata."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        record_path = Path(self.temp_dir) / "janet.json"
        envelope = json.loads(record_path.read_text())
        self.assertIn("patient_id", envelope)
        self.assertIn("access_scope", envelope)
        self.assertIn("ciphertext", envelope)
        self.assertEqual(envelope["patient_id"], "janet")

    def test_wrong_encryption_key_raises_error(self):
        """Decryption with wrong key raises SecureStorageError."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        wrong_key = Fernet.generate_key().decode()
        with patch.dict(os.environ, {"MEDGUARD_ENCRYPTION_KEY": wrong_key}):
            with self.assertRaises(self.SecureStorageError) as ctx:
                self.load("Janet", "janet")
            self.assertIn("Decryption failed", str(ctx.exception))

    def test_save_overwrites_existing_record(self):
        """Saving a record for an existing patient overwrites the old one."""
        self.save("Janet", {"meds": ["warfarin"]}, access_scope=["janet"])
        self.save("Janet", {"meds": ["metformin"]}, access_scope=["janet"])
        record = self.load("Janet", "janet")
        self.assertEqual(record["meds"], ["metformin"])

    def test_missing_encryption_key_raises_error(self):
        """Missing encryption key raises SecureStorageError immediately."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MEDGUARD_ENCRYPTION_KEY", None)
            with self.assertRaises(self.SecureStorageError) as ctx:
                self.save("Janet", {}, access_scope=["janet"])
            self.assertIn("MEDGUARD_ENCRYPTION_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)