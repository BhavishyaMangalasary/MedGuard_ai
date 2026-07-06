"""
tests/test_openfda_client.py

Unit tests for the openFDA client module.

TESTING STRATEGY:
We test the logic layer (DrugLabel normalization, context block rendering,
input validation, graceful degradation) without hitting the live API.
All HTTP calls are mocked using unittest.mock so tests run offline,
are deterministic, and don't consume API quota.

Live API connectivity is verified separately by running:
    python tools/openfda_client.py

HOW TO RUN:
    python -m pytest tests/ -v
"""

import unittest
from unittest.mock import MagicMock, patch

from tools.openfda_client import (
    DrugLabel,
    get_drug_safety_info,
    lookup_drug_label,
    lookup_multiple,
)


class TestDrugLabel(unittest.TestCase):
    """Tests for the DrugLabel dataclass and to_context_block() method."""

    def test_found_label_with_all_fields(self):
        """A fully populated label renders all sections in priority order."""
        label = DrugLabel(
            query="warfarin",
            found=True,
            brand_name="Coumadin",
            generic_name="WARFARIN SODIUM",
            rxcui=["11289"],
            boxed_warning="Warfarin can cause major or fatal bleeding.",
            drug_interactions="NSAIDs increase bleeding risk.",
            contraindications="Pregnancy.",
            warnings="Monitor INR regularly.",
        )
        block = label.to_context_block()

        # Boxed warning must appear first
        self.assertIn("BOXED WARNING", block)
        boxed_pos = block.index("BOXED WARNING")
        interactions_pos = block.index("Drug Interactions")
        self.assertLess(boxed_pos, interactions_pos)

        self.assertIn("Warfarin can cause major or fatal bleeding.", block)
        self.assertIn("NSAIDs increase bleeding risk.", block)
        self.assertIn("Pregnancy.", block)
        self.assertIn("Monitor INR regularly.", block)

    def test_found_label_uses_brand_name_first(self):
        """Brand name takes priority over generic name in the header."""
        label = DrugLabel(
            query="warfarin", found=True,
            brand_name="Coumadin", generic_name="WARFARIN SODIUM",
        )
        self.assertIn("Coumadin", label.to_context_block())

    def test_found_label_falls_back_to_generic_name(self):
        """Falls back to generic name when brand name is absent."""
        label = DrugLabel(
            query="metformin", found=True,
            brand_name=None, generic_name="METFORMIN HYDROCHLORIDE",
        )
        self.assertIn("METFORMIN HYDROCHLORIDE", label.to_context_block())

    def test_found_label_falls_back_to_query(self):
        """Falls back to query string when both name fields are absent."""
        label = DrugLabel(query="someunknowndrug", found=True)
        self.assertIn("someunknowndrug", label.to_context_block())

    def test_not_found_label_returns_clear_message(self):
        """A not-found label returns a clear message."""
        label = DrugLabel(
            query="fakdrug123", found=False,
            error="No matching FDA label found."
        )
        block = label.to_context_block()
        self.assertIn("fakdrug123", block)
        self.assertIn("No matching FDA label found.", block)

    def test_found_label_with_no_safety_sections(self):
        """A label with no safety sections includes a fallback note."""
        label = DrugLabel(query="somedrug", found=True, brand_name="SomeDrug")
        self.assertIn("no structured safety sections", label.to_context_block())

    def test_long_fields_are_truncated(self):
        """Long label sections are truncated to prevent context overflow."""
        long_text = "A" * 2000
        label = DrugLabel(
            query="warfarin", found=True,
            brand_name="Coumadin",
            drug_interactions=long_text,
            boxed_warning=long_text,
        )
        self.assertLess(len(label.to_context_block()), 10000)

    def test_query_preserved_in_header(self):
        """The original query string is always preserved in the header."""
        label = DrugLabel(
            query="Lipitor", found=True, brand_name="ATORVASTATIN CALCIUM"
        )
        self.assertIn("queried as 'Lipitor'", label.to_context_block())


class TestLookupDrugLabel(unittest.TestCase):
    """Tests for lookup_drug_label() with mocked HTTP calls."""

    def test_empty_drug_name_returns_not_found(self):
        """Empty input is rejected immediately without hitting the API."""
        result = lookup_drug_label("")
        self.assertFalse(result.found)
        self.assertIn("Empty", result.error)

    def test_whitespace_only_name_returns_not_found(self):
        """Whitespace-only input is treated as empty."""
        result = lookup_drug_label("   ")
        self.assertFalse(result.found)

    @patch("tools.openfda_client.requests.get")
    def test_successful_lookup_returns_found_label(self, mock_get):
        """A successful API response returns a found DrugLabel."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{
                "openfda": {
                    "brand_name": ["Coumadin"],
                    "generic_name": ["WARFARIN SODIUM"],
                    "rxcui": ["11289"],
                },
                "drug_interactions": ["NSAIDs increase bleeding risk."],
                "warnings": ["Monitor INR regularly."],
                "boxed_warning": ["Warfarin can cause major or fatal bleeding."],
            }]
        }
        mock_get.return_value = mock_response

        result = lookup_drug_label("warfarin")
        self.assertTrue(result.found)
        self.assertEqual(result.brand_name, "Coumadin")
        self.assertIn("NSAIDs increase bleeding risk.", result.drug_interactions)

    @patch("tools.openfda_client.requests.get")
    def test_404_response_returns_not_found(self, mock_get):
        """A 404 response returns a not-found DrugLabel."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = lookup_drug_label("totallyfakedrugname")
        self.assertFalse(result.found)

    @patch("tools.openfda_client.requests.get")
    def test_empty_results_returns_not_found(self, mock_get):
        """An empty results array returns a not-found DrugLabel."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response

        result = lookup_drug_label("unknowndrug")
        self.assertFalse(result.found)

    @patch("tools.openfda_client.requests.get")
    def test_network_error_returns_not_found_after_retries(self, mock_get):
        """Network errors are retried and return not-found after exhaustion."""
        import requests as req
        mock_get.side_effect = req.exceptions.ConnectionError("Network unreachable")

        result = lookup_drug_label("warfarin")
        self.assertFalse(result.found)
        self.assertIn("failed", result.error.lower())


class TestLookupMultiple(unittest.TestCase):
    """Tests for lookup_multiple() batch lookups."""

    @patch("tools.openfda_client.lookup_drug_label")
    def test_returns_one_result_per_drug(self, mock_lookup):
        """Returns exactly one DrugLabel per input drug name."""
        mock_lookup.return_value = DrugLabel(query="warfarin", found=True)
        results = lookup_multiple(["warfarin", "ibuprofen", "metformin"])
        self.assertEqual(len(results), 3)

    @patch("tools.openfda_client.lookup_drug_label")
    def test_preserves_input_order(self, mock_lookup):
        """Results preserve the same order as input drug names."""
        def side_effect(name):
            return DrugLabel(query=name, found=True)
        mock_lookup.side_effect = side_effect

        drugs = ["warfarin", "ibuprofen", "metformin"]
        results = lookup_multiple(drugs)
        self.assertEqual([r.query for r in results], drugs)

    @patch("tools.openfda_client.lookup_drug_label")
    def test_handles_empty_list(self, mock_lookup):
        """Empty input returns empty list without calling the API."""
        results = lookup_multiple([])
        self.assertEqual(results, [])
        mock_lookup.assert_not_called()


class TestGetDrugSafetyInfo(unittest.TestCase):
    """Tests for the get_drug_safety_info ADK tool function."""

    @patch("tools.openfda_client.lookup_multiple")
    def test_returns_required_keys(self, mock_lookup):
        """Response always contains all required keys."""
        mock_lookup.return_value = [
            DrugLabel(query="warfarin", found=True, brand_name="Coumadin"),
        ]
        result = get_drug_safety_info(["warfarin"])
        required_keys = ["labels", "lookup_failures", "source",
                         "fetched_live_at", "freshness_note"]
        for key in required_keys:
            self.assertIn(key, result)

    @patch("tools.openfda_client.lookup_multiple")
    def test_lookup_failures_populated_correctly(self, mock_lookup):
        """Drugs with found=False appear in lookup_failures."""
        mock_lookup.return_value = [
            DrugLabel(query="warfarin", found=True, brand_name="Coumadin"),
            DrugLabel(query="fakdrug", found=False, error="Not found"),
        ]
        result = get_drug_safety_info(["warfarin", "fakdrug"])
        self.assertIn("fakdrug", result["lookup_failures"])
        self.assertNotIn("warfarin", result["lookup_failures"])

    @patch("tools.openfda_client.lookup_multiple")
    def test_fetched_live_at_is_present(self, mock_lookup):
        """Response includes a UTC timestamp."""
        mock_lookup.return_value = [DrugLabel(query="warfarin", found=True)]
        result = get_drug_safety_info(["warfarin"])
        self.assertIn("UTC", result["fetched_live_at"])

    @patch("tools.openfda_client.lookup_multiple")
    def test_labels_count_matches_input(self, mock_lookup):
        """Number of label blocks matches number of input drugs."""
        mock_lookup.return_value = [
            DrugLabel(query="warfarin", found=True),
            DrugLabel(query="ibuprofen", found=True),
        ]
        result = get_drug_safety_info(["warfarin", "ibuprofen"])
        self.assertEqual(len(result["labels"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)