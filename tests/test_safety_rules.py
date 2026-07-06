"""
tests/test_safety_rules.py

Unit tests for the confidence calibration and hallucination guard layer.

HOW TO RUN:
    python -m pytest tests/ -v
"""

import unittest

from tools.safety_rules import (
    calculate_confidence_score,
    calibrate_severity,
    get_severity_emoji,
    has_valid_citation,
    process_flags,
)


class TestHasValidCitation(unittest.TestCase):

    def test_valid_citation_detected(self):
        text = "Warfarin increases bleeding risk. [Source: Warfarin FDA label -- Drug Interactions section]"
        self.assertTrue(has_valid_citation(text))

    def test_valid_citation_boxed_warning(self):
        text = "Fatal bleeding risk. [Source: Warfarin FDA label -- Boxed Warning]"
        self.assertTrue(has_valid_citation(text))

    def test_no_citation_returns_false(self):
        text = "Warfarin and ibuprofen can cause bleeding."
        self.assertFalse(has_valid_citation(text))

    def test_malformed_citation_returns_false(self):
        text = "See: Warfarin label for details."
        self.assertFalse(has_valid_citation(text))

    def test_citation_case_insensitive(self):
        text = "Risk noted. [source: warfarin fda label -- drug interactions]"
        self.assertTrue(has_valid_citation(text))

    def test_empty_string_returns_false(self):
        self.assertFalse(has_valid_citation(""))


class TestCalibrateSeverity(unittest.TestCase):

    def test_boxed_warning_always_critical(self):
        text = "This has a boxed warning for bleeding risk."
        self.assertEqual(calibrate_severity(text, "moderate"), "critical")

    def test_fatal_language_always_critical(self):
        text = "This combination can be fatal."
        self.assertEqual(calibrate_severity(text, "informational"), "critical")

    def test_lactic_acidosis_always_critical(self):
        text = "Metformin risk of lactic acidosis."
        self.assertEqual(calibrate_severity(text, "moderate"), "critical")

    def test_contraindicated_always_critical(self):
        text = "These drugs are contraindicated together."
        self.assertEqual(calibrate_severity(text, "informational"), "critical")

    def test_drug_interactions_section_at_least_moderate(self):
        text = "Per drug interactions section, monitor closely."
        self.assertEqual(calibrate_severity(text, "informational"), "moderate")

    def test_moderate_stays_moderate(self):
        text = "May increase the risk of kidney problems."
        self.assertEqual(calibrate_severity(text, "moderate"), "moderate")

    def test_critical_stays_critical(self):
        text = "Minor timing consideration."
        self.assertEqual(calibrate_severity(text, "critical"), "critical")

    def test_informational_stays_informational(self):
        text = "Take with food for best absorption."
        self.assertEqual(calibrate_severity(text, "informational"), "informational")

    def test_case_insensitive_matching(self):
        text = "BOXED WARNING: major bleeding risk."
        self.assertEqual(calibrate_severity(text, "moderate"), "critical")


class TestCalculateConfidenceScore(unittest.TestCase):

    def test_fully_cited_critical_flag_high_score(self):
        text = "Fatal bleeding risk. [Source: Warfarin FDA label -- Boxed Warning]"
        score = calculate_confidence_score(text, "critical")
        self.assertGreater(score, 0.7)

    def test_uncited_flag_lower_score(self):
        cited = "Risk noted. [Source: Warfarin FDA label -- Drug Interactions section]"
        uncited = "Risk noted."
        self.assertGreater(
            calculate_confidence_score(cited, "moderate"),
            calculate_confidence_score(uncited, "moderate")
        )

    def test_critical_scores_higher_than_informational(self):
        text = "Some safety note. [Source: Drug FDA label -- Warnings section]"
        self.assertGreater(
            calculate_confidence_score(text, "critical"),
            calculate_confidence_score(text, "informational")
        )

    def test_score_never_exceeds_one(self):
        text = "Fatal boxed warning. [Source: Warfarin FDA label -- Boxed Warning]"
        self.assertLessEqual(calculate_confidence_score(text, "critical"), 1.0)

    def test_score_always_non_negative(self):
        self.assertGreaterEqual(
            calculate_confidence_score("", "informational"), 0.0
        )


class TestProcessFlags(unittest.TestCase):

    def test_uncited_flags_are_stripped(self):
        flags = [
            {
                "text": "Cited flag. [Source: Warfarin FDA label -- Drug Interactions section]",
                "severity": "moderate",
                "drugs_involved": ["warfarin"],
            },
            {
                "text": "Uncited flag with no citation at all.",
                "severity": "moderate",
                "drugs_involved": ["ibuprofen"],
            },
        ]
        result = process_flags(flags)
        self.assertEqual(len(result), 1)
        self.assertIn("Cited flag", result[0]["text"])

    def test_severity_calibrated_in_output(self):
        flags = [{
            "text": "Boxed warning risk. [Source: Warfarin FDA label -- Boxed Warning]",
            "severity": "moderate",
            "drugs_involved": ["warfarin"],
        }]
        result = process_flags(flags)
        self.assertEqual(result[0]["severity"], "critical")

    def test_flags_sorted_by_confidence(self):
        flags = [
            {
                "text": "Low confidence. [Source: Drug FDA label -- Warnings section]",
                "severity": "informational",
                "drugs_involved": ["drugA"],
            },
            {
                "text": "Fatal boxed warning. [Source: Warfarin FDA label -- Boxed Warning]",
                "severity": "critical",
                "drugs_involved": ["warfarin"],
            },
        ]
        result = process_flags(flags)
        self.assertGreater(
            result[0]["confidence_score"],
            result[1]["confidence_score"]
        )

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(process_flags([]), [])

    def test_confidence_score_added_to_output(self):
        flags = [{
            "text": "Risk noted. [Source: Warfarin FDA label -- Drug Interactions section]",
            "severity": "moderate",
            "drugs_involved": ["warfarin"],
        }]
        result = process_flags(flags)
        self.assertIn("confidence_score", result[0])

    def test_all_uncited_flags_returns_empty(self):
        flags = [
            {"text": "No citation.", "severity": "moderate", "drugs_involved": []},
            {"text": "Also no citation.", "severity": "critical", "drugs_involved": []},
        ]
        self.assertEqual(process_flags(flags), [])


class TestGetSeverityEmoji(unittest.TestCase):

    def test_critical_emoji(self):
        self.assertEqual(get_severity_emoji("critical"), "⚠️ CRITICAL")

    def test_moderate_emoji(self):
        self.assertEqual(get_severity_emoji("moderate"), "🔶 MODERATE")

    def test_informational_emoji(self):
        self.assertEqual(get_severity_emoji("informational"), "ℹ️ INFORMATIONAL")

    def test_unknown_defaults_to_informational(self):
        self.assertEqual(get_severity_emoji("unknown"), "ℹ️ INFORMATIONAL")

    def test_case_insensitive(self):
        self.assertEqual(get_severity_emoji("CRITICAL"), "⚠️ CRITICAL")


if __name__ == "__main__":
    unittest.main(verbosity=2)