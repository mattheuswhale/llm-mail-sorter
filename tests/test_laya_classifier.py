from __future__ import annotations

import unittest

from maildigest.laya_classifier import apply_laya_answers
from maildigest.types import ExtractedSignals


def _answers(**overrides):
    values = {
        "is_promotion": 0.02,
        "is_newsletter": 0.03,
        "is_automated": 0.15,
        "from_person": 0.91,
        "asks_for_reply": 0.82,
        "asks_for_action": 0.77,
        "has_deadline": 0.71,
        "deadline_is_immediate": 0.20,
        "security_issue": 0.01,
        "payment_problem": 0.04,
        "account_problem": 0.03,
        "service_outage": 0.01,
    }
    values.update(overrides)
    result = {key: {"noul": value} for key, value in values.items()}
    result["category"] = {"choice": "work"}
    return result


class LayaAnswerTests(unittest.TestCase):
    def test_laya_overrides_structured_signals_but_preserves_free_text(self):
        base = ExtractedSignals(
            is_promotion=True,
            asks_for_reply=False,
            category="promotion",
            deadline="Friday 17:00",
            summary="Please review the attached proposal by Friday.",
        )

        refined = apply_laya_answers(base, _answers())

        self.assertFalse(refined.is_promotion)
        self.assertTrue(refined.from_person)
        self.assertTrue(refined.asks_for_reply)
        self.assertTrue(refined.asks_for_action)
        self.assertEqual(refined.category, "work")
        self.assertEqual(refined.deadline, "Friday 17:00")
        self.assertEqual(
            refined.summary,
            "Please review the attached proposal by Friday.",
        )

    def test_probability_threshold_is_half(self):
        base = ExtractedSignals()
        refined = apply_laya_answers(
            base,
            _answers(is_promotion=0.5, asks_for_reply=0.4999),
        )
        self.assertTrue(refined.is_promotion)
        self.assertFalse(refined.asks_for_reply)

    def test_missing_typed_answer_fails_closed_to_caller(self):
        base = ExtractedSignals()
        answers = _answers()
        del answers["security_issue"]
        with self.assertRaises(ValueError):
            apply_laya_answers(base, answers)

    def test_invalid_category_is_rejected(self):
        base = ExtractedSignals()
        answers = _answers()
        answers["category"] = {"choice": "urgent"}
        with self.assertRaises(ValueError):
            apply_laya_answers(base, answers)


if __name__ == "__main__":
    unittest.main()
