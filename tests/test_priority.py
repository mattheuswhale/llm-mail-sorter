
import unittest

from maildigest.llm import _to_bool
from maildigest.priority import classify_priority
from maildigest.types import EmailMessage, ExtractedSignals


def email(
    subject: str,
    body: str = "",
    *,
    list_unsubscribe: str = "",
    precedence: str = "",
    auto_submitted: str = "",
) -> EmailMessage:
    return EmailMessage(
        id="1",
        thread_id="t1",
        sender="sender@example.com",
        subject=subject,
        date="",
        snippet=body[:120],
        body=body,
        list_unsubscribe=list_unsubscribe,
        precedence=precedence,
        auto_submitted=auto_submitted,
    )


class BoolParsingTests(unittest.TestCase):
    def test_false_string_is_false(self):
        self.assertFalse(_to_bool("false"))

    def test_true_string_is_true(self):
        self.assertTrue(_to_bool("true"))


class PriorityRuleTests(unittest.TestCase):
    def test_sale_is_low_even_if_model_misses_promotion(self):
        msg = email("50% OFF - sale ends tonight", "Shop now and save.")
        signals = ExtractedSignals(
            is_promotion=False,
            asks_for_action=True,  # Tiny model may mistake "shop now" as action.
            has_deadline=True,
            deadline_is_immediate=True,
            category="other",
            summary="A store is advertising a sale.",
        )
        result = classify_priority(msg, signals)
        self.assertEqual(result.priority, "low")
        self.assertEqual(result.rule, "promotion_or_newsletter")

    def test_payment_failure_overrides_newsletter_footer(self):
        msg = email(
            "Payment failed",
            "We could not process your payment. Unsubscribe from marketing here.",
            list_unsubscribe="<mailto:unsubscribe@example.com>",
        )
        signals = ExtractedSignals(
            payment_problem=False,  # Hard rule should catch it.
            is_newsletter=True,
            summary="Payment could not be processed.",
        )
        result = classify_priority(msg, signals)
        self.assertEqual(result.priority, "urgent")
        self.assertEqual(result.rule, "critical_problem")

    def test_immediate_real_action_is_urgent(self):
        msg = email("Approval needed today", "Please approve the contract today.")
        signals = ExtractedSignals(
            asks_for_action=True,
            has_deadline=True,
            deadline_is_immediate=True,
            category="work",
            summary="Approval is requested today.",
        )
        result = classify_priority(msg, signals)
        self.assertEqual(result.priority, "urgent")
        self.assertEqual(result.rule, "immediate_action_deadline")

    def test_regular_action_is_high(self):
        msg = email("Please review the draft", "Can you review this before Friday?")
        signals = ExtractedSignals(
            asks_for_reply=True,
            asks_for_action=True,
            has_deadline=True,
            category="work",
            summary="A draft needs review before Friday.",
        )
        result = classify_priority(msg, signals)
        self.assertEqual(result.priority, "high")
        self.assertEqual(result.rule, "action_required")

    def test_automated_no_action_is_low(self):
        msg = email(
            "Monthly usage report",
            "Your monthly report is ready.",
            auto_submitted="auto-generated",
        )
        signals = ExtractedSignals(
            is_automated=False,
            category="notification",
            summary="Monthly usage report is available.",
        )
        result = classify_priority(msg, signals)
        self.assertEqual(result.priority, "low")
        self.assertEqual(result.rule, "automated_no_action")

    def test_routine_personal_information_is_normal(self):
        msg = email("FYI", "Here is the document we discussed.")
        signals = ExtractedSignals(
            from_person=True,
            category="personal",
            summary="A person shared a document for information.",
        )
        result = classify_priority(msg, signals)
        self.assertEqual(result.priority, "normal")
        self.assertEqual(result.rule, "routine_information")


if __name__ == "__main__":
    unittest.main()
