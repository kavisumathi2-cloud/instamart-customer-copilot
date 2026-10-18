
import unittest

from src.refund_agent import (
    validate_request_days,
    calculate_refund,
    issue_refund,
    run_refund_agent,
)


class TestRefundRequestDays(unittest.TestCase):

    def test_zero_days_is_allowed(self):
        valid, days, reason = validate_request_days(0)
        self.assertTrue(valid)
        self.assertEqual(days, 0.0)

    def test_seven_days_is_allowed(self):
        valid, days, reason = validate_request_days(7)
        self.assertTrue(valid)
        self.assertEqual(days, 7.0)

    def test_eight_days_is_rejected(self):
        valid, days, reason = validate_request_days(8)
        self.assertFalse(valid)
        self.assertIn("7 days", reason)

    def test_negative_days_is_rejected(self):
        valid, days, reason = validate_request_days(-1)
        self.assertFalse(valid)
        self.assertIn("Invalid", reason)

    def test_nan_is_rejected(self):
        valid, days, reason = validate_request_days(float("nan"))
        self.assertFalse(valid)
        self.assertIn("Invalid", reason)


class TestRefundSafetyRules(unittest.TestCase):

    def test_delivery_at_45_minutes_is_not_eligible(self):
        result = calculate_refund(58, 45)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["refund_inr"], 0)

    def test_delivery_over_45_minutes_is_eligible(self):
        result = calculate_refund(58, 46)

        self.assertTrue(result["eligible"])
        self.assertEqual(result["refund_inr"], 58)

    def test_zero_order_amount_is_rejected(self):
        result = calculate_refund(0, 70)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["refund_inr"], 0)

    def test_negative_delivery_time_is_rejected(self):
        result = calculate_refund(58, -1)

        self.assertFalse(result["eligible"])
        self.assertEqual(result["refund_inr"], 0)

    def test_refund_requires_approval(self):
        result = issue_refund(
            "ORD-110603",
            58,
            approved=False,
            request_days=3,
        )

        self.assertIn("error", result)
        self.assertIn("approval", result["error"].lower())

    def test_invalid_order_id_is_rejected(self):
        result = issue_refund(
            "BAD-ID",
            58,
            approved=True,
            request_days=3,
        )

        self.assertIn("error", result)

    def test_eight_day_request_uses_no_llm_calls(self):
        result = run_refund_agent(
            "ORD-110603",
            request_days=8,
            resume=False,
        )

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["step_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
