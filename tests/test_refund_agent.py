import unittest
from unittest.mock import patch

from src.refund_agent import (
    validate_request_days,
    calculate_refund,
    issue_refund,
    run_refund_agent,
    resume_agent,
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


class TestRefundAgentResume(unittest.TestCase):

    @patch("src.refund_agent._continue_agent")
    @patch("src.refund_agent.load_json")
    def test_resume_completed_run_does_not_execute_tools_again(
        self,
        mock_load_json,
        mock_continue_agent,
    ):
        completed_state = {
            "run_id": "test-completed-run",
            "scenario_id": "TEST_RESUME",
            "order_id": "ORD-110603",
            "status": "REFUND",
            "final_answer": "REFUND: INR 58.0 issued.",
            "steps": [
                {
                    "step": 1,
                    "action": "issue_refund",
                    "observation": {
                        "success": True,
                        "refund_inr": 58.0,
                    },
                }
            ],
            "llm_calls": 5,
            "tokens_used": 950,
        }

        mock_load_json.return_value = completed_state

        result = resume_agent("test-completed-run")

        self.assertEqual(result["status"], "REFUND")
        self.assertEqual(result["order_id"], "ORD-110603")
        self.assertEqual(result["step_count"], 1)
        self.assertEqual(result["llm_calls"], 5)

        # A completed run should return its saved result.
        mock_continue_agent.assert_not_called()

    @patch("src.refund_agent.load_json")
    def test_resume_missing_checkpoint_returns_error(
        self,
        mock_load_json,
    ):
        mock_load_json.return_value = None

        result = resume_agent("missing-test-run")

        self.assertEqual(result["status"], "ERROR")
        self.assertIn("Checkpoint not found", result["final_answer"])
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
