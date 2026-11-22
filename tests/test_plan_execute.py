import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import plan_execute_comparison as pe


class TestPlanExecuteSafety(unittest.TestCase):

    def setUp(self):
        self.order = {
            "status": "completed",
            "order_amount": 150.0,
            "delivery_minutes": 70.0,
        }
        self.policy = {
            "results": [{
                "text": (
                    "completed orders late delivery more than "
                    "45 minutes refund request within 7 days "
                    "automatic refund limit INR 1000 "
                    "human support lead approval"
                )
            }]
        }

    def successful_tool(self, name, arguments, **kwargs):
        if name == "get_order":
            return self.order
        if name == "search_policy":
            return self.policy
        if name == "calculate_refund":
            return {
                "eligible": True,
                "refund_inr": 150.0,
                "requires_human_approval": False,
                "reason": "Eligible",
            }
        if name == "issue_refund":
            return {"success": True}
        if name == "escalate_to_human":
            return {"escalated": True}
        return {"error": "Unexpected tool"}

    def test_missing_order_id_stops_without_tool_calls(self):
        with patch.object(pe.agent, "_agent_tool") as tool:
            result = pe.run_plan_execute(
                "Please refund my order", "TEST_MISSING"
            )
        self.assertEqual(result["status"], "FALLBACK")
        self.assertEqual(result["tool_calls"], 0)
        tool.assert_not_called()

    def test_invalid_request_days_stops_without_tool_calls(self):
        with patch.object(pe.agent, "_agent_tool") as tool:
            result = pe.run_plan_execute(
                "Refund order ORD-110603",
                "TEST_INVALID_DAYS",
                request_days=8,
            )
        self.assertEqual(result["status"], "DENIED")
        self.assertEqual(result["tool_calls"], 0)
        tool.assert_not_called()

    def test_order_not_found_stops_workflow(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "get_order":
                return {"error": "Order not found"}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603", "TEST_NOT_FOUND"
            )
        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertEqual(result["tool_calls"], 1)

    def test_cancelled_order_is_denied(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "get_order":
                return {"status": "cancelled"}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603", "TEST_CANCELLED"
            )
        self.assertEqual(result["status"], "DENIED")
        self.assertEqual(result["tool_calls"], 1)

    def test_policy_retrieval_error_stops_workflow(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "get_order":
                return self.order
            if name == "search_policy":
                return {"error": "Policy retrieval failed"}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603", "TEST_POLICY_ERROR"
            )
        self.assertEqual(result["status"], "FALLBACK_POLICY_ERROR")
        self.assertEqual(result["tool_calls"], 2)

    def test_incomplete_policy_evidence_stops_workflow(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "get_order":
                return self.order
            if name == "search_policy":
                return {"results": [{"text": "completed orders only"}]}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603", "TEST_POLICY_INCOMPLETE"
            )
        self.assertEqual(
            result["status"], "FALLBACK_POLICY_INCOMPLETE"
        )
        self.assertEqual(result["tool_calls"], 3)

    def test_calculation_error_stops_workflow(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "get_order":
                return self.order
            if name == "search_policy":
                return self.policy
            if name == "calculate_refund":
                return {"error": "Calculation failed"}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603", "TEST_CALCULATION_ERROR"
            )
        self.assertEqual(
            result["status"], "FALLBACK_CALCULATION_ERROR"
        )
        self.assertEqual(result["tool_calls"], 4)

    def test_malformed_calculation_result_does_not_crash(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "get_order":
                return self.order
            if name == "search_policy":
                return self.policy
            if name == "calculate_refund":
                return {"eligible": True}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603",
                "TEST_MALFORMED_CALCULATION",
            )
        self.assertEqual(
            result["status"], "FALLBACK_CALCULATION_ERROR"
        )
        self.assertEqual(result["tool_calls"], 4)

    def test_duplicate_tool_call_is_blocked(self):
        with patch.object(
            pe.agent, "_agent_tool", return_value=self.policy
        ) as tool:
            result = pe.run_plan_execute(
                "Please refund order ORD-110603 [LOOP]",
                "TEST_DUPLICATE_TOOL",
            )
        self.assertEqual(result["status"], "FALLBACK")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(tool.call_count, 1)

    def test_unexpected_action_response_uses_fallback(self):
        def mock_tool(name, arguments, **kwargs):
            if name == "issue_refund":
                return {"success": False}
            return self.successful_tool(name, arguments, **kwargs)

        with patch.object(pe.agent, "_agent_tool", side_effect=mock_tool):
            result = pe.run_plan_execute(
                "Refund order ORD-110603",
                "TEST_UNEXPECTED_ACTION",
            )
        self.assertEqual(
            result["status"], "FALLBACK_UNEXPECTED_RESULT"
        )


if __name__ == "__main__":
    unittest.main()
