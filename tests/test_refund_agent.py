import unittest
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from src.refund_agent import (
    validate_request_days,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    run_refund_agent,
    resume_agent,
    record_refund_ledger,
    _new_state,
    _continue_agent,
    _scenario_ledger_matches,
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



class TestHumanEscalation(unittest.TestCase):

    def test_valid_high_value_refund_is_escalated(self):
        result = escalate_to_human(
            "ORD-107652",
            2598.0,
            request_days=3,
        )

        self.assertTrue(result["escalated"])
        self.assertEqual(result["order_id"], "ORD-107652")
        self.assertEqual(result["refund_inr"], 2598.0)
        self.assertFalse(result["ledger_recorded"])
        self.assertIn("review", result["message"].lower())

    def test_incorrect_refund_amount_is_rejected(self):
        result = escalate_to_human(
            "ORD-107652",
            2000.0,
            request_days=3,
        )

        self.assertIn("error", result)
        self.assertIn("verification", result["error"].lower())

    def test_eight_day_escalation_request_is_rejected(self):
        result = escalate_to_human(
            "ORD-107652",
            2598.0,
            request_days=8,
        )

        self.assertIn("error", result)
        self.assertIn("7 days", result["error"])

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

    @patch("src.refund_agent._continue_agent")
    @patch("src.refund_agent.load_json")
    def test_resume_in_progress_checkpoint_continues_saved_state(
        self,
        mock_load_json,
        mock_continue_agent,
    ):
        saved_state = {
            "run_id": "test-in-progress-run",
            "scenario_id": "TEST_RESUME",
            "order_id": "ORD-100048",
            "status": "in_progress",
            "request_days": 3,
            "steps": [
                {
                    "step": 1,
                    "action": "get_order",
                    "observation": {"success": True},
                },
                {
                    "step": 2,
                    "action": "search_policy",
                    "observation": {"success": True},
                },
            ],
            "llm_calls": 2,
            "tokens_used": 300,
        }

        expected_result = {
            "run_id": "test-in-progress-run",
            "status": "REFUND",
            "order_id": "ORD-100048",
            "steps": saved_state["steps"],
            "step_count": 2,
            "llm_calls": 3,
        }

        mock_load_json.return_value = saved_state
        mock_continue_agent.return_value = expected_result

        result = resume_agent("test-in-progress-run")

        mock_continue_agent.assert_called_once_with(saved_state)
        self.assertEqual(result["status"], "REFUND")

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



class TestRefundLedger(unittest.TestCase):

    def test_duplicate_event_is_recorded_only_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ledger_path = Path(temp_dir) / "refund_ledger.jsonl"

            with patch(
                "src.refund_agent.REFUND_LEDGER_PATH",
                ledger_path,
            ):
                first = record_refund_ledger(
                    "issue_refund",
                    "test-run-001",
                    "TEST_IDEMPOTENCY",
                    "ORD-110603",
                    58,
                    "simulated_refund_issued",
                )
                second = record_refund_ledger(
                    "issue_refund",
                    "test-run-001",
                    "TEST_IDEMPOTENCY",
                    "ORD-110603",
                    58,
                    "simulated_refund_issued",
                )

            records = [
                json.loads(line)
                for line in ledger_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]

            self.assertEqual(len(records), 1)
            self.assertEqual(first["event_id"], second["event_id"])
            self.assertEqual(records[0]["event_id"], first["event_id"])
            self.assertFalse(records[0]["real_money_transferred"])


class TestRefundAgentBudgetSafeguards(unittest.TestCase):

    def run_isolated(self, state):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with (
                patch("src.refund_agent.STATE_DIR", root / "state"),
                patch("src.refund_agent.OUTPUT_DIR", root / "outputs"),
                patch(
                    "src.refund_agent.AGENT_RUNS_PATH",
                    root / "agent_runs.jsonl",
                ),
            ):
                return _continue_agent(state)

    def make_state(self, token_limit=6000, time_limit=10):
        return _new_state(
            "Refund request for ORD-110603",
            "TEST_BUDGET",
            3,
            6,
            token_limit,
            time_limit,
            "test-budget-run",
        )

    def test_zero_time_limit_stops_before_llm_call(self):
        state = self.make_state(time_limit=0)

        with patch("src.refund_agent.mock_llm.chat") as mock_chat:
            result = self.run_isolated(state)

        self.assertEqual(result["status"], "FALLBACK_TIMEOUT")
        self.assertEqual(result["llm_calls"], 0)
        mock_chat.assert_not_called()

    def test_zero_token_limit_stops_before_llm_call(self):
        state = self.make_state(token_limit=0)

        with patch("src.refund_agent.mock_llm.chat") as mock_chat:
            result = self.run_isolated(state)

        self.assertEqual(result["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(result["llm_calls"], 0)
        mock_chat.assert_not_called()

    def test_prompt_estimate_at_limit_stops_before_llm_call(self):
        state = self.make_state(token_limit=100)

        with (
            patch("src.refund_agent._token_estimate", return_value=100),
            patch("src.refund_agent.mock_llm.chat") as mock_chat,
        ):
            result = self.run_isolated(state)

        self.assertEqual(result["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(result["llm_calls"], 0)
        mock_chat.assert_not_called()

    def test_actual_response_token_overflow_stops_before_tool(self):
        state = self.make_state(token_limit=100)

        response = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=101)
        )

        with (
            patch("src.refund_agent._token_estimate", return_value=1),
            patch(
                "src.refund_agent.mock_llm.chat",
                return_value=response,
            ) as mock_chat,
            patch("src.refund_agent._agent_tool") as mock_tool,
        ):
            result = self.run_isolated(state)

        self.assertEqual(result["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(result["tokens_used"], 101)
        mock_chat.assert_called_once()
        mock_tool.assert_not_called()

    def test_elapsed_time_limit_stops_before_llm_call(self):
        state = self.make_state(time_limit=5)

        with (
            patch(
                "src.refund_agent.time.monotonic",
                side_effect=[100.0, 105.0],
            ),
            patch("src.refund_agent.mock_llm.chat") as mock_chat,
        ):
            result = self.run_isolated(state)

        self.assertEqual(result["status"], "FALLBACK_TIMEOUT")
        self.assertEqual(result["llm_calls"], 0)
        mock_chat.assert_not_called()

    def test_tool_exception_stops_safely_and_saves_checkpoint(self):
        state = self.make_state()

        tool_call = SimpleNamespace(
            id="tool-call-001",
            name="get_order",
            arguments='{"order_id": "ORD-110603"}',
        )

        message = SimpleNamespace(
            content=None,
            tool_calls=[tool_call],
        )

        response = SimpleNamespace(
            usage=SimpleNamespace(total_tokens=10),
            choices=[SimpleNamespace(message=message)],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with (
                patch("src.refund_agent.STATE_DIR", root / "state"),
                patch("src.refund_agent.OUTPUT_DIR", root / "outputs"),
                patch(
                    "src.refund_agent.AGENT_RUNS_PATH",
                    root / "agent_runs.jsonl",
                ),
                patch(
                    "src.refund_agent.mock_llm.chat",
                    return_value=response,
                ),
                patch(
                    "src.refund_agent._agent_tool",
                    side_effect=RuntimeError("Simulated tool failure"),
                ),
            ):
                result = _continue_agent(state)

                self.assertEqual(result["status"], "FALLBACK_TOOL_ERROR")
                self.assertIn(
                    "Simulated tool failure",
                    result["final_answer"],
                )
                self.assertEqual(result["step_count"], 0)
                self.assertFalse(result["ledger_recorded"])
                self.assertTrue(
                    Path(result["checkpoint_path"]).exists()
                )

if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestScenarioLedgerValidation(unittest.TestCase):

    def make_record(self):
        return {
            "status": "simulated_refund_issued",
            "refund_inr": 58.0,
            "real_money_transferred": False,
            "event_id": "run-001:issue_refund:ORD-110603",
            "timestamp": "2026-11-24T10:00:00+00:00",
            "tool_name": "issue_refund",
        }

    def test_complete_ledger_record_matches(self):
        self.assertTrue(
            _scenario_ledger_matches(
                [self.make_record()],
                "simulated_refund_issued",
                58.0,
            )
        )

    def test_missing_timestamp_does_not_match(self):
        record = self.make_record()
        record.pop("timestamp")

        self.assertFalse(
            _scenario_ledger_matches(
                [record],
                "simulated_refund_issued",
                58.0,
            )
        )

    def test_no_ledger_record_matches_denied_scenario(self):
        self.assertTrue(
            _scenario_ledger_matches([], None, 0.0)
        )
