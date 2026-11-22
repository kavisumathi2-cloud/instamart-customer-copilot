"""Simulated crash after the second tool checkpoint, then resume.

The first run is interrupted when the third tool call (calculate_refund)
is about to execute. The run must then be resumed from its checkpoint
without repeating get_order or search_policy.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.refund_agent as agent


class SimulatedCrash(BaseException):
    """Raised to imitate a process crash (not caught by `except Exception`)."""


class TestCrashAndResume(unittest.TestCase):

    def test_crash_after_second_tool_then_resume(self):
        original_tool = agent._agent_tool

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "outputs"
            state_dir = tmp_path / "state"
            output_dir.mkdir()
            state_dir.mkdir()

            with patch.object(agent, "OUTPUT_DIR", output_dir), \
                 patch.object(agent, "STATE_DIR", state_dir), \
                 patch.object(
                     agent,
                     "AGENT_RUNS_PATH",
                     output_dir / "agent_runs.jsonl",
                 ), \
                 patch.object(
                     agent,
                     "REFUND_LEDGER_PATH",
                     output_dir / "refund_ledger.jsonl",
                 ):

                # ---- Run 1: crash before the third tool executes ----
                first_run_tools = []

                def crashing_tool(name, *args, **kwargs):
                    if len(first_run_tools) == 2:
                        raise SimulatedCrash()
                    first_run_tools.append(name)
                    return original_tool(name, *args, **kwargs)

                with patch.object(agent, "_agent_tool", crashing_tool):
                    with self.assertRaises(SimulatedCrash):
                        agent.run_agent(
                            user_message=(
                                "Process a late-delivery refund for "
                                "ORD-110603. Request age is 3 days. "
                                "Verify using tools."
                            ),
                            scenario_id="CRASH_RESUME_TEST",
                            request_days=3,
                        )

                self.assertEqual(
                    first_run_tools,
                    ["get_order", "search_policy"],
                )

                checkpoints = list(state_dir.glob("refund_agent_*.json"))
                self.assertEqual(len(checkpoints), 1)

                saved = json.loads(
                    checkpoints[0].read_text(encoding="utf-8")
                )
                run_id = saved["run_id"]

                self.assertEqual(saved["status"], "in_progress")
                self.assertEqual(
                    [step["action"] for step in saved["steps"]],
                    ["get_order", "search_policy"],
                )

                # ---- Run 2: resume from the checkpoint ----
                resumed_tools = []

                def recording_tool(name, *args, **kwargs):
                    resumed_tools.append(name)
                    return original_tool(name, *args, **kwargs)

                with patch.object(agent, "_agent_tool", recording_tool):
                    result = agent.resume_agent(run_id)

                # The two completed tools are not executed again.
                self.assertEqual(
                    resumed_tools,
                    ["calculate_refund", "issue_refund"],
                )

                self.assertEqual(result["status"], "REFUND")
                self.assertEqual(
                    [step["action"] for step in result["steps"]],
                    [
                        "get_order",
                        "search_policy",
                        "calculate_refund",
                        "issue_refund",
                    ],
                )

                ledger_path = output_dir / "refund_ledger.jsonl"
                records = [
                    json.loads(line)
                    for line in ledger_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                refunds = [
                    r for r in records
                    if r.get("tool_name") == "issue_refund"
                ]
                self.assertEqual(len(refunds), 1)


if __name__ == "__main__":
    unittest.main()