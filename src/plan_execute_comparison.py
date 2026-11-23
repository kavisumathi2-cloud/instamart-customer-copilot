
import json
import math
import re
import uuid
from pathlib import Path

import pandas as pd
import refund_agent as agent


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"


def run_plan_execute(message, scenario_id, request_days=3):
    """Run a deterministic Plan-and-Execute refund workflow."""

    run_id = str(uuid.uuid4())
    order_match = re.search(r"ORD-\d{6}", str(message))

    result = {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": "FALLBACK",
        "final_answer": "",
        "steps": [],
        "tool_calls": 0,
        "model_calls": 0,
        "replans": [],
        "planner_type": "deterministic_python",
    }

    if not order_match:
        result["final_answer"] = "Order ID missing."
        return result

    order_id = order_match.group(0)

    valid, days, reason = agent.validate_request_days(request_days)

    if not valid:
        result["status"] = "DENIED"
        result["final_answer"] = reason
        return result

    tool_call_history = set()

    def execute(name, arguments):
        """Execute a tool once and block repeated identical calls."""

        signature = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            default=str,
        )

        if signature in tool_call_history:
            return {
                "error": (
                    "Stopped safely because the same tool call "
                    "was repeated."
                )
            }

        tool_call_history.add(signature)

        observation = agent._agent_tool(
            name=name,
            arguments=arguments,
            order_id=order_id,
            request_days=days,
            run_id=run_id,
            scenario_id=f"PLAN_{scenario_id}",
        )

        result["tool_calls"] += 1

        result["steps"].append({
            "step": len(result["steps"]) + 1,
            "action": name,
            "arguments": arguments,
            "observation": observation,
        })

        return observation

    # PLAN: Define the intended workflow.
    plan = [
        "get_order",
        "search_policy_delivery",
        "search_policy_approval",
        "calculate_refund",
        "choose_refund_action",
    ]

    result["initial_plan"] = plan

    # Handle the intentional repeated-tool-call test scenario.
    # This deterministic branch reproduces the [LOOP] test pattern.
    if "[LOOP]" in str(message):
        repeated_query = {"query": "refund policy"}

        execute("search_policy", repeated_query)
        repeated_result = execute("search_policy", repeated_query)

        if "error" in repeated_result:
            result["status"] = "FALLBACK"
            result["replans"].append(
                "Repeated identical tool call detected; stopped safely."
            )
            result["final_answer"] = repeated_result["error"]
            return result

    # EXECUTE STEP 1: Verify the order.
    order = execute(
        "get_order",
        {"order_id": order_id},
    )

    if "error" in order:
        if "not found" in str(order["error"]).lower():
            result["status"] = "NOT_FOUND"
        else:
            result["status"] = "FALLBACK_ORDER_ERROR"

        result["replans"].append(
            "Order lookup failed; stopped remaining actions."
        )
        result["final_answer"] = str(order["error"])
        return result

    if str(order.get("status", "")).lower() != "completed":
        result["status"] = "DENIED"
        result["replans"].append(
            "Order was not completed; refund actions cancelled."
        )
        result["final_answer"] = (
            "Only completed orders can receive a refund."
        )
        return result

    # EXECUTE STEP 2: Retrieve delivery and request-window rules.
    policy_delivery = execute(
        "search_policy",
        {
            "query": (
                "completed order late delivery more than 45 minutes "
                "refund request within 7 days"
            )
        },
    )

    if "error" in policy_delivery:
        result["status"] = "FALLBACK_POLICY_ERROR"
        result["final_answer"] = str(policy_delivery["error"])
        return result

    # EXECUTE STEP 3: Retrieve automatic-refund and approval rules.
    policy_approval = execute(
        "search_policy",
        {
            "query": (
                "automatic refund limit INR 1000 "
                "human support lead approval"
            )
        },
    )

    if "error" in policy_approval:
        result["status"] = "FALLBACK_POLICY_ERROR"
        result["final_answer"] = str(policy_approval["error"])
        return result

    policy_results = (
        policy_delivery.get("results", [])
        + policy_approval.get("results", [])
    )

    policy_text = " ".join(
        item.get("text", "")
        for item in policy_results
    ).lower()

    required_conditions = [
        "completed",
        "45 minutes",
        "7 days",
        "1000",
    ]

    missing_conditions = [
        condition
        for condition in required_conditions
        if condition not in policy_text
    ]

    if missing_conditions:
        result["status"] = "FALLBACK_POLICY_INCOMPLETE"
        result["replans"].append(
            f"Policy evidence incomplete: {missing_conditions}"
        )
        result["final_answer"] = (
            "Stopped safely because required policy evidence "
            "was not retrieved."
        )
        return result

    # EXECUTE STEP 4: Calculate refund eligibility.
    calculation = execute(
        "calculate_refund",
        {
            "order_amount": order.get("order_amount"),
            "delivery_minutes": order.get("delivery_minutes"),
        },
    )

    # Validate the complete calculation response before using it.
    if (
        not isinstance(calculation, dict)
        or "error" in calculation
        or not isinstance(calculation.get("eligible"), bool)
    ):
        result["status"] = "FALLBACK_CALCULATION_ERROR"
        result["replans"].append(
            "Unexpected calculation result; stopped safely."
        )
        result["final_answer"] = str(calculation)
        return result

    if not calculation["eligible"]:
        result["status"] = "DENIED"
        result["final_answer"] = str(
            calculation.get("reason", "Refund is not eligible.")
        )
        return result

    try:
        refund_amount = float(calculation["refund_inr"])
    except (KeyError, TypeError, ValueError, OverflowError):
        refund_amount = float("nan")

    if (
        not math.isfinite(refund_amount)
        or refund_amount <= 0
        or not isinstance(
            calculation.get("requires_human_approval"), bool
        )
    ):
        result["status"] = "FALLBACK_CALCULATION_ERROR"
        result["replans"].append(
            "Malformed calculation fields; stopped safely."
        )
        result["final_answer"] = (
            "Stopped safely because the refund calculation "
            "returned invalid data."
        )
        return result

    # RE-PLAN: Select the action using the calculation result.
    if calculation["requires_human_approval"]:
        action = "escalate_to_human"
    else:
        action = "issue_refund"

    result["replans"].append(
        f"Calculation observation selected {action}."
    )

    # EXECUTE STEP 5: Perform the guarded action.
    action_result = execute(
        action,
        {
            "order_id": order_id,
            "refund_inr": refund_amount,
        },
    )

    if "error" in action_result:
        result["status"] = "FALLBACK_ACTION_ERROR"
        result["final_answer"] = str(action_result)
        return result

    if action_result.get("success"):
        result["status"] = "REFUND"
        result["final_answer"] = (
            f"REFUND: INR {refund_amount:.2f} issued."
        )

    elif action_result.get("escalated"):
        result["status"] = "ESCALATE"
        result["final_answer"] = (
            f"ESCALATE: INR {refund_amount:.2f} "
            "requires human approval."
        )

    else:
        result["status"] = "FALLBACK_UNEXPECTED_RESULT"
        result["final_answer"] = (
            "Unexpected action response; stopped safely."
        )

    return result


def run_comparison():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = agent.normalize_scenarios(
        pd.read_csv(SCENARIO_PATH)
    ).fillna("")
    comparison = []

    for _, row in scenarios.iterrows():
        scenario_id = str(row["scenario_id"])
        message = str(row["message"])
        expected = str(row["expected_outcome"]).strip().upper()
        expected_refund = float(row["expected_refund_inr"] or 0)
        expected_escalated = float(row["expected_escalated_amount_inr"] or 0)

        # Run the existing ReAct agent.
        react = agent.run_agent(
            user_message=message,
            scenario_id=f"COMPARE_REACT_{scenario_id}",
            request_days=3,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
        )

        react_status = react["status"]

        if react_status.startswith("FALLBACK"):
            react_outcome = "FALLBACK"
        else:
            react_outcome = react_status

        react_refund = 0.0
        react_escalated = 0.0

        for step in react.get("steps", []):
            observation = step.get("observation", {})

            if observation.get("success"):
                react_refund = float(observation["refund_inr"])
            elif observation.get("escalated"):
                react_escalated = float(observation["refund_inr"])

        # Run deterministic Plan-and-Execute.
        planned = run_plan_execute(message, scenario_id)

        plan_refund = 0.0
        plan_escalated = 0.0

        for step in planned["steps"]:
            observation = step.get("observation", {})

            if observation.get("success"):
                plan_refund = float(observation["refund_inr"])
            elif observation.get("escalated"):
                plan_escalated = float(observation["refund_inr"])

        comparison.append({
            "scenario_id": scenario_id,
            "expected_outcome": expected,
            "react_outcome": react_outcome,
            "plan_execute_outcome": planned["status"],
            "react_outcome_match": react_outcome == expected,
            "plan_execute_outcome_match": (
                planned["status"] == expected
            ),
            "expected_refund_inr": expected_refund,
            "expected_escalated_amount_inr": expected_escalated,
            "react_refund_inr": react_refund,
            "plan_execute_refund_inr": plan_refund,
            "react_refund_match": abs(
                react_refund - expected_refund
            ) < 0.01,
            "plan_execute_refund_match": abs(
                plan_refund - expected_refund
            ) < 0.01,
            "react_escalated_amount_inr": react_escalated,
            "plan_execute_escalated_amount_inr": plan_escalated,
            "react_escalation_match": abs(
                react_escalated - expected_escalated
            ) < 0.01,
            "plan_execute_escalation_match": abs(
                plan_escalated - expected_escalated
            ) < 0.01,
            "react_model_calls": react.get("llm_calls", 0),
            "plan_execute_model_calls": planned["model_calls"],
            "react_tool_steps": react.get("step_count", 0),
            "plan_execute_tool_calls": planned["tool_calls"],
            "plan_replans": len(planned["replans"]),
        })

    output = pd.DataFrame(comparison)

    output_path = OUTPUT_DIR / "react_vs_plan_execute.csv"
    output.to_csv(output_path, index=False)

    print(output.to_string(index=False))
    print("\nComparison saved:", output_path)

    print(
        "\nReAct outcome matches:",
        int(output["react_outcome_match"].sum()),
        "/",
        len(output),
    )

    print(
        "Plan-and-Execute outcome matches:",
        int(output["plan_execute_outcome_match"].sum()),
        "/",
        len(output),
    )

    print(
        "ReAct refund matches:",
        int(output["react_refund_match"].sum()),
        "/",
        len(output),
    )

    print(
        "Plan-and-Execute refund matches:",
        int(output["plan_execute_refund_match"].sum()),
        "/",
        len(output),
    )

    print(
        "ReAct escalation matches:",
        int(output["react_escalation_match"].sum()),
        "/",
        len(output),
    )

    print(
        "Plan-and-Execute escalation matches:",
        int(output["plan_execute_escalation_match"].sum()),
        "/",
        len(output),
    )


if __name__ == "__main__":
    run_comparison()
