
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

OUTPUT_DIR = ROOT / "outputs"
POLICY_DIR = ROOT / "data" / "policies"
STATE_DIR = ROOT / "state"

RESULT_PATH = OUTPUT_DIR / "refund_agent_demo.json"
CHECKPOINT_PATH = OUTPUT_DIR / "refund_agent_checkpoint.json"
AGENT_RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"
REFUND_LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"

MAX_STEPS = 6
MAX_LLM_CALLS = 20
AUTO_REFUND_LIMIT = 1000.0
REFUND_REQUEST_WINDOW_DAYS = 7
LATE_DELIVERY_LIMIT_MINUTES = 45.0


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def append_jsonl(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, default=str) + "\n")


def validate_request_days(request_days):
    try:
        days = float(request_days)
    except (TypeError, ValueError):
        return False, None, "Invalid request age."

    if not math.isfinite(days) or days < 0:
        return False, days, "Invalid request age."

    if days > REFUND_REQUEST_WINDOW_DAYS:
        return False, days, (
            "Refund request must be raised within 7 days of delivery."
        )

    return True, days, "Request is within the allowed time window."


def get_order(order_id):
    candidates = [
        ROOT / "data" / "clean" / "orders.csv",
        ROOT / "data" / "cleaned" / "orders.csv",
        ROOT / "data" / "raw" / "orders.csv",
    ]

    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {"error": "Orders CSV not found."}

    try:
        orders = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        return {"error": f"Could not read orders CSV: {exc}"}

    required = {
        "order_id", "status", "order_amount", "delivery_minutes"
    }
    if not required.issubset(orders.columns):
        return {"error": "Required order columns are missing."}

    matches = orders[orders["order_id"].astype(str) == str(order_id)]
    if matches.empty:
        return {"error": "Order not found."}

    row = matches.iloc[0].to_dict()
    result = {}

    for key, value in row.items():
        if pd.isna(value):
            result[key] = None
        elif hasattr(value, "item"):
            result[key] = value.item()
        else:
            result[key] = value

    return result


def search_policy(query):
    if not POLICY_DIR.exists():
        return {"results": []}

    query_words = set(re.findall(r"[a-z]+", str(query).lower()))
    results = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+", content.replace("\n", " ")
        )

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            words = set(re.findall(r"[a-z]+", sentence.lower()))
            overlap = len(query_words & words)

            if overlap:
                results.append({
                    "source": path.name,
                    "text": sentence,
                    "overlap": overlap,
                })

    results.sort(key=lambda item: item["overlap"], reverse=True)
    return {"results": results[:5]}


def calculate_refund(order_amount, delivery_minutes, request_days=3):
    valid, days, reason = validate_request_days(request_days)

    if not valid:
        return {
            "refund_inr": 0,
            "eligible": False,
            "requires_human_approval": False,
            "reason": reason,
        }

    try:
        amount = float(order_amount)
        minutes = float(delivery_minutes)
    except (TypeError, ValueError):
        return {
            "refund_inr": 0,
            "eligible": False,
            "requires_human_approval": False,
            "reason": "Invalid amount or delivery time.",
        }

    if (
        not math.isfinite(amount)
        or not math.isfinite(minutes)
        or amount <= 0
        or minutes < 0
    ):
        return {
            "refund_inr": 0,
            "eligible": False,
            "requires_human_approval": False,
            "reason": "Invalid order details.",
        }

    if minutes <= LATE_DELIVERY_LIMIT_MINUTES:
        return {
            "refund_inr": 0,
            "eligible": False,
            "requires_human_approval": False,
            "reason": "Delivery took 45 minutes or less.",
        }

    return {
        "refund_inr": round(amount, 2),
        "eligible": True,
        "requires_human_approval": amount > AUTO_REFUND_LIMIT,
        "reason": (
            "Refund exceeds INR 1000 automatic limit."
            if amount > AUTO_REFUND_LIMIT
            else "Late-delivery refund conditions satisfied."
        ),
    }


def issue_refund(order_id, refund_inr, approved=False, request_days=3):
    valid, days, reason = validate_request_days(request_days)

    if not valid:
        return {"error": reason}

    if not re.fullmatch(r"ORD-\d{6}", str(order_id)):
        return {"error": "Invalid order ID format."}

    if not approved:
        return {"error": "Refund approval is required."}

    try:
        amount = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "Invalid refund amount."}

    if (
        not math.isfinite(amount)
        or amount <= 0
        or amount > AUTO_REFUND_LIMIT
    ):
        return {
            "error": "Automatic refund must be above zero and <= INR 1000."
        }

    order = get_order(order_id)
    if "error" in order:
        return {"error": "Order verification failed."}

    if str(order.get("status", "")).lower() != "completed":
        return {"error": "Only completed orders can be refunded."}

    eligibility = calculate_refund(
        order.get("order_amount"),
        order.get("delivery_minutes"),
        request_days=days,
    )

    if not eligibility["eligible"]:
        return {"error": eligibility["reason"]}

    if eligibility["requires_human_approval"]:
        return {"error": "Human approval is required."}

    if not math.isclose(
        amount,
        float(eligibility["refund_inr"]),
        rel_tol=0,
        abs_tol=0.001,
    ):
        return {"error": "Refund amount failed verification."}

    return {
        "success": True,
        "order_id": str(order_id),
        "refund_inr": round(amount, 2),
        "message": (
            "Simulated refund recorded. No real money was transferred."
        ),
    }


def escalate_to_human(order_id, refund_inr):
    try:
        amount = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "Invalid refund amount."}

    if not math.isfinite(amount) or amount <= 0:
        return {"error": "Invalid refund amount."}

    return {
        "escalated": True,
        "order_id": str(order_id),
        "refund_inr": round(amount, 2),
        "message": "Sent to human support lead for review.",
    }


def execute_tool(name, arguments, request_days=3):
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be an object."}

    valid, days, reason = validate_request_days(request_days)
    if not valid and name in {"calculate_refund", "issue_refund"}:
        return {"error": reason, "refund_inr": 0}

    if name == "get_order":
        order_id = str(arguments.get("order_id", ""))

        if not re.fullmatch(r"ORD-\d{6}", order_id):
            return {"error": "Invalid order ID format."}

        return get_order(order_id)

    if name == "search_policy":
        return search_policy(
            str(arguments.get("query", "refund policy"))
        )

    if name == "calculate_refund":
        return calculate_refund(
            arguments.get("order_amount"),
            arguments.get("delivery_minutes"),
            request_days=days,
        )

    if name == "issue_refund":
        return issue_refund(
            arguments.get("order_id", ""),
            arguments.get("refund_inr", 0),
            approved=False,
            request_days=days,
        )

    if name == "escalate_to_human":
        return escalate_to_human(
            arguments.get("order_id", ""),
            arguments.get("refund_inr", 0),
        )

    return {"error": f"Tool not allowed: {name}"}


def make_initial_messages(order_id, request_days):
    return [
        {
            "role": "system",
            "content": (
                "AGENT: You are an Instamart refund support agent. "
                "Verify orders and policies. Only completed orders qualify. "
                "Late-delivery refunds require delivery over 45 minutes. "
                "Requests must be raised within 7 days. Refunds above "
                "INR 1000 require human approval. All refunds are simulated."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Process a late-delivery refund for {order_id}. "
                f"Request age is {request_days} days. Verify using tools."
            ),
        },
    ]


def _agent_tool(name, arguments, order_id, request_days):
    """Run an allowed tool and independently verify refund side effects."""
    if name == "issue_refund":
        if str(arguments.get("order_id", "")) != str(order_id):
            return {"error": "Order ID mismatch."}

        order = get_order(order_id)
        if "error" in order:
            return {"error": "Order verification failed."}

        if str(order.get("status", "")).lower() != "completed":
            return {"error": "Only completed orders can be refunded."}

        eligibility = calculate_refund(
            order.get("order_amount"),
            order.get("delivery_minutes"),
            request_days=request_days,
        )

        if not eligibility["eligible"]:
            return {"error": eligibility["reason"]}

        if eligibility["requires_human_approval"]:
            return {
                "error": "Human approval is required.",
                "requires_human_approval": True,
            }

        return issue_refund(
            order_id,
            eligibility["refund_inr"],
            approved=True,
            request_days=request_days,
        )

    if name == "escalate_to_human":
        if str(arguments.get("order_id", "")) != str(order_id):
            return {"error": "Order ID mismatch."}

        order = get_order(order_id)
        if "error" in order:
            return {"error": "Order verification failed."}

        if str(order.get("status", "")).lower() != "completed":
            return {"error": "Only completed orders can be escalated."}

        eligibility = calculate_refund(
            order.get("order_amount"),
            order.get("delivery_minutes"),
            request_days=request_days,
        )

        if not eligibility["eligible"]:
            return {"error": eligibility["reason"]}

        if not eligibility["requires_human_approval"]:
            return {"error": "Human approval is not required for this amount."}

        if not math.isclose(
            float(arguments.get("refund_inr", 0)),
            float(eligibility["refund_inr"]),
            rel_tol=0,
            abs_tol=0.001,
        ):
            return {"error": "Refund amount failed verification."}

        return escalate_to_human(order_id, eligibility["refund_inr"])

    return execute_tool(name, arguments, request_days=request_days)


def _token_estimate(messages):
    total = 0
    for message in messages:
        content = message.get("content") or ""
        total += mock_llm.est_tokens(str(content))
        if message.get("tool_calls"):
            total += mock_llm.est_tokens(
                json.dumps(message["tool_calls"], default=str)
            )
    return total


def _new_state(
    user_message,
    scenario_id,
    request_days,
    max_iterations,
    token_limit,
    time_limit_seconds,
    run_id,
):
    match = re.search(r"ORD-\d{6}", str(user_message))
    order_id = match.group(0) if match else ""

    return {
        "run_id": run_id,
        "scenario_id": str(scenario_id),
        "user_message": str(user_message),
        "order_id": order_id,
        "request_days": request_days,
        "max_iterations": max(0, int(max_iterations)),
        "token_limit": max(0, int(token_limit)),
        "time_limit_seconds": max(0, float(time_limit_seconds)),
        "messages": [
            {
                "role": "system",
                "content": (
                    "AGENT: You are an Instamart refund support agent. "
                    "Verify orders and policies. Only completed orders "
                    "qualify. Late-delivery refunds require delivery over "
                    "45 minutes. Requests must be raised within 7 days. "
                    "Refunds above INR 1000 require human approval. "
                    "All refunds are simulated."
                ),
            },
            {"role": "user", "content": str(user_message)},
        ],
        "steps": [],
        "llm_calls": 0,
        "tokens_used": 0,
        "iterations": 0,
        "status": "in_progress",
        "final_answer": "",
        "ledger_recorded": False,
        "run_logged": False,
    }


def _checkpoint(state):
    path = STATE_DIR / f"refund_agent_{state['run_id']}.json"
    save_json(path, state)
    state["checkpoint_path"] = str(path)
    return path


def _finish(state, status, answer):
    state["status"] = status
    state["final_answer"] = str(answer)

    result = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "order_id": state["order_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "steps": state["steps"],
        "step_count": len(state["steps"]),
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "request_days": state["request_days"],
        "checkpoint_path": str(
            STATE_DIR / f"refund_agent_{state['run_id']}.json"
        ),
    }

    if status == "REFUND" and not state.get("ledger_recorded"):
        refund_step = next(
            (
                step for step in reversed(state["steps"])
                if step["action"] == "issue_refund"
                and step["observation"].get("success")
            ),
            None,
        )

        if refund_step:
            append_jsonl(REFUND_LEDGER_PATH, {
                "run_id": state["run_id"],
                "scenario_id": state["scenario_id"],
                "order_id": state["order_id"],
                "refund_inr": refund_step["observation"]["refund_inr"],
                "status": "simulated_refund_issued",
                "real_money_transferred": False,
            })
            state["ledger_recorded"] = True

    if not state.get("run_logged"):
        append_jsonl(AGENT_RUNS_PATH, result)
        state["run_logged"] = True

    save_json(STATE_DIR / f"refund_agent_{state['run_id']}.json", state)
    save_json(OUTPUT_DIR / f"refund_agent_{state['run_id']}.json", result)
    return result


def _continue_agent(state):
    start_time = time.monotonic()
    max_iterations = state["max_iterations"]

    valid, days, reason = validate_request_days(state["request_days"])
    if not valid:
        return _finish(state, "DENIED", reason)

    if not state["order_id"] or not re.fullmatch(
        r"ORD-\d{6}", state["order_id"]
    ):
        return _finish(state, "NOT_FOUND", "Please provide a valid order ID.")

    while state["iterations"] < max_iterations:
        if state["time_limit_seconds"] == 0:
            return _finish(
                state, "FALLBACK_TIMEOUT",
                "Stopped safely because the time limit is zero.",
            )

        elapsed = time.monotonic() - start_time
        if elapsed >= state["time_limit_seconds"]:
            return _finish(
                state, "FALLBACK_TIMEOUT",
                "Stopped safely because the time limit was reached.",
            )

        if state["tokens_used"] >= state["token_limit"]:
            return _finish(
                state, "FALLBACK_TOKEN_LIMIT",
                "Stopped safely because the token limit was reached.",
            )

        if state["llm_calls"] >= MAX_LLM_CALLS:
            return _finish(
                state, "FALLBACK_MAX_ITERATIONS",
                "Stopped safely because the LLM call limit was reached.",
            )

        estimated_prompt = _token_estimate(state["messages"])
        if estimated_prompt >= state["token_limit"]:
            return _finish(
                state, "FALLBACK_TOKEN_LIMIT",
                "Stopped safely before the next LLM call: token budget exhausted.",
            )

        try:
            response = mock_llm.chat(state["messages"])
            state["llm_calls"] += 1
            state["iterations"] += 1
        except Exception as exc:
            state["status"] = "in_progress"
            state["last_error"] = str(exc)
            save_json(
                STATE_DIR / f"refund_agent_{state['run_id']}.json",
                state,
            )
            return _finish(
                state, "FALLBACK_LLM_ERROR",
                f"LLM call failed safely: {exc}",
            )

        usage = getattr(response, "usage", None)
        call_tokens = int(getattr(usage, "total_tokens", 0) or 0)

        if state["tokens_used"] + call_tokens > state["token_limit"]:
            state["tokens_used"] += call_tokens
            return _finish(
                state, "FALLBACK_TOKEN_LIMIT",
                "Stopped safely because the response exceeded the token budget.",
            )

        state["tokens_used"] += call_tokens
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", []) or []

        if not tool_calls:
            answer = message.content or "No final answer returned."

            if answer.startswith("REFUND:"):
                return _finish(state, "REFUND", answer)
            if answer.startswith("DENIED:"):
                return _finish(state, "DENIED", answer)
            if answer.startswith("NOT_FOUND:"):
                return _finish(state, "NOT_FOUND", answer)
            if answer.startswith("ESCALATE:"):
                return _finish(state, "ESCALATE", answer)

            return _finish(state, "FALLBACK", answer)

        tool_call = tool_calls[0]
        name = tool_call.name

        try:
            arguments = tool_call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object.")
        except (json.JSONDecodeError, TypeError, ValueError):
            return _finish(
                state, "FALLBACK",
                "Stopped safely because tool arguments were invalid.",
            )

        observation = _agent_tool(
            name,
            arguments,
            state["order_id"],
            days,
        )

        step = {
            "step": len(state["steps"]) + 1,
            "action": name,
            "arguments": arguments,
            "observation": observation,
        }
        state["steps"].append(step)

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "name": name,
                "arguments": json.dumps(arguments),
            }],
        })
        state["messages"].append({
            "role": "tool",
            "name": name,
            "content": json.dumps(observation, default=str),
        })

        # Save immediately after each tool execution.
        _checkpoint(state)

        if name == "issue_refund" and observation.get("success"):
            # Do not finish here. The next LLM call must produce the final answer.
            continue

        if observation.get("escalated"):
            continue

    return _finish(
        state,
        "FALLBACK_MAX_ITERATIONS",
        f"Stopped safely after {max_iterations} iterations.",
    )


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    request_days=3,
):
    """Run a new scenario and checkpoint every executed tool."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    state = _new_state(
        user_message=user_message,
        scenario_id=scenario_id,
        request_days=request_days,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        run_id=run_id,
    )
    _checkpoint(state)
    return _continue_agent(state)


def resume_agent(run_id):
    """Resume a saved run without repeating tools already in its checkpoint."""
    path = STATE_DIR / f"refund_agent_{run_id}.json"
    state = load_json(path)

    if not state:
        return {
            "run_id": run_id,
            "status": "ERROR",
            "final_answer": "Checkpoint not found or unreadable.",
            "steps": [],
            "step_count": 0,
            "llm_calls": 0,
        }

    if state.get("status") not in {"in_progress"}:
        return {
            "run_id": state["run_id"],
            "scenario_id": state.get("scenario_id"),
            "order_id": state.get("order_id"),
            "status": state.get("status"),
            "final_answer": state.get("final_answer", ""),
            "steps": state.get("steps", []),
            "step_count": len(state.get("steps", [])),
            "llm_calls": state.get("llm_calls", 0),
            "tokens_used": state.get("tokens_used", 0),
        }

    return _continue_agent(state)


def run_scenarios():
    """Run all rows in data/raw/agent_scenarios.csv."""
    scenario_path = ROOT / "data" / "raw" / "agent_scenarios.csv"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {scenario_path}"
        )

    scenarios = pd.read_csv(scenario_path).fillna("")
    results = []

    for _, row in scenarios.iterrows():
        scenario_id = str(row["scenario_id"])
        message = str(row["message"])
        expected = str(row["expected_outcome"]).strip().upper()

        result = run_agent(
            user_message=message,
            scenario_id=scenario_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10,
        )

        actual = result["status"]
        if actual.startswith("FALLBACK"):
            actual_outcome = "FALLBACK"
        else:
            actual_outcome = actual

        expected_refund = float(row["expected_refund_inr"] or 0)
      
        actual_refund = 0.0

        for step in result.get("steps", []):
            observation = step.get("observation", {})

            if observation.get("success"):
                actual_refund = float(observation["refund_inr"])

            elif observation.get("escalated"):
                # Escalated amount is tracked for comparison only.
                # It is NOT a completed refund.
                actual_refund = float(observation["refund_inr"])


        results.append({
            "scenario_id": scenario_id,
            "expected_outcome": expected,
            "actual_outcome": actual_outcome,
            "outcome_match": actual_outcome == expected,
            "expected_refund_inr": expected_refund,
            "actual_refund_inr": actual_refund,
            "refund_match": math.isclose(
                expected_refund, actual_refund, abs_tol=0.01
            ),
            "run_id": result.get("run_id"),
            "status": actual,
            "llm_calls": result.get("llm_calls", 0),
        })

    output_path = OUTPUT_DIR / "refund_agent_scenarios.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(pd.DataFrame(results).to_string(index=False))
    print("\nScenario results saved:", output_path)
    print("Agent run log:", AGENT_RUNS_PATH)
    print("Refund ledger:", REFUND_LEDGER_PATH)
    return results


def run_refund_agent(order_id, request_days=3, resume=True):
    """Compatibility wrapper for the original demo and existing tests."""
    valid, days, reason = validate_request_days(request_days)

    if not valid:
        result = {
            "order_id": str(order_id),
            "status": "rejected",
            "final_answer": reason,
            "steps": [],
            "step_count": 0,
            "llm_calls": 0,
            "request_days": days,
        }
        save_json(RESULT_PATH, result)
        save_json(CHECKPOINT_PATH, result)
        return result

    result = run_agent(
        user_message=(
            f"Process a late-delivery refund for {order_id}. "
            f"Request age is {days} days. Verify using tools."
        ),
        scenario_id="DEMO",
        request_days=days,
        max_iterations=MAX_STEPS,
        token_limit=6000,
        time_limit_seconds=10,
    )

    old_status = {
        "REFUND": "simulated_refund_issued",
        "ESCALATE": "pending_human_approval",
        "DENIED": "rejected",
        "NOT_FOUND": "order_not_found",
    }.get(result["status"], result["status"].lower())

    legacy_result = {
        "order_id": str(order_id),
        "status": old_status,
        "final_answer": result["final_answer"],
        "steps": result["steps"],
        "step_count": result["step_count"],
        "llm_calls": result["llm_calls"],
        "request_days": days,
    }

    save_json(RESULT_PATH, legacy_result)
    save_json(CHECKPOINT_PATH, legacy_result)
    return legacy_result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--scenarios" in sys.argv:
        run_scenarios()
        return

    order_id = "ORD-110603"
    request_days = 3

    order = get_order(order_id)
    if "error" in order:
        print(order["error"])
        return

    print("Demo order:", order_id)
    print("Actual delivery:", order.get("delivery_minutes"), "minutes")
    print("Order amount: INR", order.get("order_amount"))

    result = run_refund_agent(
        order_id,
        request_days=request_days,
        resume=False,
    )

    print("\nAgent status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Steps:", result["step_count"])
    print("LLM calls:", result["llm_calls"])
    print("Request age (days):", result["request_days"])
    print("\nResult saved:", RESULT_PATH)
    print("Checkpoint saved:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
