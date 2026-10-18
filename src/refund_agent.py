import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

OUTPUT_DIR = ROOT / "outputs"
POLICY_DIR = ROOT / "data" / "policies"
RESULT_PATH = OUTPUT_DIR / "refund_agent_demo.json"
CHECKPOINT_PATH = OUTPUT_DIR / "refund_agent_checkpoint.json"

MAX_STEPS = 6
MAX_LLM_CALLS = 20
AUTO_REFUND_LIMIT = 1000.0
REFUND_REQUEST_WINDOW_DAYS = 7
LATE_DELIVERY_LIMIT_MINUTES = 45.0


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_request_days(request_days):
    try:
        days = float(request_days)
    except (TypeError, ValueError):
        return False, None, "Invalid request age."

    if not math.isfinite(days) or days < 0:
        return False, days, "Invalid request age."

    if days > REFUND_REQUEST_WINDOW_DAYS:
        return False, days, "Refund request must be raised within 7 days of delivery."

    return True, days, "Request is within the allowed time window."


def get_order(order_id):
    path = ROOT / "data" / "clean" / "orders.csv"
    if not path.exists():
        return {"error": "Orders CSV not found."}

    try:
        orders = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        return {"error": f"Could not read orders CSV: {exc}"}

    required = {"order_id", "status", "order_amount", "delivery_minutes"}
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

        for sentence in re.split(r"(?<=[.!?])\s+", content.replace("\n", " ")):
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

    if not math.isfinite(amount) or amount <= 0 or amount > AUTO_REFUND_LIMIT:
        return {"error": "Automatic refund must be above zero and <= INR 1000."}

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
        amount, float(eligibility["refund_inr"]), rel_tol=0, abs_tol=0.001
    ):
        return {"error": "Refund amount failed verification."}

    return {
        "success": True,
        "order_id": str(order_id),
        "refund_inr": round(amount, 2),
        "message": "Simulated refund recorded. No real money was transferred.",
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
        return search_policy(str(arguments.get("query", "refund policy")))

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
                "You are an Instamart refund support agent. Verify orders and "
                "policies. Only completed orders qualify. Late-delivery refunds "
                "require delivery over 45 minutes. Requests must be raised "
                "within 7 days. Refunds above INR 1000 require human approval. "
                "All refunds are simulated."
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


def run_refund_agent(order_id, request_days=3, resume=True):
    # Mandatory policy guard: runs before checkpoint loading and any LLM call.
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
        save_json(CHECKPOINT_PATH, {
            "order_id": str(order_id),
            "request_days": days,
            "status": "rejected",
            "steps": [],
            "llm_calls": 0,
        })
        return result

    order_id = str(order_id)
    if not re.fullmatch(r"ORD-\d{6}", order_id):
        result = {
            "order_id": order_id,
            "status": "rejected",
            "final_answer": "Invalid order ID format.",
            "steps": [],
            "step_count": 0,
            "llm_calls": 0,
            "request_days": days,
        }
        save_json(RESULT_PATH, result)
        return result

    checkpoint = load_json(CHECKPOINT_PATH) if resume else None
    if (
        checkpoint
        and checkpoint.get("order_id") == order_id
        and checkpoint.get("request_days") == days
        and checkpoint.get("status") == "in_progress"
        and isinstance(checkpoint.get("messages"), list)
    ):
        messages = checkpoint["messages"]
        steps = checkpoint.get("steps", [])
        seen_actions = {tuple(item) for item in checkpoint.get("seen_actions", [])}
        llm_calls = int(checkpoint.get("llm_calls", 0))
    else:
        messages = make_initial_messages(order_id, days)
        steps = []
        seen_actions = set()
        llm_calls = 0

    status = "incomplete"
    final_answer = ""

    for step_number in range(len(steps) + 1, MAX_STEPS + 1):
        if llm_calls >= MAX_LLM_CALLS:
            status = "stopped_call_limit"
            final_answer = "Stopped: LLM call limit reached."
            break

        try:
            response = mock_llm.chat(messages)
            llm_calls += 1
        except Exception as exc:
            status = "llm_error"
            final_answer = f"LLM request failed: {exc}"
            break

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", []) or []

        if not tool_calls:
            final_answer = message.content or "No final answer returned."
            if final_answer.startswith("DENIED:"):
                status = "rejected"
            elif final_answer.startswith("NOT_FOUND:"):
                status = "order_not_found"
            elif final_answer.startswith("ESCALATE:"):
                status = "pending_human_approval"
            else:
                status = "completed"
            break

        tool_call = tool_calls[0]
        name = tool_call.name

        try:
            arguments = tool_call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object.")
        except (json.JSONDecodeError, TypeError, ValueError):
            status = "invalid_tool_arguments"
            final_answer = "Stopped: invalid tool arguments."
            break

        signature = (
            name,
            json.dumps(arguments, sort_keys=True, default=str),
        )
        if signature in seen_actions:
            status = "loop_detected"
            final_answer = "Stopped: repeated tool call detected."
            break
        seen_actions.add(signature)

        if name == "issue_refund":
            if str(arguments.get("order_id", "")) != order_id:
                observation = {"error": "Order ID mismatch."}
            else:
                order = get_order(order_id)
                if "error" in order:
                    observation = {"error": "Order verification failed."}
                elif str(order.get("status", "")).lower() != "completed":
                    observation = {"error": "Only completed orders can be refunded."}
                else:
                    eligibility = calculate_refund(
                        order.get("order_amount"),
                        order.get("delivery_minutes"),
                        request_days=days,
                    )
                    if not eligibility["eligible"]:
                        observation = {"error": eligibility["reason"]}
                    elif eligibility["requires_human_approval"]:
                        observation = {
                            "escalated": True,
                            "order_id": order_id,
                            "refund_inr": eligibility["refund_inr"],
                            "message": "Human approval is required.",
                        }
                    else:
                        observation = issue_refund(
                            order_id,
                            eligibility["refund_inr"],
                            approved=True,
                            request_days=days,
                        )
        else:
            observation = execute_tool(name, arguments, request_days=days)

        steps.append({
            "step": step_number,
            "action": name,
            "arguments": arguments,
            "observation": observation,
        })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "name": name,
                "arguments": json.dumps(arguments),
            }],
        })
        messages.append({
            "role": "tool",
            "name": name,
            "content": json.dumps(observation, default=str),
        })

        save_json(CHECKPOINT_PATH, {
            "order_id": order_id,
            "request_days": days,
            "status": "in_progress",
            "messages": messages,
            "steps": steps,
            "seen_actions": [list(item) for item in seen_actions],
            "llm_calls": llm_calls,
        })

        if name == "issue_refund" and observation.get("success"):
            status = "simulated_refund_issued"
            final_answer = observation["message"]
            break

        if observation.get("escalated"):
            status = "pending_human_approval"
            final_answer = observation.get("message", "Human approval is required.")
            break

        if name == "issue_refund" and observation.get("error"):
            status = "refund_blocked"
            final_answer = observation["error"]
            break
    else:
        status = "stopped_step_limit"
        final_answer = f"Stopped after {MAX_STEPS} steps."

    result = {
        "order_id": order_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "step_count": len(steps),
        "llm_calls": llm_calls,
        "request_days": days,
    }
    save_json(RESULT_PATH, result)
    save_json(CHECKPOINT_PATH, {
        "order_id": order_id,
        "request_days": days,
        "status": status,
        "steps": steps,
        "llm_calls": llm_calls,
    })
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
