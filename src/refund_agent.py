
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm


OUTPUT_DIR = ROOT / "outputs"
POLICY_DIR = ROOT / "data" / "policies"

MAX_STEPS = 6
MAX_LLM_CALLS = 20
AUTO_REFUND_LIMIT = 1000.0

RESULT_PATH = OUTPUT_DIR / "refund_agent_demo.json"
CHECKPOINT_PATH = OUTPUT_DIR / "refund_agent_checkpoint.json"


# --------------------------------------------------
# JSON helpers
# --------------------------------------------------

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path):
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------
# Tool 1: Get order
# --------------------------------------------------

def get_order(order_id):
    path = ROOT / "data" / "clean" / "orders.csv"

    if not path.exists():
        return {"error": "Orders CSV not found."}

    orders = pd.read_csv(path)

    required = {
        "order_id",
        "status",
        "order_amount",
        "delivery_minutes",
    }

    if not required.issubset(orders.columns):
        return {"error": "Required order columns are missing."}

    matches = orders[
        orders["order_id"].astype(str) == str(order_id)
    ]

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


# --------------------------------------------------
# Tool 2: Search policy
# --------------------------------------------------

def search_policy(query):
    if not POLICY_DIR.exists():
        return {"results": []}

    query_words = set(re.findall(r"[a-z]+", query.lower()))
    results = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8")

        sentences = re.split(
            r"(?<=[.!?])\s+",
            content.replace("\n", " "),
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            words = set(
                re.findall(r"[a-z]+", sentence.lower())
            )
            overlap = len(query_words & words)

            if overlap:
                results.append({
                    "source": path.name,
                    "text": sentence,
                    "overlap": overlap,
                })

    results.sort(
        key=lambda item: item["overlap"],
        reverse=True,
    )

    return {"results": results[:5]}


# --------------------------------------------------
# Tool 3: Calculate refund
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
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
        not pd.notna(amount)
        or not pd.notna(minutes)
        or amount <= 0
        or minutes < 0
    ):
        return {
            "refund_inr": 0,
            "eligible": False,
            "requires_human_approval": False,
            "reason": "Invalid order details.",
        }

    if minutes <= 45:
        return {
            "refund_inr": 0,
            "eligible": False,
            "requires_human_approval": False,
            "reason": "Delivery took 45 minutes or less.",
        }

    if amount > AUTO_REFUND_LIMIT:
        return {
            "refund_inr": round(amount, 2),
            "eligible": True,
            "requires_human_approval": True,
            "reason": "Refund exceeds INR 1000 automatic limit.",
        }

    return {
        "refund_inr": round(amount, 2),
        "eligible": True,
        "requires_human_approval": False,
        "reason": "Late-delivery refund conditions satisfied.",
    }


# --------------------------------------------------
# Tool 4: Issue simulated refund
# --------------------------------------------------

def issue_refund(order_id, refund_inr, approved=False):
    if not approved:
        return {"error": "Refund approval is required."}

    try:
        amount = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "Invalid refund amount."}

    if not 0 < amount <= AUTO_REFUND_LIMIT:
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
    )

    if not eligibility["eligible"]:
        return {"error": eligibility["reason"]}

    if eligibility["requires_human_approval"]:
        return {"error": "Human approval is required."}

    if amount != float(eligibility["refund_inr"]):
        return {"error": "Refund amount failed verification."}

    return {
        "success": True,
        "order_id": str(order_id),
        "refund_inr": round(amount, 2),
        "message": (
            "Simulated refund recorded. No real money was transferred."
        ),
    }


# --------------------------------------------------
# Tool 5: Escalate to human
# --------------------------------------------------

def escalate_to_human(order_id, refund_inr):
    return {
        "escalated": True,
        "order_id": str(order_id),
        "refund_inr": float(refund_inr),
        "message": "Sent to human support lead for review.",
    }


# --------------------------------------------------
# Allowlisted tool dispatcher
# --------------------------------------------------

def execute_tool(name, arguments):
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be an object."}

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
        )

    if name == "issue_refund":
        return issue_refund(
            arguments.get("order_id", ""),
            arguments.get("refund_inr", 0),
            approved=False,
        )

    if name == "escalate_to_human":
        return escalate_to_human(
            arguments.get("order_id", ""),
            arguments.get("refund_inr", 0),
        )

    return {"error": f"Tool not allowed: {name}"}


# --------------------------------------------------
# Initial conversation
# --------------------------------------------------

def make_initial_messages(order_id, request_days):
    return [
        {
            "role": "system",
            "content": (
                "AGENT. You are an Instamart refund support agent. "
                "Verify the order, search the policy, calculate eligibility, "
                "then issue a refund or escalate. Never issue a refund when "
                "the calculator returns zero. Only completed orders qualify. "
                "Refunds above INR 1000 require human approval. "
                "Never invent order data."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Process late-delivery refund request for {order_id}. "
                f"Request was raised {request_days} days after delivery. "
                "Use tools to verify all facts."
            ),
        },
    ]


# --------------------------------------------------
# ReAct agent with checkpointing
# --------------------------------------------------

def run_refund_agent(order_id, request_days=3, resume=True):
    checkpoint = load_json(CHECKPOINT_PATH) if resume else None

    if (
        checkpoint
        and checkpoint.get("order_id") == order_id
        and checkpoint.get("request_days") == request_days
        and checkpoint.get("status") == "in_progress"
    ):
        messages = checkpoint["messages"]
        steps = checkpoint.get("steps", [])
        seen_actions = {
            tuple(item)
            for item in checkpoint.get("seen_actions", [])
        }
        llm_calls = checkpoint.get("llm_calls", 0)
        print("Resuming saved checkpoint.")
    else:
        messages = make_initial_messages(order_id, request_days)
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

        except mock_llm.TransientLLMError:
            if llm_calls >= MAX_LLM_CALLS:
                status = "stopped_call_limit"
                final_answer = "Stopped after transient error."
                break

            try:
                response = mock_llm.chat(messages)
                llm_calls += 1
            except Exception as exc:
                status = "llm_error"
                final_answer = f"LLM request failed: {exc}"
                break

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
            elif final_answer.startswith("REFUND:"):
                status = "simulated_refund_issued"
            elif final_answer.startswith("NEEDS_INFO:"):
                status = "needs_information"
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
                raise ValueError("Tool arguments must be a JSON object.")

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

        # Verify refund requests against the actual CSV record.
        if name == "issue_refund":
            if str(arguments.get("order_id", "")) != order_id:
                observation = {"error": "Order ID mismatch."}
            else:
                verified_order = get_order(order_id)

                if "error" in verified_order:
                    observation = {"error": "Order verification failed."}

                elif str(
                    verified_order.get("status", "")
                ).lower() != "completed":
                    observation = {
                        "error": "Only completed orders can be refunded."
                    }

                else:
                    eligibility = calculate_refund(
                        verified_order.get("order_amount"),
                        verified_order.get("delivery_minutes"),
                    )

                    if not eligibility["eligible"]:
                        observation = {
                            "error": eligibility["reason"],
                            "refund_inr": 0,
                        }

                    elif eligibility["requires_human_approval"]:
                        observation = {
                            "error": "Human approval is required."
                        }

                    elif float(
                        arguments.get("refund_inr", 0)
                    ) != float(eligibility["refund_inr"]):
                        observation = {
                            "error": "Refund amount failed verification."
                        }

                    else:
                        observation = issue_refund(
                            order_id,
                            eligibility["refund_inr"],
                            approved=True,
                        )

        else:
            observation = execute_tool(name, arguments)

        steps.append({
            "step": step_number,
            "action": name,
            "arguments": arguments,
            "observation": observation,
        })

        # Append tool call and its observation to the LLM history.
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

        # Save progress after each tool observation.
        save_json(CHECKPOINT_PATH, {
            "order_id": order_id,
            "request_days": request_days,
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

        if name == "escalate_to_human" and observation.get("escalated"):
            status = "pending_human_approval"
            final_answer = observation["message"]
            break

        if name == "issue_refund" and observation.get("error"):
            status = "refund_blocked"
            final_answer = observation["error"]
            break

    else:
        status = "stopped_step_limit"
        final_answer = f"Stopped after {MAX_STEPS} tool steps."

    result = {
        "order_id": order_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "step_count": len(steps),
        "llm_calls": llm_calls,
        "request_days": request_days,
    }

    save_json(RESULT_PATH, result)

    # Mark checkpoint terminal; do not resume a finished run.
    save_json(CHECKPOINT_PATH, {
        "order_id": order_id,
        "request_days": request_days,
        "status": status,
        "steps": steps,
        "llm_calls": llm_calls,
    })

    return result


# --------------------------------------------------
# Main demo
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    orders_path = ROOT / "data" / "clean" / "orders.csv"

    if not orders_path.exists():
        print("Orders CSV not found.")
        return

    orders = pd.read_csv(orders_path)

    required = {
        "order_id",
        "status",
        "order_amount",
        "delivery_minutes",
    }

    if not required.issubset(orders.columns):
        print("Missing required columns.")
        print(orders.columns.tolist())
        return

    completed = orders[
        orders["status"].astype(str).str.lower() == "completed"
    ]

    if completed.empty:
        print("No completed orders found.")
        return

    # Change this ID to test another order.
    order_id = "ORD-110603"

    selected = completed[
        completed["order_id"].astype(str) == order_id
    ]

    if selected.empty:
        print("Selected order is not a completed order:", order_id)
        return

    demo_order = selected.iloc[0]

    print("Demo order:", order_id)
    print("Actual delivery:", demo_order["delivery_minutes"], "minutes")
    print("Order amount: INR", demo_order["order_amount"])

    result = run_refund_agent(
        order_id=order_id,
        request_days=3,
        resume=True,
    )

    print("\nAgent status:", result["status"])
    print("Final answer:", result["final_answer"])
    print("Steps:", result["step_count"])
    print("LLM calls:", result["llm_calls"])

    for step in result["steps"]:
        print(f"\nStep {step['step']}: {step['action']}")
        print(json.dumps(step, indent=2, default=str))

    print("\nResult saved:", RESULT_PATH)
    print("Checkpoint saved:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
