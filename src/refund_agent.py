
import json
import math
import re
import sys
import time
import uuid
import sqlite3
from datetime import datetime, timezone
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


# --------------------------------------------------
# FILE HELPERS
# --------------------------------------------------

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


# --------------------------------------------------
# REFUND LEDGER
# --------------------------------------------------

def record_refund_ledger(
    tool_name,
    run_id,
    scenario_id,
    order_id,
    refund_inr,
    status,
):
    """Write one idempotent simulated refund or escalation event."""

    event_id = f"{run_id}:{tool_name}:{order_id}"

    REFUND_LEDGER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if REFUND_LEDGER_PATH.exists():
        with REFUND_LEDGER_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if existing.get("event_id") == event_id:
                    return existing

    record = {
        "event_id": event_id,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": str(order_id),
        "refund_inr": round(float(refund_inr), 2),
        "status": status,
        "real_money_transferred": False,
    }

    append_jsonl(REFUND_LEDGER_PATH, record)

    return record


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate_request_days(request_days):
    try:
        days = float(request_days)
    except (TypeError, ValueError):
        return False, None, "Invalid request age."

    if not math.isfinite(days) or days < 0:
        return False, days, "Invalid request age."

    if days > REFUND_REQUEST_WINDOW_DAYS:
        return (
            False,
            days,
            "Refund request must be raised within 7 days of delivery.",
        )

    return True, days, "Request is within the allowed time window."


# --------------------------------------------------
# ORDER DATABASE TOOL
# --------------------------------------------------

def get_order(order_id):
    """Fetch an order from SQLite using a parameterized query."""

    db_path = ROOT / "data" / "instamart.db"

    if not db_path.exists():
        return {"error": "SQLite database not found."}

    if not re.fullmatch(r"ORD-\d{6}", str(order_id)):
        return {"error": "Invalid order ID format."}

    try:
        with sqlite3.connect(db_path) as connection:
            connection.row_factory = sqlite3.Row

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT *
                FROM orders
                WHERE order_id = ?
                LIMIT 1
                """,
                (str(order_id),),
            )

            row = cursor.fetchone()

            if row is None:
                return {"error": "Order not found."}

            return dict(row)

    except sqlite3.Error as exc:
        return {
            "error": f"Could not read SQLite database: {exc}"
        }


# --------------------------------------------------
# POLICY SEARCH TOOL
# --------------------------------------------------

def search_policy(query):
    """Search policy documents and exclude generic headings."""

    if not POLICY_DIR.exists():
        return {"results": []}

    stop_words = {
        "the", "a", "an", "is", "are", "for",
        "to", "of", "and", "or", "in", "on",
        "what", "how", "can", "do", "does",
        "policy", "document", "please", "tell",
        "me", "about",
    }

    query_words = {
        word
        for word in re.findall(r"[a-z]+", str(query).lower())
        if word not in stop_words
    }

    if not query_words:
        query_words = {"refund"}

    results = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            content.replace("\r", " ").replace("\n", " "),
        )

        filename_words = set(
            re.findall(r"[a-z]+", path.stem.lower())
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            words = set(
                re.findall(r"[a-z]+", sentence.lower())
            )

            # Ignore short headings such as "Refund Policy."
            if len(words) < 5:
                continue

            overlap = len(query_words & words)

            if overlap == 0:
                continue

            filename_match = len(query_words & filename_words)
            score = overlap + filename_match

            results.append({
                "source": path.name,
                "text": sentence,
                "overlap": score,
            })

    results.sort(
        key=lambda item: item["overlap"],
        reverse=True,
    )

    return {"results": results[:5]}


# --------------------------------------------------
# REFUND CALCULATION TOOL
# --------------------------------------------------

def calculate_refund(
    order_amount,
    delivery_minutes,
    request_days=3,
):
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


# --------------------------------------------------
# ISSUE REFUND TOOL
# --------------------------------------------------

def issue_refund(
    order_id,
    refund_inr,
    approved=False,
    request_days=3,
    run_id=None,
    scenario_id=None,
):
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
            "error": (
                "Automatic refund must be above zero "
                "and <= INR 1000."
            )
        }

    order = get_order(order_id)

    if "error" in order:
        return {"error": "Order verification failed."}

    if str(order.get("status", "")).lower() != "completed":
        return {
            "error": "Only completed orders can be refunded."
        }

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

    ledger_record = None

    if run_id:
        ledger_record = record_refund_ledger(
            tool_name="issue_refund",
            run_id=run_id,
            scenario_id=scenario_id,
            order_id=order_id,
            refund_inr=amount,
            status="simulated_refund_issued",
        )

    return {
        "success": True,
        "order_id": str(order_id),
        "refund_inr": round(amount, 2),
        "ledger_recorded": ledger_record is not None,
        "message": (
            "Simulated refund recorded. "
            "No real money was transferred."
        ),
    }


# --------------------------------------------------
# HUMAN ESCALATION TOOL
# --------------------------------------------------

def escalate_to_human(
    order_id,
    refund_inr,
    request_days=3,
    run_id=None,
    scenario_id=None,
):
    if not re.fullmatch(r"ORD-\d{6}", str(order_id)):
        return {"error": "Invalid order ID format."}

    try:
        amount = float(refund_inr)
    except (TypeError, ValueError):
        return {"error": "Invalid refund amount."}

    if not math.isfinite(amount) or amount <= 0:
        return {"error": "Invalid refund amount."}

    order = get_order(order_id)

    if "error" in order:
        return {"error": "Order verification failed."}

    if str(order.get("status", "")).lower() != "completed":
        return {
            "error": "Only completed orders can be escalated."
        }

    eligibility = calculate_refund(
        order.get("order_amount"),
        order.get("delivery_minutes"),
        request_days=request_days,
    )

    if not eligibility["eligible"]:
        return {"error": eligibility["reason"]}

    if not eligibility["requires_human_approval"]:
        return {
            "error": "Human approval is not required for this amount."
        }

    if not math.isclose(
        amount,
        float(eligibility["refund_inr"]),
        rel_tol=0,
        abs_tol=0.001,
    ):
        return {"error": "Refund amount failed verification."}

    ledger_record = None

    if run_id:
        ledger_record = record_refund_ledger(
            tool_name="escalate_to_human",
            run_id=run_id,
            scenario_id=scenario_id,
            order_id=order_id,
            refund_inr=amount,
            status="escalated_to_human",
        )

    return {
        "escalated": True,
        "order_id": str(order_id),
        "refund_inr": round(amount, 2),
        "ledger_recorded": ledger_record is not None,
        "message": "Sent to human support lead for review.",
    }


# --------------------------------------------------
# TOOL SPECIFICATION
# --------------------------------------------------

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve an order from the SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search refund and support policy documents "
                "for relevant policy rules."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility and amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"},
                },
                "required": [
                    "order_amount",
                    "delivery_minutes",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an eligible simulated refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a refund above INR 1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# --------------------------------------------------
# TOOL DISPATCH
# --------------------------------------------------

def execute_tool(
    name,
    arguments,
    request_days=3,
    run_id=None,
    scenario_id=None,
):
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be an object."}

    if name not in TOOL_MAP:
        return {"error": f"Tool not allowed: {name}"}

    valid, days, reason = validate_request_days(request_days)

    if not valid and name in {
        "calculate_refund",
        "issue_refund",
        "escalate_to_human",
    }:
        return {
            "error": reason,
            "refund_inr": 0,
        }

    if name == "get_order":
        return TOOL_MAP[name](
            arguments.get("order_id", "")
        )

    if name == "search_policy":
        return TOOL_MAP[name](
            str(arguments.get("query", "refund policy"))
        )

    if name == "calculate_refund":
        return TOOL_MAP[name](
            arguments.get("order_amount"),
            arguments.get("delivery_minutes"),
            request_days=days,
        )

    if name == "issue_refund":
        return TOOL_MAP[name](
            arguments.get("order_id", ""),
            arguments.get("refund_inr", 0),
            approved=False,
            request_days=days,
            run_id=run_id,
            scenario_id=scenario_id,
        )

    if name == "escalate_to_human":
        return TOOL_MAP[name](
            arguments.get("order_id", ""),
            arguments.get("refund_inr", 0),
            request_days=days,
            run_id=run_id,
            scenario_id=scenario_id,
        )

    return {"error": f"Tool not allowed: {name}"}


# --------------------------------------------------
# INITIAL MESSAGES
# --------------------------------------------------

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


# --------------------------------------------------
# GUARDED AGENT TOOL EXECUTION
# --------------------------------------------------

def _agent_tool(
    name,
    arguments,
    order_id,
    request_days,
    run_id=None,
    scenario_id=None,
):
    """Validate tool requests before executing them."""

    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be an object."}

    if name == "issue_refund":
        if str(arguments.get("order_id", "")) != str(order_id):
            return {"error": "Order ID mismatch."}

        order = get_order(order_id)

        if "error" in order:
            return {"error": "Order verification failed."}

        if str(order.get("status", "")).lower() != "completed":
            return {
                "error": "Only completed orders can be refunded."
            }

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

        try:
            requested_amount = float(
                arguments.get("refund_inr", 0)
            )
        except (TypeError, ValueError):
            return {"error": "Invalid refund amount."}

        if not math.isclose(
            requested_amount,
            float(eligibility["refund_inr"]),
            rel_tol=0,
            abs_tol=0.001,
        ):
            return {"error": "Refund amount failed verification."}

        return issue_refund(
            order_id,
            eligibility["refund_inr"],
            approved=True,
            request_days=request_days,
            run_id=run_id,
            scenario_id=scenario_id,
        )

    if name == "escalate_to_human":
        if str(arguments.get("order_id", "")) != str(order_id):
            return {"error": "Order ID mismatch."}

        order = get_order(order_id)

        if "error" in order:
            return {"error": "Order verification failed."}

        if str(order.get("status", "")).lower() != "completed":
            return {
                "error": "Only completed orders can be escalated."
            }

        eligibility = calculate_refund(
            order.get("order_amount"),
            order.get("delivery_minutes"),
            request_days=request_days,
        )

        if not eligibility["eligible"]:
            return {"error": eligibility["reason"]}

        if not eligibility["requires_human_approval"]:
            return {
                "error": "Human approval is not required for this amount."
            }

        try:
            requested_amount = float(
                arguments.get("refund_inr", 0)
            )
        except (TypeError, ValueError):
            return {"error": "Invalid refund amount."}

        if not math.isclose(
            requested_amount,
            float(eligibility["refund_inr"]),
            rel_tol=0,
            abs_tol=0.001,
        ):
            return {"error": "Refund amount failed verification."}

        return escalate_to_human(
            order_id,
            eligibility["refund_inr"],
            request_days=request_days,
            run_id=run_id,
            scenario_id=scenario_id,
        )

    return execute_tool(
        name,
        arguments,
        request_days=request_days,
        run_id=run_id,
        scenario_id=scenario_id,
    )


# --------------------------------------------------
# TOKEN ESTIMATION
# --------------------------------------------------

def _token_estimate(messages):
    total = 0

    for message in messages:
        content = message.get("content") or ""
        total += mock_llm.est_tokens(str(content))

        if message.get("tool_calls"):
            total += mock_llm.est_tokens(
                json.dumps(
                    message["tool_calls"],
                    default=str,
                )
            )

    return total


# --------------------------------------------------
# AGENT STATE
# --------------------------------------------------

def _new_state(
    user_message,
    scenario_id,
    request_days,
    max_iterations,
    token_limit,
    time_limit_seconds,
    run_id,
):
    match = re.search(
        r"ORD-\d{6}",
        str(user_message),
    )

    order_id = match.group(0) if match else ""

    return {
        "run_id": run_id,
        "scenario_id": str(scenario_id),
        "user_message": str(user_message),
        "order_id": order_id,
        "request_days": request_days,
        "max_iterations": max(0, int(max_iterations)),
        "token_limit": max(0, int(token_limit)),
        "time_limit_seconds": max(
            0,
            float(time_limit_seconds),
        ),
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
            {
                "role": "user",
                "content": str(user_message),
            },
        ],
        "steps": [],
        "llm_calls": 0,
        "tokens_used": 0,
        "iterations": 0,
        "status": "in_progress",
        "final_answer": "",
        "ledger_recorded": False,
        "run_logged": False,
        "tool_call_history": [],
    }


# --------------------------------------------------
# CHECKPOINT
# --------------------------------------------------

def _checkpoint(state):
    path = STATE_DIR / (
        f"refund_agent_{state['run_id']}.json"
    )

    state["checkpoint_path"] = str(path)

    save_json(path, state)

    return path


# --------------------------------------------------
# FINISH AGENT RUN
# --------------------------------------------------

def _finish(state, status, answer):
    """Finish a run only when refund or escalation is verified."""

    steps = state.get("steps", [])

    refund_confirmed = any(
        step.get("action") == "issue_refund"
        and step.get("observation", {}).get("success") is True
        and step.get("observation", {}).get("ledger_recorded") is True
        for step in steps
    )

    escalation_confirmed = any(
        step.get("action") == "escalate_to_human"
        and step.get("observation", {}).get("escalated") is True
        and step.get("observation", {}).get("ledger_recorded") is True
        for step in steps
    )

    if status == "REFUND" and not refund_confirmed:
        status = "FALLBACK"
        answer = (
            "Refund claim rejected: no successful "
            "ledger-recorded refund was found."
        )

    elif status == "ESCALATE" and not escalation_confirmed:
        status = "FALLBACK"
        answer = (
            "Escalation claim rejected: no successful "
            "ledger-recorded escalation was found."
        )

    state["status"] = status
    state["final_answer"] = str(answer)

    state["ledger_recorded"] = (
        refund_confirmed or escalation_confirmed
    )

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
        "ledger_recorded": state["ledger_recorded"],
        "checkpoint_path": str(
            STATE_DIR / f"refund_agent_{state['run_id']}.json"
        ),
    }

    if not state.get("run_logged"):
        append_jsonl(AGENT_RUNS_PATH, result)
        state["run_logged"] = True

    save_json(
        STATE_DIR / f"refund_agent_{state['run_id']}.json",
        state,
    )

    save_json(
        OUTPUT_DIR / f"refund_agent_{state['run_id']}.json",
        result,
    )

    return result


# --------------------------------------------------
# AGENT EXECUTION LOOP
# --------------------------------------------------

def _continue_agent(state):
    start_time = time.monotonic()
    max_iterations = state["max_iterations"]

    # Support checkpoints created by an older version.
    state.setdefault("tool_call_history", [])

    valid, days, reason = validate_request_days(
        state["request_days"]
    )

    if not valid:
        return _finish(state, "DENIED", reason)

    if not state["order_id"] or not re.fullmatch(
        r"ORD-\d{6}",
        state["order_id"],
    ):
        return _finish(
            state,
            "NOT_FOUND",
            "Please provide a valid order ID.",
        )

    while state["iterations"] < max_iterations:

        if state["time_limit_seconds"] == 0:
            return _finish(
                state,
                "FALLBACK_TIMEOUT",
                "Stopped safely because the time limit is zero.",
            )

        elapsed = time.monotonic() - start_time

        if elapsed >= state["time_limit_seconds"]:
            return _finish(
                state,
                "FALLBACK_TIMEOUT",
                "Stopped safely because the time limit was reached.",
            )

        if state["tokens_used"] >= state["token_limit"]:
            return _finish(
                state,
                "FALLBACK_TOKEN_LIMIT",
                "Stopped safely because the token limit was reached.",
            )

        if state["llm_calls"] >= MAX_LLM_CALLS:
            return _finish(
                state,
                "FALLBACK_MAX_ITERATIONS",
                "Stopped safely because the LLM call limit was reached.",
            )

        estimated_prompt = _token_estimate(
            state["messages"]
        )

        if estimated_prompt >= state["token_limit"]:
            return _finish(
                state,
                "FALLBACK_TOKEN_LIMIT",
                (
                    "Stopped safely before the next LLM call: "
                    "token budget exhausted."
                ),
            )

        try:
            response = mock_llm.chat(
                state["messages"],
                tools=TOOLS_SPEC,
            )

            state["llm_calls"] += 1
            state["iterations"] += 1

        except Exception as exc:
            state["status"] = "in_progress"
            state["last_error"] = str(exc)

            _checkpoint(state)

            return _finish(
                state,
                "FALLBACK_LLM_ERROR",
                f"LLM call failed safely: {exc}",
            )

        usage = getattr(response, "usage", None)

        call_tokens = int(
            getattr(usage, "total_tokens", 0) or 0
        )

        if (
            state["tokens_used"] + call_tokens
            > state["token_limit"]
        ):
            state["tokens_used"] += call_tokens

            return _finish(
                state,
                "FALLBACK_TOKEN_LIMIT",
                (
                    "Stopped safely because the response "
                    "exceeded the token budget."
                ),
            )

        state["tokens_used"] += call_tokens

        message = response.choices[0].message

        tool_calls = getattr(
            message,
            "tool_calls",
            [],
        ) or []

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
                raise ValueError(
                    "Tool arguments must be an object."
                )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return _finish(
                state,
                "FALLBACK",
                "Stopped safely because tool arguments were invalid.",
            )

        # Detect an identical repeated tool call before executing it.
        call_signature = json.dumps(
            {
                "name": name,
                "arguments": arguments,
            },
            sort_keys=True,
            default=str,
        )

        if call_signature in state["tool_call_history"]:
            return _finish(
                state,
                "FALLBACK",
                (
                    "Stopped safely because the agent repeated "
                    "the same tool call without making progress."
                ),
            )

        state["tool_call_history"].append(call_signature)

        try:
            observation = _agent_tool(
                name,
                arguments,
                state["order_id"],
                days,
                run_id=state["run_id"],
                scenario_id=state["scenario_id"],
            )
        except Exception as exc:
            state["last_error"] = (
                f"Tool '{name}' failed: {exc}"
            )
            return _finish(
                state,
                "FALLBACK_TOOL_ERROR",
                (
                    f"Stopped safely because tool "
                    f"'{name}' failed: {exc}"
                ),
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
            "content": json.dumps(
                observation,
                default=str,
            ),
        })

        # Save immediately after every executed tool.
        _checkpoint(state)

        if (
            name == "issue_refund"
            and observation.get("success")
        ):
            continue

        if observation.get("escalated"):
            continue

    return _finish(
        state,
        "FALLBACK_MAX_ITERATIONS",
        f"Stopped safely after {max_iterations} iterations.",
    )


# --------------------------------------------------
# START A NEW AGENT RUN
# --------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    request_days=3,
):
    """Start a new run and checkpoint each executed tool."""

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


# --------------------------------------------------
# RESUME AN EXISTING RUN
# --------------------------------------------------

def resume_agent(run_id):
    """Resume a checkpoint without re-executing completed tools."""

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

    if state.get("status") != "in_progress":
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
            "ledger_recorded": state.get("ledger_recorded", False),
        }

    return _continue_agent(state)


# --------------------------------------------------
# RUN ALL SCENARIOS
# --------------------------------------------------

def normalize_scenarios(scenarios):
    """Return scenarios with the escalation-amount column present.

    The supplied generate_data.py writes agent_scenarios.csv without an
    ``expected_escalated_amount_inr`` column; for an ESCALATE scenario it
    stores the escalated order amount in ``expected_refund_inr``. When the
    column is absent, derive it so that no refund is expected for ESCALATE
    rows and the escalated amount is kept separately.
    """

    scenarios = scenarios.copy()

    if "expected_escalated_amount_inr" in scenarios.columns:
        return scenarios

    amounts = pd.to_numeric(
        scenarios["expected_refund_inr"], errors="coerce"
    ).fillna(0.0)
    is_escalation = (
        scenarios["expected_outcome"].astype(str).str.strip().str.upper()
        == "ESCALATE"
    )

    scenarios["expected_escalated_amount_inr"] = amounts.where(
        is_escalation, 0.0
    )
    scenarios["expected_refund_inr"] = amounts.where(
        ~is_escalation, 0.0
    )

    return scenarios


def run_scenarios():
    """Run scenarios and independently verify refunds and escalations."""

    scenario_path = (
        ROOT / "data" / "raw" / "agent_scenarios.csv"
    )

    if not scenario_path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {scenario_path}"
        )

    scenarios = normalize_scenarios(
        pd.read_csv(scenario_path)
    ).fillna("")

    required_columns = {
        "scenario_id",
        "message",
        "expected_outcome",
        "expected_refund_inr",
        "expected_escalated_amount_inr",
    }

    missing_columns = required_columns - set(scenarios.columns)

    if missing_columns:
        raise ValueError(
            "Scenario CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    results = []

    for _, row in scenarios.iterrows():
        scenario_id = str(row["scenario_id"])
        message = str(row["message"])
        expected = str(
            row["expected_outcome"]
        ).strip().upper()

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

        expected_refund = float(
            row["expected_refund_inr"] or 0
        )

        expected_escalated = float(
            row["expected_escalated_amount_inr"] or 0
        )

        actual_refund = 0.0
        actual_escalated = 0.0

        # Independently verify this scenario's ledger records
        run_id = result.get("run_id")

        ledger_records = []
        if REFUND_LEDGER_PATH.exists():
            with open(REFUND_LEDGER_PATH, encoding="utf-8") as ledger_file:
                ledger_records = [
                    json.loads(line)
                    for line in ledger_file
                    if line.strip()
                ]

        scenario_ledger = [
            record for record in ledger_records
            if record.get("run_id") == run_id
            and record.get("scenario_id") == scenario_id
        ]

        if expected_refund > 0:
            expected_ledger_status = "simulated_refund_issued"
            expected_ledger_amount = expected_refund
        elif expected_escalated > 0:
            expected_ledger_status = "escalated_to_human"
            expected_ledger_amount = expected_escalated
        else:
            expected_ledger_status = None
            expected_ledger_amount = 0.0

        if expected_ledger_status is None:
            ledger_match = len(scenario_ledger) == 0
        else:
            ledger_match = (
                len(scenario_ledger) == 1
                and scenario_ledger[0].get("status") == expected_ledger_status
                and math.isclose(
                    float(scenario_ledger[0].get("refund_inr", 0)),
                    expected_ledger_amount,
                    rel_tol=0,
                    abs_tol=0.01,
                )
                and scenario_ledger[0].get("real_money_transferred") is False
                and bool(scenario_ledger[0].get("event_id"))
                and scenario_ledger[0].get("tool_name") in (
                    "issue_refund",
                    "escalate_to_human",
                )
            )

        for step in result.get("steps", []):
            observation = step.get("observation", {})

            if observation.get("success") is True:
                actual_refund = float(
                    observation.get("refund_inr", 0)
                )

            if observation.get("escalated") is True:
                actual_escalated = float(
                    observation.get("refund_inr", 0)
                )

        refund_match = math.isclose(
            expected_refund,
            actual_refund,
            rel_tol=0,
            abs_tol=0.01,
        )

        escalation_match = math.isclose(
            expected_escalated,
            actual_escalated,
            rel_tol=0,
            abs_tol=0.01,
        )

        results.append({
            "scenario_id": scenario_id,
            "expected_outcome": expected,
            "actual_outcome": actual_outcome,
            "outcome_match": actual_outcome == expected,
            "expected_refund_inr": expected_refund,
            "actual_refund_inr": actual_refund,
            "refund_match": refund_match,
            "expected_escalated_amount_inr": expected_escalated,
            "actual_escalated_amount_inr": actual_escalated,
            "escalation_match": escalation_match,
            "ledger_match": ledger_match,
            "run_id": result.get("run_id"),
            "status": actual,
            "llm_calls": result.get("llm_calls", 0),
        })

    output_path = (
        OUTPUT_DIR / "refund_agent_scenarios.csv"
    )

    results_df = pd.DataFrame(results)
    print(
        "Ledger matches:",
        int(results_df["ledger_match"].sum()),
        "/",
        len(results_df),
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n========== REFUND SCENARIO RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\n========== VALIDATION SUMMARY ==========")
    print(
        "Outcome matches:",
        int(results_df["outcome_match"].sum()),
        "/",
        len(results_df),
    )
    print(
        "Refund matches:",
        int(results_df["refund_match"].sum()),
        "/",
        len(results_df),
    )
    print(
        "Escalation matches:",
        int(results_df["escalation_match"].sum()),
        "/",
        len(results_df),
    )

    all_passed = (
        results_df["outcome_match"].all()
        and results_df["refund_match"].all()
        and results_df["escalation_match"].all()
        and results_df["ledger_match"].all()
    )

    print("All scenario checks passed:", all_passed)

    print("\nScenario results saved:", output_path)
    print("Agent run log:", AGENT_RUNS_PATH)
    print("Refund ledger:", REFUND_LEDGER_PATH)

    return results


# --------------------------------------------------
# COMPATIBILITY WRAPPER
# --------------------------------------------------

def run_refund_agent(
    order_id,
    request_days=3,
    resume=True,
):
    """Compatibility wrapper for the original demo and tests."""

    valid, days, reason = validate_request_days(
        request_days
    )

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
    }.get(
        result["status"],
        result["status"].lower(),
    )

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


# --------------------------------------------------
# MAIN
# --------------------------------------------------

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
    print(
        "Actual delivery:",
        order.get("delivery_minutes"),
        "minutes",
    )
    print(
        "Order amount: INR",
        order.get("order_amount"),
    )

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
