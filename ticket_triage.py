
import json
import re
import time
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm



ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

# Safety and reliability settings
MAX_ATTEMPTS = 3
MAX_LLM_CALLS = 500
RETRY_DELAY_SECONDS = 0.1


def redact_pii(text):
    """Mask email addresses and phone-like numbers."""
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL_REDACTED]",
        text,
    )

    def mask_phone(match):
        value = match.group()
        digit_count = sum(char.isdigit() for char in value)

        if digit_count >= 10:
            return "[PHONE_REDACTED]"

        return value

    text = re.sub(
        r"(?<!\w)(?:\+?\d[\d\s().-]{8,}\d)(?!\w)",
        mask_phone,
        text,
    )

    return text


def build_system_prompt():
    """Tell the mock LLM to classify tickets using allowed categories."""
    return (
        "CLASSIFY support tickets. "
        "Return ONLY a JSON object with keys category and reason. "
        "category must be exactly one of: billing, delivery, technical, fraud. "
        "Do not include personal information in the response."
    )


def classify_ticket(redacted_text, budget):
    """
    Classify one redacted ticket.
    Retries temporary failures, validates JSON and category,
    and falls back safely when classification fails.
    """
    if budget["calls"] >= MAX_LLM_CALLS:
        return "technical", "call_budget_exhausted", 0, False

    attempts = 0

    while attempts < MAX_ATTEMPTS:
        if budget["calls"] >= MAX_LLM_CALLS:
            return "technical", "call_budget_exhausted", attempts, False

        attempts += 1
        budget["calls"] += 1

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": redacted_text},
        ]

        try:
            response = mock_llm.chat(messages)
            content = response.choices[0].message.content

            result = json.loads(content)

            category = result.get("category")
            reason = result.get("reason", "not_provided")

            if category not in ALLOWED_CATEGORIES:
                return "technical", "invalid_category_fallback", attempts, False

            return category, str(reason), attempts, True

        except mock_llm.TransientLLMError:
            if attempts < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

        except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
            return "technical", "invalid_response_fallback", attempts, False

        except Exception:
            return "technical", "unexpected_error_fallback", attempts, False

    return "technical", "retry_limit_fallback", attempts, False


def load_customer_segments():
    """Load segment information produced by Part B."""
    path = OUTPUT_DIR / "customer_segments.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run customer_segmentation.py first."
        )

    segments = pd.read_csv(path)

    required = {"customer_id", "segment_name", "is_premium"}
    missing = required - set(segments.columns)

    if missing:
        raise ValueError(f"Missing columns in customer_segments.csv: {missing}")

    segments["customer_id"] = segments["customer_id"].astype(str)

    return segments[["customer_id", "segment_name", "is_premium"]]


def route_ticket(category, segment_name, is_premium):
    """Apply deterministic routing rules after classification."""
    segment = str(segment_name).strip().lower()

    if category == "fraud":
        return "urgent_fraud_review"

    if category == "technical":
        return "technical_support"

    if category == "billing":
        return "billing_support"

    if category == "delivery":
        if bool(is_premium) or segment == "top value":
            return "priority_delivery_support"

        if segment == "no completed orders":
            return "new_customer_support"

        return "standard_delivery_support"

    return "human_review"


def calculate_metrics(results):
    """Calculate classification metrics when ground-truth labels exist."""
    valid = results[
        results["classification_valid"]
        & results["true_category"].isin(ALLOWED_CATEGORIES)
    ]

    if valid.empty:
        return {
            "evaluated_tickets": 0,
            "accuracy": None,
        }

    accuracy = (valid["predicted_category"] == valid["true_category"]).mean()

    return {
        "evaluated_tickets": int(len(valid)),
        "accuracy": round(float(accuracy), 4),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets_path = CLEAN_DIR / "tickets.csv"

    if not tickets_path.exists():
        raise FileNotFoundError(
            f"Missing {tickets_path}. Run clean_data.py first."
        )

    tickets = pd.read_csv(tickets_path)

    required = {"ticket_id", "customer_id", "order_id", "text", "true_category"}
    missing = required - set(tickets.columns)

    if missing:
        raise ValueError(f"Missing columns in tickets.csv: {missing}")

    # Reset deterministic mock behaviour for a reproducible run.
    mock_llm.reset()

    tickets["customer_id"] = tickets["customer_id"].astype(str)
    tickets["redacted_text"] = tickets["text"].apply(redact_pii)

    segments = load_customer_segments()

    tickets = tickets.merge(
        segments,
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    tickets["segment_name"] = tickets["segment_name"].fillna("Unknown")
    tickets["is_premium"] = tickets["is_premium"].fillna(False).astype(bool)

    budget = {"calls": 0}
    records = []

    for _, row in tickets.iterrows():
        category, reason, attempts, valid = classify_ticket(
            row["redacted_text"], budget
        )

        route = route_ticket(
            category,
            row["segment_name"],
            row["is_premium"],
        )

        records.append(
            {
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "order_id": row["order_id"],
                "redacted_text": row["redacted_text"],
                "true_category": row["true_category"],
                "predicted_category": category,
                "classification_reason": reason,
                "classification_valid": valid,
                "attempts": attempts,
                "segment_name": row["segment_name"],
                "is_premium": row["is_premium"],
                "route": route,
            }
        )

    results = pd.DataFrame(records)

    # Save redacted input separately from classification results.
    results[
        [
            "ticket_id",
            "customer_id",
            "order_id",
            "redacted_text",
            "true_category",
        ]
    ].to_csv(OUTPUT_DIR / "tickets_redacted.csv", index=False)

    results.to_csv(OUTPUT_DIR / "ticket_triage_results.csv", index=False)

    # Small summary report.
    metrics = calculate_metrics(results)

    summary = {
        "tickets_processed": int(len(results)),
        "llm_calls_used": int(budget["calls"]),
        "llm_call_budget": MAX_LLM_CALLS,
        "valid_classifications": int(results["classification_valid"].sum()),
        "fallback_classifications": int((~results["classification_valid"]).sum()),
        "retry_attempts": int((results["attempts"] > 1).sum()),
        "accuracy_on_valid_classifications": metrics["accuracy"],
        "evaluated_tickets": metrics["evaluated_tickets"],
    }

    with open(OUTPUT_DIR / "ticket_triage_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Ticket triage completed.")
    print(f"Tickets processed: {len(results)}")
    print(f"LLM calls used: {budget['calls']} / {MAX_LLM_CALLS}")
    print(f"Valid classifications: {summary['valid_classifications']}")
    print(f"Fallback classifications: {summary['fallback_classifications']}")
    print(f"Tickets retried: {summary['retry_attempts']}")
    print(f"Accuracy on valid classifications: {metrics['accuracy']}")
    print(f"Results saved: {OUTPUT_DIR / 'ticket_triage_results.csv'}")
    print(f"Summary saved: {OUTPUT_DIR / 'ticket_triage_summary.json'}")

    print("\nCategory counts:")
    print(results["predicted_category"].value_counts())

    print("\nRouting counts:")
    print(results["route"].value_counts())


if __name__ == "__main__":
    main()
