
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm  # Do not modify mock_llm.py


CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

MAX_RETRIES = 3
BASE_DELAY = 0.05
LONG_TICKET_TOKEN_LIMIT = 400


ZERO_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only:
{"category": "...", "reason": "..."}

Use only the allowed categories.
Do not include markdown or text outside the JSON.
""".strip()


FEW_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only:
{"category": "...", "reason": "..."}

Use only the allowed categories.
Do not include markdown or text outside the JSON.

EXAMPLES:

Ticket: I was charged twice for my order.
Output: {"category": "billing", "reason": "Duplicate charge"}

Ticket: My order has not arrived.
Output: {"category": "delivery", "reason": "Delivery delay"}

Ticket: The app crashes when I open it.
Output: {"category": "technical", "reason": "App crash"}

Ticket: Someone accessed my account without permission.
Output: {"category": "fraud", "reason": "Possible unauthorized access"}
""".strip()


# --------------------------------------------------
# 1. PII REDACTION
# --------------------------------------------------

def redact_pii(text):
    """Redact email, phone and order ID before model processing."""

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text,
    )

    text = re.sub(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text,
    )

    text = re.sub(
        r"\bORD-\d{6}\b",
        "[ORDER_ID]",
        text,
        flags=re.IGNORECASE,
    )

    return text


def count_pii(text):
    """Count occurrences of the PII patterns used by this project."""

    text = "" if pd.isna(text) else str(text)

    return {
        "emails": len(
            re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
        ),
        "phones": len(
            re.findall(r"\b[6-9]\d{9}\b", text)
        ),
        "order_ids": len(
            re.findall(r"\bORD-\d{6}\b", text, flags=re.IGNORECASE)
        ),
    }


# --------------------------------------------------
# 2. CUSTOMER SEGMENTS
# --------------------------------------------------

def load_customer_segments():
    path = OUTPUT_DIR / "customer_segments.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Generate customer_segments.csv first."
        )

    df = pd.read_csv(path)

    if "customer_id" not in df.columns:
        raise ValueError(
            "customer_segments.csv must contain customer_id."
        )

    segment_col = next(
        (
            col
            for col in ("segment_name", "segment", "cluster_name")
            if col in df.columns
        ),
        None,
    )

    if segment_col is None:
        raise ValueError(
            "customer_segments.csv must contain a segment column."
        )

    if "is_premium" not in df.columns:
        raise ValueError(
            "customer_segments.csv must contain is_premium."
        )

    segments = {}

    for _, row in df.iterrows():
        customer_id = str(row["customer_id"]).strip()
        premium_value = str(row["is_premium"]).strip().lower()

        segments[customer_id] = {
            "segment_name": (
                str(row[segment_col])
                if pd.notna(row[segment_col])
                else "Unknown"
            ),
            "is_premium": premium_value in {
                "true", "1", "yes"
            },
        }

    return segments


# --------------------------------------------------
# 3. MODEL RESPONSE PARSING
# --------------------------------------------------

def parse_model_output(raw_text):
    """
    Valid JSON with an allowed category -> classified_json.
    Invalid JSON with exactly one allowed category keyword
    -> classified_keyword.
    Other outputs -> flagged_invalid_output.
    """

    try:
        payload = json.loads(raw_text)

        if isinstance(payload, dict):
            category = str(
                payload.get("category", "")
            ).strip().lower()

            reason = str(
                payload.get("reason", "")
            ).strip()

            if category in ALLOWED_CATEGORIES:
                return category, reason, "classified_json"

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    words = re.findall(r"[a-z]+", str(raw_text).lower())

    category_words = [
        word for word in words
        if word in ALLOWED_CATEGORIES
    ]

    unique_categories = set(category_words)

    if len(unique_categories) == 1:
        category = next(iter(unique_categories))

        return (
            category,
            "Recovered from category keyword",
            "classified_keyword",
        )

    return "", "", "flagged_invalid_output"


# --------------------------------------------------
# 4. MODEL CALL WITH RETRIES
# --------------------------------------------------

def call_model_with_retry(system_prompt, user_text, ticket_id):
    """Call mock_llm with retries for transient errors."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)

            raw_text = (
                response.choices[0].message.content or ""
            )

            token_count = int(response.usage.total_tokens)

            return raw_text, token_count

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry: ticket_id={ticket_id}, "
                f"retry_number={retry_number}, "
                f"wait_seconds={wait_seconds:.4f}"
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected retry-loop exit")


# --------------------------------------------------
# 5. ROUTING
# --------------------------------------------------

def route_ticket(status, category, is_premium):
    """Select ticket destination and priority."""

    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    if status in human_review_statuses:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


# --------------------------------------------------
# 6. METRICS
# --------------------------------------------------

def calculate_accuracy(results):
    """
    Calculate accuracy only for rows with valid actual and
    predicted categories. Empty/unknown labels are excluded.
    """

    if (
        "true_category" not in results.columns
        or "category" not in results.columns
    ):
        return None

    actual = (
        results["true_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    predicted = (
        results["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    valid_rows = (
        actual.isin(ALLOWED_CATEGORIES)
        & predicted.isin(ALLOWED_CATEGORIES)
    )

    if not valid_rows.any():
        return None

    return float(
        (actual[valid_rows] == predicted[valid_rows]).mean()
    )


def make_confusion_matrix(results):
    """Build a confusion matrix from valid labelled tickets."""

    if (
        "true_category" not in results.columns
        or "category" not in results.columns
    ):
        return {}

    evaluated = results.copy()

    evaluated["actual"] = (
        evaluated["true_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    evaluated["predicted"] = (
        evaluated["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    evaluated = evaluated[
        evaluated["actual"].isin(ALLOWED_CATEGORIES)
        & evaluated["predicted"].isin(ALLOWED_CATEGORIES)
    ]

    if evaluated.empty:
        return {}

    labels = sorted(ALLOWED_CATEGORIES)

    matrix = pd.crosstab(
        evaluated["actual"],
        evaluated["predicted"],
    ).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )

    return {
        "labels": labels,
        "matrix": matrix.values.tolist(),
    }


# --------------------------------------------------
# 7. TRIAGE ONE BATCH
# --------------------------------------------------

def run_triage(
    tickets,
    segments,
    system_prompt,
    run_name,
    token_budget=None,
):
    results = []
    total_tokens = 0
    successful_model_calls = 0

    for _, ticket in tickets.iterrows():
        ticket_id = ticket.get("ticket_id", "")
        customer_id = ticket.get("customer_id", "")
        original_text = ticket.get("text", "")

        # Redact sensitive information before model processing.
        redacted_text = redact_pii(original_text)
        pii_remaining = count_pii(redacted_text)

        segment_info = segments.get(
            str(customer_id).strip(),
            {
                "segment_name": "Unknown",
                "is_premium": False,
            },
        )

        segment_name = segment_info["segment_name"]
        is_premium = segment_info["is_premium"]

        category = ""
        reason = ""
        status = ""
        tokens_used = 0

        # Guardrail 1: skip empty tickets.
        if not redacted_text.strip():
            status = "skipped_empty"

        else:
            estimated_ticket_tokens = mock_llm.est_tokens(
                redacted_text
            )

            # Guardrail 2: send overly long tickets for review.
            if estimated_ticket_tokens > LONG_TICKET_TOKEN_LIMIT:
                status = "flagged_long"

            else:
                estimated_call_tokens = (
                    mock_llm.est_tokens(system_prompt)
                    + estimated_ticket_tokens
                )

                # Guardrail 3: don't start a call if the estimated
                # prompt reaches or exceeds the remaining budget.
                if (
                    token_budget is not None
                    and total_tokens + estimated_call_tokens
                    >= token_budget
                ):
                    status = "budget_exhausted"

                else:
                    try:
                        raw_output, tokens_used = (
                            call_model_with_retry(
                                system_prompt=system_prompt,
                                user_text=redacted_text,
                                ticket_id=ticket_id,
                            )
                        )

                        successful_model_calls += 1
                        total_tokens += tokens_used

                        category, reason, status = (
                            parse_model_output(raw_output)
                        )

                    except mock_llm.TransientLLMError:
                        status = "flagged_invalid_output"
                        reason = (
                            "Transient model failure after retries"
                        )

        route, priority = route_ticket(
            status,
            category,
            is_premium,
        )

        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "reason": reason,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "emails_remaining": pii_remaining["emails"],
            "phones_remaining": pii_remaining["phones"],
            "order_ids_remaining": pii_remaining["order_ids"],
            "redacted_text": redacted_text,
        }

        if "true_category" in tickets.columns:
            result["true_category"] = ticket.get(
                "true_category", ""
            )

        results.append(result)

    output = pd.DataFrame(results)

    classified_json = int(
        (output["status"] == "classified_json").sum()
    )

    classified_keyword = int(
        (output["status"] == "classified_keyword").sum()
    )

    classified_total = classified_json + classified_keyword

    summary = {
        "run_name": run_name,
        "tickets_total": len(tickets),
        "model_calls": successful_model_calls,
        "tokens_used_total": total_tokens,
        "classified_json": classified_json,
        "classified_keyword": classified_keyword,
        "classified_total": classified_total,
        "flagged_invalid_output": int(
            (output["status"] == "flagged_invalid_output").sum()
        ),
        "skipped_empty": int(
            (output["status"] == "skipped_empty").sum()
        ),
        "flagged_long": int(
            (output["status"] == "flagged_long").sum()
        ),
        "budget_exhausted": int(
            (output["status"] == "budget_exhausted").sum()
        ),
        "accuracy": calculate_accuracy(output),
        "tokens_per_classified": (
            total_tokens / classified_total
            if classified_total
            else None
        ),
    }

    return output, summary


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tickets_path = CLEAN_DIR / "tickets.csv"

    if not tickets_path.exists():
        raise FileNotFoundError(
            f"Missing {tickets_path}. "
            "Run the data preparation scripts first."
        )

    tickets = pd.read_csv(tickets_path)

    required_columns = {
        "ticket_id",
        "customer_id",
        "text",
    }

    missing_columns = required_columns - set(tickets.columns)

    if missing_columns:
        raise ValueError(
            f"tickets.csv is missing columns: "
            f"{sorted(missing_columns)}"
        )

    segments = load_customer_segments()

    # PII audit: count patterns before and after redaction.
    pii_before = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0,
    }

    pii_after = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0,
    }

    for original_text in tickets["text"].fillna("").astype(str):
        before = count_pii(original_text)
        after = count_pii(redact_pii(original_text))

        for key in pii_before:
            pii_before[key] += before[key]
            pii_after[key] += after[key]

    pii_audit = {
        "before_redaction": pii_before,
        "after_redaction": pii_after,
        "total_substitutions": sum(pii_before.values()),
    }

    # Run 1: zero-shot classification.
    mock_llm.reset()

    zero_results, zero_summary = run_triage(
        tickets,
        segments,
        ZERO_SHOT_PROMPT,
        "zero_shot",
    )

    zero_results.to_csv(
        OUTPUT_DIR / "triage_zero_shot.csv",
        index=False,
    )

    confusion = make_confusion_matrix(zero_results)

    if confusion:
        confusion_df = pd.DataFrame(
            confusion["matrix"],
            index=confusion["labels"],
            columns=confusion["labels"],
        )

        confusion_df.index.name = "actual"

        confusion_df.to_csv(
            OUTPUT_DIR / "triage_confusion_zero_shot.csv"
        )

    # Run 2: few-shot classification.
    mock_llm.reset()

    few_results, few_summary = run_triage(
        tickets,
        segments,
        FEW_SHOT_PROMPT,
        "few_shot",
    )

    few_results.to_csv(
        OUTPUT_DIR / "triage_few_shot.csv",
        index=False,
    )

    # Run 3: token-budget guardrail demo.
    budget = 6000

    mock_llm.reset()

    budget_results, budget_summary = run_triage(
        tickets,
        segments,
        ZERO_SHOT_PROMPT,
        "budget_demo",
        token_budget=budget,
    )

    if not (
        budget_results["status"] == "budget_exhausted"
    ).any():
        budget = 5000

        mock_llm.reset()

        budget_results, budget_summary = run_triage(
            tickets,
            segments,
            ZERO_SHOT_PROMPT,
            "budget_demo",
            token_budget=budget,
        )

    budget_results.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False,
    )

    # Run 4: final few-shot classification with a large budget.
    mock_llm.reset()

    final_results, final_summary = run_triage(
        tickets,
        segments,
        FEW_SHOT_PROMPT,
        "final_few_shot",
        token_budget=100000,
    )

    final_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    final_results[final_columns].to_csv(
        OUTPUT_DIR / "triage_results.csv",
        index=False,
    )

    # Redacted text and PII audit columns only.
    final_results[
        [
            "ticket_id",
            "redacted_text",
            "emails_remaining",
            "phones_remaining",
            "order_ids_remaining",
        ]
    ].to_csv(
        OUTPUT_DIR / "tickets_redacted.csv",
        index=False,
    )

    summary = {
        "pii_audit": pii_audit,
        "zero_shot": zero_summary,
        "zero_shot_confusion_matrix": confusion,
        "few_shot": few_summary,
        "budget_demo": budget_summary,
        "final_few_shot": final_summary,
        "budget_used": budget,
    }

    with open(
        OUTPUT_DIR / "ticket_triage_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n=== PII Audit ===")
    print(json.dumps(pii_audit, indent=2))

    print("\n=== Ticket Triage Summary ===")
    print(json.dumps(summary, indent=2))

    print("\nSaved outputs:")
    print(OUTPUT_DIR / "triage_results.csv")
    print(OUTPUT_DIR / "tickets_redacted.csv")
    print(OUTPUT_DIR / "triage_zero_shot.csv")
    print(OUTPUT_DIR / "triage_few_shot.csv")
    print(OUTPUT_DIR / "triage_budget_demo.csv")
    print(OUTPUT_DIR / "triage_confusion_zero_shot.csv")
    print(OUTPUT_DIR / "ticket_triage_summary.json")


if __name__ == "__main__":
    main()
