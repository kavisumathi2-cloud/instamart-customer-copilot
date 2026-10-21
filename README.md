# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone project demonstrates an end-to-end customer intelligence and support workflow using fictional Instamart-style data.

The project includes:

- Data cleaning, validation, and SQLite database creation
- SQL analytics
- Customer segmentation
- Customer lapse prediction
- Privacy-aware support ticket triage with guardrails
- Policy retrieval using TF-IDF
- A guarded refund-agent demonstration with simulated actions

**Important:** All data is synthetic. The LLM is mocked, and refund actions are simulations only. No real money is transferred.

## Project Parts

| Part | Description |
|---|---|
| Part A: Data Foundation | Clean and validate data, build SQLite tables, and run SQL analysis |
| Part B: Customer Intelligence | Build customer features, segment customers, and predict lapse risk |
| Part C: Support Ticket Triage | Redact personal information, classify tickets, and apply safety guardrails |
| Part D: Policy RAG and Refund Agent | Retrieve policy information and demonstrate guarded refund decisions |

## Technologies

- Python 3.11+
- SQLite 3.25+
- pandas
- NumPy
- scikit-learn
- joblib
- pytest

The supplied `generate_data.py` and `mock_llm.py` files must remain unmodified.

## Repository Structure

```text
instamart-customer-copilot/
|-- data/
|   |-- raw/
|   |-- clean/
|   |-- policies/
|   `-- instamart.db
|-- outputs/
|-- sql/
|   `-- queries.sql
|-- src/
|   |-- build_database.py
|   |-- check_database.py
|   |-- clean_data.py
|   |-- customer_features.py
|   |-- customer_segmentation.py
|   |-- inspect_data.py
|   |-- lapse_prediction.py
|   |-- policy_rag.py
|   |-- refund_agent.py
|   |-- sql_analysis.py
|   `-- ticket_triage.py
|-- state/
|-- tests/
|-- check_clean_data.py
|-- check_order_stats.py
|-- check_sql_queries.py
|-- generate_data.py
|-- mock_llm.py
|-- requirements.txt
`-- README.md
```

Generated files may be created in the `data`, `outputs`, and `state` folders when the scripts run.

## Setup and Reproduction

Run all commands from the repository root in Windows PowerShell.

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

### 3. Generate and prepare data

```powershell
python generate_data.py
python src/clean_data.py
python src/build_database.py
python src/sql_analysis.py
python src/customer_features.py
```

The original generator run reported:

```text
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

Running the generator again may overwrite generated datasets.

### 4. Run customer intelligence

```powershell
python src/customer_segmentation.py
python src/lapse_prediction.py
```

### 5. Run ticket triage and policy/refund workflows

```powershell
python src/ticket_triage.py
python src/policy_rag.py
python src/refund_agent.py
```

### 6. Run validation checks and tests

```powershell
python check_clean_data.py
python src/check_database.py
python check_sql_queries.py
python check_order_stats.py
python -m pytest -v
```

A recorded test run completed with **14 tests passed**. Results may change if code, data, configuration, or dependencies change.

## Part A: Data Foundation

The synthetic data generator creates customers, products, orders, tickets, policy-evaluation questions, and refund-agent scenarios.

Recorded cleaned-data counts:

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

The cleaned orders dataset contains 11,110 completed orders and 1,199 cancelled orders. Pending orders are excluded from the final cleaned orders dataset.

The cleaning workflow includes timestamp parsing, duplicate handling, amount validation and imputation, delivery-time validation, customer-field standardisation, and removal of pending orders.

Database checks cover row counts, unique customer/product/order IDs, valid order statuses, customer references, missing customer IDs, and negative order amounts. The latest local validation reported **ALL CHECKPOINTS PASSED**.

The seven queries in `sql/queries.sql` demonstrate:

- Customer order and spending summaries
- Revenue by city and product category
- Products priced above average
- Customers without completed orders
- Customers spending above average
- Ranking with window functions
- Daily and cumulative revenue

## Part B: Customer Segmentation and Lapse Prediction

### Customer segmentation

Customer features follow an RFM-style approach:

- **Recency:** Days since the last completed order
- **Frequency:** Number of completed orders
- **Monetary:** Total spending on completed orders

The RFM snapshot date is 1 October 2025. Customers without completed orders are assigned a separate segment.

Recorded segment distribution:

| Segment | Customers |
|---|---:|
| Regular Customers | 436 |
| High Value | 419 |
| Low Spend | 183 |
| Top Value | 137 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

K-means was evaluated for cluster counts from 2 to 8. In the recorded comparison, K=4 had the highest silhouette score of 0.525.

### Lapse prediction

The lapse label identifies eligible customers with no completed order during September 2025. The feature cutoff is 1 September 2025. The recorded eligible cohort contains 1,156 customers.

Recorded model comparison:

| Model | Accuracy | Lapsed Precision | Lapsed Recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.728 | 0.64 | 0.65 | 0.65 |
| Random Forest | 0.737 | 0.66 | 0.65 | 0.66 |

Random Forest was selected based on cross-validation F1 in the recorded run. Preprocessing is fitted on the training split, and model features use historical data before the cutoff.

**Evaluation caveat:** The retention list is illustrative, not a fully leakage-free evaluation, because the premium flag is derived from whole-window segmentation. Metrics apply to the supplied synthetic data and evaluation split.

Main outputs include:

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C: Support Ticket Triage with Guardrails

The ticket pipeline processes 164 synthetic tickets and demonstrates:

- Empty-text and long-input checks
- Redaction of email addresses, phone numbers, and order IDs
- Zero-shot and few-shot classification into `billing`, `delivery`, `fraud`, and `technical`
- JSON output validation and keyword fallback
- Retry handling for transient model errors
- Token-budget controls
- Customer-aware routing and human review for unsafe or rejected cases

Recorded synthetic evaluation results:

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets in comparison | 147 | 147 |
| Accuracy | 85.7% | 100% |
| Total tokens | 11,584 | 22,660 |
| Tokens per classified ticket | 78.80 | 154.15 |

The separate token-budget demonstration used a budget of 6,000 tokens and recorded 5,951 tokens across 80 model calls. These results apply to the mock model and synthetic tickets, not real support traffic.

Routing examples include human review for guardrail-rejected cases, high-priority security routing for fraud, high-priority routing for premium customers, and normal support routing for other eligible tickets.

Main outputs include:

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`
- `outputs/triage_results.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_confusion_zero_shot.csv`
- `outputs/triage_budget_demo.csv`

## Part D: Policy RAG Assistant and Refund Agent

### Policy retrieval

The project uses five fictional policy documents covering refunds, delivery, payments, privacy, and escalation.

Policy passages are retrieved using TF-IDF and n-gram similarity. The main evaluation uses top-K retrieval with K=3.

The recorded evaluation contains 14 questions: 12 answerable questions and two unanswerable questions. In the summary evaluation, all 12 answerable questions passed and both unanswerable questions used the fallback.

Example fictional policy rules include:

- A late-delivery refund requires a completed order delivered in more than 45 minutes.
- The request must be made within seven days of delivery.
- The failed-payment workflow uses a 48-hour request window.
- Approved refunds are described as returning to the original payment method within five working days.
- Payment-card numbers are not stored by the workflow.

These are fictional project rules, not actual Instamart policy statements.

### Guarded refund-agent demonstration

The agent demonstrates a controlled workflow:

1. Validate the request and order identifier.
2. Retrieve order details from SQLite.
3. Retrieve relevant policy information.
4. Check eligibility with deterministic Python rules.
5. Validate refund amount and approval requirements.
6. Simulate an eligible refund, deny an ineligible request, or escalate the case.
7. Record execution and checkpoint information.

Safety controls include tool allowlisting, order-ID validation, order-status checks, refund-amount validation, a seven-day request window, an automatic refund limit of INR 1,000, approval checks, repeated-action detection, execution limits, and checkpoint/recovery support.

**No real refund is issued.** The project does not connect to a payment gateway or live customer account. A production implementation would require stronger transactional guarantees, authentication, authorisation, durable auditing, and duplicate-action protection.

Main outputs include:

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_summary.json`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`
- `outputs/refund_agent_scenarios.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

Crash/resume evidence files are maintained separately in `state/`.

## Validation Summary

Recorded validation checks include:

- Cleaned-data validation passed
- Database validation passed
- SQL query checks passed
- Automated test run: 14 tests passed

Rerun the validation commands against the current checkout before submission if any project files have changed.

## Limitations

- The datasets are synthetic and do not represent real Instamart customers.
- The LLM is mocked; behaviour may differ from a live model.
- Model metrics depend on synthetic labels, feature construction, preprocessing, and the test split.
- Feature importance and model coefficients describe model behaviour; they do not prove causation.
- Ticket-triage accuracy and token counts apply only to the evaluated mock-LLM runs.
- TF-IDF retrieval can miss relevant context or return incomplete passages.
- The RAG evaluation set is small and does not establish broad real-world reliability.
- Refund actions are simulated; no payment transaction occurs.
- Automated tests cover selected cases, not every possible production failure.

## Conclusion

This project combines data engineering, SQL, customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and a guarded refund-agent workflow in one educational repository.

It is intended for capstone evaluation and learning, not for production customer support or real financial transactions.
