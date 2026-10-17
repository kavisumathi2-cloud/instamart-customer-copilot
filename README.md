# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data cleaning, SQL analytics, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a simulated refund agent.

The project demonstrates a complete data and AI workflow using Python, pandas, SQLite, scikit-learn, TF-IDF, and a mock LLM.

**Important:** This is an educational prototype. The data is synthetic, the LLM is simulated, and refund processing does not transfer real money.

## Reproduce

### Requirements

- Python 3.11+
- VS Code or another Python IDE
- A virtual environment
- The supplied `generate_data.py` and `mock_llm.py` files

### Setup

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages if they are not already installed:

```powershell
pip install pandas numpy scikit-learn joblib
```

Generate the synthetic dataset:

```powershell
python generate_data.py
```

Run the project scripts from the repository root in this order:

```powershell
python src/clean_data.py
python src/build_database.py
python src/sql_analysis.py
python src/customer_features.py
python src/customer_segmentation.py
python src/lapse_prediction.py
python src/ticket_triage.py
python src/policy_rag.py
python src/refund_agent.py
```

The scripts create cleaned datasets, a SQLite database, trained model artifacts, evaluation outputs, ticket triage results, policy retrieval results, and a refund-agent demonstration.

## Part A — Data Engineering, SQL & EDA

### Work completed

- Generated synthetic customers, products, orders, and support tickets.
- Cleaned and validated source datasets.
- Saved cleaned CSV files under `data/clean/`.
- Built a SQLite database at `data/instamart.db`.
- Created customer-level RFM features.
- Performed SQL analysis and exploratory data analysis.

### Data summary

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Support tickets | 164 |

The cleaned order dataset contains 11,110 completed orders and 1,199 cancelled orders.

### Main outputs

- `data/clean/`
- `data/instamart.db`
- `data/clean/customer_features.csv`

## Part B — Customer Segmentation & Lapse Prediction

### Customer segmentation

RFM features represent:

- **Recency:** Time since the customer's last completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total completed-order spending.

K-means clustering was used to identify customer groups.

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |

The premium-customer flag identifies the selected high-value segment for downstream support prioritization.

### Lapse prediction

Two classifiers were evaluated:

- Logistic Regression
- Random Forest

The Random Forest model was selected based on cross-validation F1 performance.

| Metric | Random Forest |
|---|---:|
| Test accuracy | 0.737 |
| Test F1 for lapsed customers | 0.655 |
| Precision for lapsed customers | 0.659 |
| Recall for lapsed customers | 0.652 |

Frequency and monetary value were among the most influential features in the trained Random Forest model.

Class balancing was also explored to assess the trade-off between recall and precision.

### Main outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage

### Work completed

- Masked email addresses and phone numbers before sending ticket text to the mock LLM.
- Classified tickets into billing, delivery, technical, and fraud categories.
- Added retry handling for transient failures.
- Validated model output and category names.
- Enforced an LLM-call budget.
- Routed tickets using category and customer-segment information.
- Recorded fallback classifications and evaluation metrics.

### Results

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| Valid classifications | 143 |
| Fallback classifications | 21 |
| Accuracy on valid classifications | 84.89% |
| Accuracy including fallback predictions | 76.22% |
| LLM calls used | 184 of 500 |

These metrics describe this synthetic dataset and mock LLM only. They should not be interpreted as expected real-world production performance.

### Main outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`

## Part D — Policy RAG & Refund Agent

### Policy retrieval

Policy documents cover refunds, delivery, payments, privacy, and escalation.

The retrieval workflow uses TF-IDF and n-gram similarity to find relevant policy sentences. Deterministic rules handle selected high-impact questions, such as late-delivery refunds and payment issues.

Seven test questions returned the expected answers in the initial evaluation. This small test set does not establish general policy-answering accuracy.

### Refund agent

The refund agent demonstrates a ReAct-style workflow:

1. Retrieve the order.
2. Search relevant policy documents.
3. Calculate refund eligibility using Python rules.
4. Simulate a refund or escalate for human review.

Safety controls include an allowlisted tool dispatcher, order-status checks, refund amount verification, an automatic refund limit, repeated-action detection, and step/call limits.

### Demonstration tests

**Ineligible order**

- Order: `ORD-110093`
- Delivery time: 19 minutes
- Outcome: `rejected`
- Refund amount: ₹0

**Eligible order**

- Order: `ORD-110603`
- Delivery time: 70 minutes
- Order amount: ₹58
- Outcome: `simulated_refund_issued`
- Tool steps: 4

No real money was transferred. Human approval is simulated rather than connected to an actual approval system.

### Main outputs

- `outputs/policy_rag_results.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

## Limitations and Next Steps

### Limitations

- All datasets are synthetic.
- `mock_llm.py` simulates LLM behavior and is not a production language model.
- Ticket classifications and policy responses can be incorrect.
- TF-IDF retrieval uses lexical similarity and may miss semantically relevant policy text.
- The lapse model's metrics depend on the synthetic data and selected time window.
- The customer premium flag and retention list should be reviewed for temporal leakage before production use.
- The refund agent is a demonstration, not a payment integration.
- JSON checkpointing provides basic saved progress, not production-grade crash recovery or exactly-once execution.
- Human approval and escalation are simulated.

### Next steps

- Evaluate models on real, permissioned, representative data.
- Add stronger temporal validation and leakage checks.
- Improve policy retrieval with a more comprehensive evaluation set.
- Route uncertain or invalid ticket classifications to human review.
- Add automated unit tests for refund eligibility, tool validation, and failure handling.
- Implement durable checkpoint recovery, audit logging, and idempotency before any real refund integration.
- Add privacy, access-control, and monitoring safeguards before production deployment.
