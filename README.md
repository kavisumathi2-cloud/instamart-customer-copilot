# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data engineering, SQL analytics, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a simulated refund agent.

The project demonstrates how data and machine learning can help an e-commerce business understand customer behaviour, identify customers who may lapse, prioritize support tickets, and respond to refund requests using policy-based guardrails.

**Important:** This project uses synthetic data. The refund agent simulates refund decisions and does not transfer real money.

## Project Objectives

- Build and validate a structured data foundation using Python, pandas, and SQLite.
- Analyse customer orders and spending using SQL.
- Segment customers based on purchasing behaviour.
- Predict customers who may lapse using machine learning.
- Classify support tickets and apply routing and safety guardrails.
- Retrieve relevant policy information for customer questions.
- Simulate refund decisions with eligibility checks and safety controls.
- Test important project behaviour and document evaluation results.

## Technology Stack

- Python 3.11+
- SQLite 3.25+
- pandas
- NumPy
- scikit-learn
- joblib
- Python standard library
- Git and GitHub
- pytest

## Project Structure

```text
instamart-customer-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── sql/
│   └── queries.sql
├── src/
│   ├── build_database.py
│   ├── check_database.py
│   ├── clean_data.py
│   ├── inspect_data.py
│   ├── sql_analysis.py
│   ├── customer_features.py
│   ├── customer_segmentation.py
│   ├── lapse_prediction.py
│   ├── ticket_triage.py
│   ├── policy_rag.py
│   └── refund_agent.py
├── tests/
│   └── test_refund_agent.py
├── outputs/
├── state/
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `outputs/` directory contains generated datasets, model artifacts, evaluation results, and demonstration logs. The `state/` directory stores refund-agent workflow checkpoints when generated. Runtime-generated files may vary depending on which scripts have been executed.

## Setup and Reproduction

Run commands from the repository root using Python 3.11 or later.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this in the same terminal and then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Generate and prepare the data

```powershell
python generate_data.py
python src/clean_data.py
python src/build_database.py
python src/check_database.py
```

### 5. Run SQL analysis and customer analytics

```powershell
python src/sql_analysis.py
python src/customer_features.py
python src/customer_segmentation.py
python src/lapse_prediction.py
```

### 6. Run support ticket triage and policy retrieval

```powershell
python src/ticket_triage.py
python src/policy_rag.py
```

### 7. Run the refund-agent demonstration

```powershell
python src/refund_agent.py
```

### 8. Run automated tests

```powershell
python -m pytest -q
```

The most recently verified test run completed with **12 passed**.

**Reproduction note:** Some scripts create or overwrite generated files. Running the complete workflow again can change dataset counts, evaluation results, or generated artifacts if the data or configuration differs. Check the current outputs after rerunning the workflow. The provided `generate_data.py` and `mock_llm.py` files are used as project dependencies and should remain unmodified.

## Part A — Data Engineering, SQL and EDA

### Data Foundation

Synthetic Instamart-style datasets are generated using `generate_data.py`. The project cleans the datasets, validates their contents, and loads them into a SQLite database.

The datasets cover customers, products, orders, and support tickets.

### Data Cleaning and Database

The data-cleaning workflow checks duplicate records, handles invalid values, and prepares the data for analysis. The cleaned CSV files are stored under `data/clean/`.

The latest verified cleaned-data counts were:

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

The raw order data contained duplicate records and pending orders. The cleaning process removed duplicate records and excluded pending orders from the cleaned order dataset.

### SQL Analysis

SQL queries are used to analyse customer spending, order activity, product performance, and order trends.

The verified cleaned order dataset contained:

| Order status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| **Total cleaned orders** | **12,309** |

The previously verified completed-order spending total was **₹2,998,668.80**.

The SQL analysis includes:

- Customers ranked by total spending.
- Customer order counts and spending summaries.
- Product and category performance.
- Customers without completed orders.
- Date-based order analysis and cumulative spending.

These aggregations demonstrate how SQL can support business reporting and customer analysis.

### Main Outputs

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

## Part B — Customer Segmentation and Lapse Prediction

### Customer Segmentation

Customer features are calculated from order history, including order frequency, completed spending, and recency.

K-Means clustering is evaluated using silhouette scores. PCA is also used to examine feature variation in a lower-dimensional representation.

The latest verified segmentation run produced:

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The premium-customer flag identifies customers selected for additional support prioritisation according to the configured segmentation rules.

These segments can help businesses design customer engagement strategies and prioritize support. Segment names describe the project's configured rules and should not be interpreted as universal customer categories.

### Lapse Prediction

The lapse-prediction component uses historical customer behaviour to estimate whether a customer may lapse within a defined target window.

The workflow compares candidate models using cross-validation and selects a Random Forest model based on its cross-validation F1 score.

The previously recorded evaluation metrics were:

| Metric | Score |
|---|---:|
| Accuracy | 0.737 |
| Lapsed-class precision | 0.659 |
| Lapsed-class recall | 0.652 |
| Lapsed-class F1 score | 0.655 |

Precision indicates how many customers predicted as lapsed were actually lapsed. Recall indicates how many actual lapsed customers were identified. The F1 score balances precision and recall.

The project also generates a retention list to help prioritize customers predicted to be at risk of lapsing.

**Evaluation limitation:** Lapse predictions depend on the selected cutoff date, target window, features, and evaluation split. Features should only use information available before the prediction cutoff to prevent temporal leakage. The retention list is an illustrative prioritisation output, not a guarantee of successful customer retention.

### Main Outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage with Guardrails

### Ticket Classification

The ticket-triage component classifies support tickets into categories such as billing, delivery, fraud, and technical issues.

It evaluates zero-shot and few-shot classification workflows, validates model outputs, and applies keyword-based fallback handling when required.

Potentially sensitive information is redacted before ticket processing.

### PII Redaction Results

The recorded audit found the following values before and after redaction:

| PII type | Detected before redaction | Detected after redaction |
|---|---:|---:|
| Email addresses | 55 | 0 |
| Phone numbers | 91 | 0 |
| Order IDs | 132 | 0 |

These results reflect the patterns checked by the implemented audit. They do not guarantee that every possible sensitive value will be detected.

### Few-Shot Classification Results

The recorded final few-shot evaluation processed 164 tickets.

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| JSON classifications | 135 |
| Keyword fallback classifications | 12 |
| Valid classifications | 147 |
| Invalid outputs | 9 |
| Empty tickets | 4 |
| Long tickets flagged | 4 |
| Model calls | 156 |
| Tokens used | 22,660 |
| Accuracy on valid classifications | 100% |

Accuracy is calculated on the 147 valid classifications and excludes tickets without a valid classification. The result applies to the synthetic dataset and evaluation configuration; it is not a guarantee of real-world performance.

### Token Budget Evaluation

A separate budget demonstration used a token budget of 6,000.

| Metric | Result |
|---|---:|
| Token budget | 6,000 |
| Tokens used | 5,951 |
| Model calls | 80 |
| Valid classifications | 76 |
| Accuracy on valid classifications | Approximately 84.21% |

The budget-limited result is evaluated separately from the final few-shot run. It demonstrates how token budgets can limit the number of tickets processed.

### Routing and Guardrails

The workflow tracks invalid model outputs, empty tickets, long inputs, classification fallbacks, and token-budget exhaustion. Configured classification and priority rules help route tickets to appropriate support queues.

### Main Outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_confusion_zero_shot.csv`
- `outputs/triage_few_shot.csv`

## Part D — Policy RAG Assistant and Refund Agent

### Policy Retrieval

The policy assistant loads policy documents, splits them into word chunks, and retrieves relevant information using TF-IDF similarity. Deterministic policy rules handle recognized policy questions, while a fallback response is used when relevant policy information cannot be found.

The recorded evaluation used:

| Setting | Result |
|---|---:|
| Policy documents | 5 |
| Indexed chunks | 15 |
| Chunk size | 40 words |
| Chunk overlap | 10 words |
| Default retrieved chunks | 3 |
| Evaluation questions | 14 |
| Answerable questions passed | 12/12 |
| Fallback questions passed | 2/2 |
| Overall evaluation | 14/14 |

### Retrieval Comparison

The overlap comparison demonstrated that retrieval results depend on chunking and the number of retrieved chunks.

In the recorded comparison, the overlapping strategy with top-3 retrieval missed the 7-day request condition. Increasing retrieval to top-5 retrieved all three required conditions. The no-overlap strategy retrieved all three conditions at top-3 in that experiment.

This demonstrates that overlap does not automatically guarantee better retrieval. Retrieved information must still be checked for relevance and completeness.

### Simulated Refund Agent

The refund agent follows a controlled workflow to evaluate refund requests against configured rules.

The configured guardrails include:

- Late-delivery eligibility requires a completed order with a delivery time greater than 45 minutes.
- The refund request must be within the configured 7-day request window.
- The automatic refund limit is ₹1,000.
- Requests that fail eligibility checks are rejected or handled according to the workflow's guardrails.
- Eligible requests within the automatic refund limit can proceed through the simulated refund workflow.
- Workflow state is checkpointed to support inspection and recovery.

### Successful Demonstration

The recorded successful demo used the following details:

| Field | Result |
|---|---|
| Order ID | ORD-110603 |
| Order amount | ₹58.00 |
| Actual delivery time | 70 minutes |
| Request age | 3 days |
| Agent status | `simulated_refund_issued` |
| Workflow steps | 4 |
| LLM calls | 5 |
| Final answer | `REFUND: INR 58.0 issued.` |

The demo order met the configured late-delivery, request-window, and automatic refund-limit conditions.

A request made after the configured 7-day window can be rejected. In a separate recorded check, the same demo order was rejected when the request age was 8 days.

### Scenario and Recovery Testing

The seven recorded refund scenarios matched their expected outcomes and refund amounts. The project also tested token-limit and timeout guardrails.

A simulated crash-and-resume test confirmed that the agent could continue from a saved checkpoint without repeating the initial order-lookup tool call in that test.

**Safety boundary:** Refunds are simulated only. The agent does not connect to a payment gateway or transfer real money. Its outputs demonstrate workflow behaviour rather than production payment processing.

### Main Outputs

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/policy_rag_summary.json`
- `outputs/refund_agent_demo.json`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

Runtime checkpoint files may be stored in `state/` or other configured checkpoint paths. Their presence depends on the scripts executed and the repository's file-tracking configuration.

## Testing

The project includes automated tests for important refund-agent workflow and guardrail behaviour.

Run the test suite from the repository root:

```powershell
python -m pytest -q
```

The latest verified run completed with:

```text
12 passed
```

The test suite covers refund-agent behaviour. Additional automated tests would be needed to comprehensively validate every project component.

## Limitations and Next Steps

- The project uses synthetic data and does not represent actual Instamart customer records.
- SQL totals and model metrics depend on the current generated datasets and evaluation configuration.
- Customer segments depend on the configured features and segmentation rules.
- Lapse predictions are estimates and should not be treated as certain outcomes.
- Temporal leakage must be avoided by using only features available before the prediction cutoff.
- Ticket classification results may not generalize to real-world customer support messages.
- PII redaction depends on implemented detection patterns and may not identify every possible sensitive value.
- Policy retrieval depends on the available documents, chunking strategy, and retrieval quality.
- The refund agent simulates decisions and does not issue actual refunds.
- Production use would require stronger monitoring, security controls, policy validation, and integration testing.
- Recorded evaluation results should be regenerated when data or configurations change.

Possible future improvements include evaluating the system on additional datasets, expanding automated tests, improving ticket-classification robustness, improving retrieval evaluation, and monitoring model performance over time.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates an end-to-end workflow combining data preparation, SQL analysis, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a guarded refund simulation.

The project highlights how analytics and machine learning can support customer retention and customer service while maintaining validation, testing, and clear safety boundaries.