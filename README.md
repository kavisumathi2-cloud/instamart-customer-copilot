# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data engineering, SQL analytics, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a simulated refund agent.

The project demonstrates how data and machine learning can help an e-commerce business understand customer behaviour, identify customers who may lapse, prioritize support tickets, and respond to refund requests using policy-based guardrails.

**Important:** This project uses synthetic data. The refund agent simulates refund decisions and does not transfer real money.

## Project Objectives

- Build and validate a structured data foundation using Python, pandas, and SQLite.
- Analyse orders and customer spending using SQL.
- Segment customers based on purchasing behaviour.
- Predict customers who may lapse using machine learning.
- Classify support tickets and route them to appropriate support teams.
- Retrieve relevant policy information for customer questions.
- Simulate refund decisions with eligibility checks and safety guardrails.
- Test key project behaviour and document the results.

## Technology Stack

- Python 3.11+
- SQLite 3.25+
- pandas
- NumPy
- scikit-learn
- joblib
- Python standard library
- Git and GitHub

## Project Structure

```text
instamart-customer-copilot/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
├── models/
├── outputs/
│   ├── customer_segments.csv
│   ├── kmeans_scores.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── tickets_redacted.csv
│   ├── ticket_triage_results.csv
│   ├── ticket_triage_summary.json
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── triage_confusion_zero_shot.csv
│   ├── triage_few_shot.csv
│   ├── policy_rag_results.csv
│   ├── policy_rag_overlap_comparison.csv
│   ├── policy_rag_summary.json
│   ├── refund_agent_demo.json
│   └── refund_agent_checkpoint.json
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
├── generate_data.py
├── mock_llm.py
├── README.md
└── requirements.txt
```

Note: This is the expected project structure. Actual files and folders may vary depending on which scripts have been run and which generated artifacts are committed.

## Reproduce

Run commands from the repository root using Python 3.11 or later. Install the dependencies listed in `requirements.txt` and activate the virtual environment before running the scripts.

Example setup on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run scripts in the order required by their input files:

```powershell
python generate_data.py
python src/clean_data.py
python src/build_database.py
python src/check_database.py
python src/sql_analysis.py
python src/customer_features.py
python src/customer_segmentation.py
python src/lapse_prediction.py
python src/ticket_triage.py
python src/policy_rag.py
python src/refund_agent.py
python -m pytest -q
```

Some scripts generate or overwrite files. Review the script configuration and output paths before rerunning the complete workflow.

The provided `generate_data.py` and `mock_llm.py` files are used as project dependencies and should remain unmodified.

## Part A — Data Engineering, SQL and EDA

### Data Foundation

Synthetic Instamart-style datasets are generated using `generate_data.py`. The project cleans the datasets, validates their contents, and loads them into a SQLite database.

The database contains customer, product, order, and support ticket information.

### Data Cleaning and Database

The cleaning process checks for duplicate records and prepares the datasets for analysis. The cleaned datasets are stored under `data/clean/`.

The previously verified database contained:

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,965 |
| Support tickets | 164 |

### SQL Analysis Results

The previously verified SQL analysis returned the following order status counts:

| Status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 656 |
| **Total** | **12,965** |

The previously reported total order amount was **₹3,465,352.30**.

These figures are from an earlier verified run. If the current database produces different results, rerun `src/sql_analysis.py` and update this section using the actual output from that run.

The SQL analysis also identifies customers with high order counts and high total spending. These aggregations demonstrate how SQL can support business reporting and customer analysis.

### Main Outputs

- Cleaned datasets under `data/clean/`
- SQLite database: `data/instamart.db`
- SQL analysis script: `src/sql_analysis.py`

## Part B — Customer Segmentation and Lapse Prediction

### Customer Segmentation

Customer features are calculated from order history, including order counts, completed spending, and order recency. Customers are grouped into behavioural segments to help the business understand differences in purchasing activity.

The latest documented segment counts are:

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

These segments can support targeted promotions, customer engagement, and support prioritisation. Segment names describe the project's configured rules and should not be interpreted as universal customer categories.

### Lapse Prediction

A machine learning model predicts whether a customer may lapse based on historical customer behaviour and a defined prediction target.

The project compares candidate models using cross-validation and selects a Random Forest model based on its F1 score.

The previously recorded evaluation metrics were:

| Metric | Score |
|---|---:|
| Accuracy | 0.737 |
| Lapsed-class precision | 0.659 |
| Lapsed-class recall | 0.652 |
| Lapsed-class F1 score | 0.655 |

These metrics describe a recorded evaluation run and may change when the data, feature definitions, or evaluation configuration changes.

The project also generates a retention list to help prioritise customers predicted to be at risk of lapsing.

**Evaluation limitation:** The lapse target uses a defined time window. Features must be restricted to information available before the prediction cutoff to avoid temporal leakage. Predictions are estimates, not guarantees that a customer will lapse.

### Main Outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage with Guardrails

### Ticket Classification

The ticket triage component classifies support tickets into billing, delivery, fraud, and technical categories. It uses zero-shot and few-shot classification workflows, validates model outputs, and applies keyword fallback handling when required.

Potentially sensitive information is redacted before ticket processing. The PII audit recorded 55 email addresses, 91 phone numbers, and 132 order IDs before redaction, with zero detected instances remaining after redaction.

### Classification Results

The final few-shot run processed 164 tickets.

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

Accuracy is calculated on the 147 valid classifications in the final few-shot run. It excludes tickets without a valid classification. The result applies to this synthetic dataset and evaluation configuration and is not a guarantee of real-world performance.

### Token Budget Evaluation

A separate budget demonstration used a token budget of 6,000. It used 5,951 tokens and made 80 model calls, producing 76 classifications before the remaining tickets were affected by budget exhaustion.

The measured accuracy for this budget run was approximately 84.21% on its valid classifications. This result should be considered separately from the final few-shot run.

### Routing and Guardrails

The system routes tickets to support queues based on classification and configured priority rules. Invalid outputs, empty tickets, long inputs, and token-budget exhaustion are tracked for monitoring and safer handling.

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

The recorded evaluation loaded 5 policy documents and indexed 15 overlapping word chunks, using a chunk size of 40 words, an overlap of 10 words, and top-3 retrieval.

| Evaluation | Result |
|---|---:|
| Total evaluation questions | 14 |
| Answerable questions | 12 |
| Answerable questions passed | 12 |
| Fallback questions | 2 |
| Fallback questions passed | 2 |
| Overall evaluation | 14/14 passed |

The overlap comparison showed that retrieval results depend on chunking and the number of retrieved chunks. In the recorded comparison, the overlapping strategy missed the 7-day condition at top-3, but retrieved all three required conditions at top-5. The no-overlap strategy retrieved all three conditions at top-3 in that experiment.

These results illustrate that overlap does not automatically guarantee better retrieval. Retrieved information must still be checked for relevance and completeness.

### Simulated Refund Agent

The refund agent follows a controlled workflow to evaluate refund requests against configured rules. These include order status, delivery delay, request timing, and the automatic refund limit.

The configured rules include:

- Late-delivery eligibility requires a completed order with a delivery time greater than 45 minutes.
- The refund request must be within the configured 7-day request window.
- The automatic refund limit is ₹1,000.
- Requests that fail eligibility checks are rejected or handled according to the workflow's guardrails.
- Eligible requests within the automatic limit can proceed through the simulated refund workflow.

The successful demonstration used order `ORD-110603`, with an order amount of ₹58, a delivery time of 70 minutes, and a request age of 3 days. The agent reported the status `simulated_refund_issued` after 4 steps and 4 LLM calls.

The result message was: `Simulated refund recorded. No real money was transferred.`

This is a simulation only. It does not connect to a payment gateway or transfer real money. The checkpoint file records the workflow state for inspection and recovery.

### Main Outputs

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/policy_rag_summary.json`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

## Testing

The project includes automated tests for important refund agent workflow and guardrail behaviour.

Run the tests from the project root:

```powershell
python -m pytest -q
```

The latest previously recorded test result was **12 tests passed**. Rerun the test command after making changes to verify the current state.

## Limitations and Next Steps

- The project uses synthetic data and does not represent actual Instamart customer records.
- SQL totals and model metrics depend on the current generated datasets and evaluation configuration.
- Customer segments depend on the configured features and segmentation rules.
- Lapse predictions are estimates and should not be treated as certain outcomes.
- Temporal leakage must be avoided by using only features available before the prediction cutoff.
- Ticket classification results may not generalize to real-world customer support messages.
- PII redaction depends on the patterns implemented and may not identify every possible sensitive value.
- Policy retrieval depends on the available documents, chunking strategy, and retrieval quality.
- The refund agent simulates decisions and does not issue actual refunds.
- Production use would require stronger monitoring, security controls, policy validation, and integration testing.

Possible future improvements include evaluating the system on additional datasets, expanding automated tests, improving ticket classification robustness, and monitoring retrieval quality across different chunking configurations.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates an end-to-end workflow combining data preparation, SQL analysis, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a guarded refund simulation.

The project highlights how analytics and machine learning can support customer retention and customer service while maintaining validation, testing, and clear safety boundaries.
