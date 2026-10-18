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
│   ├── policy_rag_results.csv
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

*Note: This structure documents the expected project files. Generated files and folders may vary depending on which scripts have been run.*

## Part A — Data Engineering, SQL and EDA

### Data Foundation

Synthetic Instamart-style datasets are generated using the provided `generate_data.py` script. The project cleans the datasets, validates their contents, and loads them into a SQLite database.

The database contains customer, product, order, and support ticket information.

### Data Cleaning and Database

The cleaning process checks for duplicate records and prepares the datasets for analysis. The cleaned datasets are stored under `data/clean/`.

The current SQLite database was verified to contain:

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

### SQL Analysis Results

The SQL analysis examines order status, total order amount, customer order counts, and customer spending.

**Order status summary**

| Status | Orders |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| **Total** | **12,309** |

The current status query returned completed and cancelled orders. No pending-order rows were returned in this database.

**Total order amount:** ₹3,332,000.55

The analysis also identifies customers with high order counts and customers with high total spending. These results demonstrate how SQL can aggregate transaction data to support business decisions.

### Main Outputs

- Cleaned datasets under `data/clean/`
- SQLite database: `data/instamart.db`
- SQL analysis script: `src/sql_analysis.py`

## Part B — Customer Segmentation and Lapse Prediction

### Customer Segmentation

Customer features are calculated from order history, including completed order counts, completed spending, and order recency. Customers are grouped into behavioural segments to help the business understand differences in purchasing activity.

The current customer segment counts are:

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

These segments can support targeted promotions, customer engagement, and support prioritisation. Segment names describe the project's rules and should not be interpreted as universal customer categories.

### Lapse Prediction

A machine learning model predicts whether a customer may lapse based on historical customer behaviour and a defined prediction target.

The project compares candidate models using cross-validation and selects a Random Forest model based on its F1 score.

**Reported evaluation metrics**

| Metric | Score |
|---|---:|
| Accuracy | 0.737 |
| Lapsed-class precision | 0.659 |
| Lapsed-class recall | 0.652 |
| Lapsed-class F1 score | 0.655 |

These metrics describe the current evaluation run. Model performance may vary with the data, feature definitions, and evaluation method.

The project also generates a retention list to help prioritise customers predicted to be at risk of lapsing.

**Evaluation limitation:** The lapse target uses a defined time window. Features must be restricted to information available before the prediction cutoff to avoid temporal leakage. The retention list should be treated as a prioritisation aid, not a guarantee that a customer will lapse.

### Main Outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage with Guardrails

### Ticket Classification

The ticket triage component processes support ticket text and assigns categories using the project's classification workflow. It also applies validation and fallback handling when a classification is invalid or cannot be accepted.

Ticket text is redacted in the exported dataset to reduce exposure of potentially sensitive information.

### Current Run Results

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| Valid classifications | 143 |
| Fallback classifications | 21 |
| Tickets retried | 20 |
| LLM calls | 184 of 500 |
| Accuracy on valid classifications | 84.89% |

Accuracy is calculated only on valid classifications with recognized ground-truth labels. In the current run, 143 classifications were valid, and the reported accuracy was 84.89%.

The system routes tickets to support queues based on their classifications. Fraud-related tickets are directed to urgent fraud review, while other categories are routed to their corresponding support teams.

### Main Outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`

## Part D — Policy RAG Assistant and Refund Agent

### Policy Retrieval

The policy assistant retrieves relevant policy information to help answer customer questions. The evaluation output records the questions and retrieval results for inspection.

Retrieval helps ground responses in the available policy documents. However, retrieved text should still be checked for relevance and accuracy before being used to make a decision.

### Simulated Refund Agent

The refund agent follows a controlled workflow to evaluate refund requests against configured rules. These include order status, delivery delay, request timing, and the automatic refund limit.

The current project rules include:

- Late-delivery eligibility requires a completed order with a delivery time greater than 45 minutes.
- The refund request must be within the configured 7-day request window.
- The automatic refund limit is ₹1,000.
- Requests that fail eligibility checks are rejected or handled according to the workflow's guardrails.

The refund agent uses a simulated workflow. A successful simulated refund does **not** mean that real money has been transferred.

The demonstration outputs include a successful simulated refund scenario and a checkpoint file that records the workflow state.

### Main Outputs

- `outputs/policy_rag_results.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

## Testing

The refund agent includes automated tests for important workflow and guardrail behaviour.

Run the tests from the project root:

```bash
python -m pytest -q
```

**Latest recorded test result:** 12 tests passed.

This result reflects the test run performed during project verification. Run the command again after making code changes to confirm the current state.

## Running the Project

Run commands from the repository root, with the project's virtual environment activated and dependencies installed.

Example workflow:

```bash
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

Run scripts in the order required by their input files. Some scripts may overwrite generated data or output files. Review the script configuration and output paths before rerunning the complete workflow.

The provided `generate_data.py` and `mock_llm.py` files are used as project dependencies and should remain unmodified.

## Limitations

- The project uses synthetic data and does not represent actual Instamart customer records.
- Machine learning metrics are specific to the current dataset and evaluation configuration.
- Customer segments are rule- and feature-dependent.
- Lapse predictions are estimates and should not be treated as certain outcomes.
- Policy retrieval depends on the available documents and retrieval quality.
- The refund agent simulates decisions; it does not connect to a real payment gateway or issue actual refunds.
- Ticket classification and policy responses require validation before use in a real customer-support environment.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates an end-to-end workflow combining data preparation, SQL analysis, customer segmentation, lapse prediction, ticket triage, policy retrieval, and a guarded refund simulation. It highlights how analytics and machine learning can support customer retention and customer service while maintaining validation, testing, and clear safety boundaries.