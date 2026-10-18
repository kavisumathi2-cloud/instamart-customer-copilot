# Instamart Customer Intelligence & Support Copilot

## Overview

The Instamart Customer Intelligence & Support Copilot is an end-to-end data analytics and AI workflow built using synthetic Instamart-style e-commerce data.

The project demonstrates how data engineering, SQL analytics, machine learning, support-ticket classification, policy retrieval, and rule-based safeguards can support customer intelligence and customer service workflows.

The project covers four major areas:

- **Part A — Data Engineering, SQL and EDA:** Data generation, cleaning, database construction, SQL analysis, and exploratory data analysis.
- **Part B — Customer Segmentation and Lapse Prediction:** Customer-level RFM features, K-means clustering, and lapse prediction using classification models.
- **Part C — Support Ticket Triage:** Ticket classification, privacy-aware text masking, retries, validation, fallback handling, and routing.
- **Part D — Policy RAG and Refund Agent:** Policy retrieval, rule-based refund eligibility checks, controlled tool execution, and simulated refund processing.

**Important:** This is an educational prototype. All datasets are synthetic, the LLM behavior is simulated, and no real refund or payment transaction is performed.

## Technologies Used

- Python 3.11+
- pandas and NumPy
- SQLite
- scikit-learn
- TF-IDF and n-gram similarity
- joblib
- Python `unittest` and pytest for automated testing
- Git and GitHub

## Project Structure

```text
instamart-customer-copilot/
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
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
│   ├── customer_features.py
│   ├── customer_segmentation.py
│   ├── inspect_data.py
│   ├── lapse_prediction.py
│   ├── policy_rag.py
│   ├── refund_agent.py
│   ├── sql_analysis.py
│   └── ticket_triage.py
├── tests/
│   ├── __init__.py
│   └── test_refund_agent.py
├── generate_data.py
├── mock_llm.py
├── .gitignore
└── README.md
```

Generated files and directories may vary depending on which scripts have been executed. The supplied `generate_data.py` and `mock_llm.py` files should remain unmodified.

## Project Setup and Reproduction

### Requirements

- Python 3.11 or newer
- A virtual environment
- VS Code or another Python IDE
- The supplied `generate_data.py` and `mock_llm.py` files

### 1. Create and activate a virtual environment

Run the commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install pandas numpy scikit-learn joblib
```

### 3. Generate synthetic data

```powershell
python generate_data.py
```

The script generates synthetic customer, product, order, and support-ticket datasets.

### 4. Run the project pipeline

Execute the scripts from the repository root in the documented order:

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

Some scripts may overwrite generated datasets, model artifacts, or evaluation outputs when rerun. Use the same environment and follow the documented order when reproducing results.

### 5. Run automated tests

```powershell
python -m unittest discover -s tests -v
```

Alternatively, if pytest is installed:

```powershell
python -m pytest -q
```

The recorded pytest run completed with **12 passing tests**.

## Part A — Data Engineering, SQL and EDA

### Implementation

- Generated synthetic customer, product, order, and support-ticket datasets.
- Cleaned and validated the source datasets.
- Removed duplicate order records during cleaning.
- Built a SQLite database for structured analysis.
- Created customer-level features using completed-order history.
- Performed SQL aggregations and exploratory data analysis.

### Dataset Summary

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Support tickets | 164 |

The project database was verified to contain 1,200 customers, 40 products, 12,309 orders, and 164 tickets.

The project summary reports 11,110 completed orders, 1,199 cancelled orders, and 656 pending orders.

### SQL Analysis

The analysis includes:

- Order counts grouped by status
- Total order amount
- Order frequency by customer
- Highest-spending customers
- Order amounts grouped by status

These queries demonstrate how relational data can be transformed into useful operational summaries.

### Main Outputs

- Cleaned datasets under `data/clean/` or `data/cleaned/`, depending on the script
- `data/instamart.db`
- Customer-level feature data generated by the feature-engineering script

## Part B — Customer Segmentation and Lapse Prediction

### Customer Segmentation

Customer-level features represent three common RFM concepts:

- **Recency:** Time since the customer's most recent completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total spending on completed orders.

K-means clustering was used to identify customer groups. Silhouette scores were used to evaluate clustering configurations. Customers without completed-order history were represented separately in the final segmentation output.

### Segment Summary

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The segment labels provide a descriptive view of customer behavior. The premium-customer flag supports downstream support prioritization, but segment definitions would require business validation before operational use.

### Lapse Prediction

Two classification approaches were evaluated:

- Logistic Regression
- Random Forest

The Random Forest model was selected based on cross-validation F1 performance.

The lapse prediction workflow defines a historical cutoff and a September observation window. Customers with no completed order during the label window are classified as lapsed for this experiment.

Features are constructed from the historical period before the cutoff, and the dataset is split into training and testing sets using stratification.

### Random Forest Evaluation

| Metric | Result |
|---|---:|
| Test accuracy | 0.737 |
| F1 score for lapsed customers | 0.655 |
| Precision for lapsed customers | 0.659 |
| Recall for lapsed customers | 0.652 |

Frequency and monetary value were among the influential features in the trained model. Class balancing was also explored to assess the trade-off between precision and recall.

The retention list contains the top 10 premium customers from the test set, ranked by predicted lapse probability. It is an illustrative prioritization output, not a list of customers guaranteed to lapse.

**Potential limitation:** The premium-customer flag is derived from whole-window RFM information. This creates a potential temporal leakage concern for the retention-list analysis. The list should therefore be treated as illustrative rather than a fully leakage-free evaluation.

These metrics apply to the selected synthetic dataset, label definition, and evaluation setup. They should not be interpreted as evidence of expected production performance.

### Main Outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage

### Implementation

The support-ticket workflow demonstrates how incoming messages can be classified and routed while applying basic safeguards.

Implemented features include:

- Masking email addresses and phone numbers before sending ticket text to the mock LLM
- Classification into billing, delivery, technical, and fraud categories
- Retry handling for transient failures
- Validation of returned categories and model output
- An LLM-call budget
- Fallback classifications when valid output is unavailable
- Routing using ticket categories and customer-segment information
- Evaluation of classification results

### Results

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| Valid classifications | 143 |
| Fallback classifications | 21 |
| Retry attempts | 20 |
| Accuracy on valid classifications | 84.89% |
| LLM calls used | 184 of 500 |

The saved evaluation summary reports 139 evaluated tickets for the accuracy calculation. The exact denominator and evaluation rules should be checked in the evaluation code before interpreting this percentage.

These results describe this synthetic dataset and mock LLM only. They do not establish expected real-world classification accuracy, privacy compliance, or production readiness.

### Main Outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`

## Part D — Policy RAG and Refund Agent

### Policy Retrieval

The policy corpus covers five areas:

- Refunds
- Delivery
- Payments
- Privacy
- Escalation

The retrieval workflow uses TF-IDF and n-gram similarity to identify relevant policy sentences. Deterministic rules handle selected high-impact questions, including late-delivery refunds and payment issues.

In the initial evaluation, seven test questions returned their expected answers. This small test set is useful as an initial check but does not establish general policy-answering accuracy.

### Refund Agent

The refund agent demonstrates a ReAct-style workflow combining information retrieval, tool calls, and rule-based decisions.

The workflow follows four main steps:

1. Retrieve the relevant order.
2. Search relevant policy information.
3. Evaluate refund eligibility using Python rules.
4. Simulate a refund or escalate for human review.

### Safety Controls

The refund agent includes:

- An allowlisted tool dispatcher
- Order-ID validation
- Order-status validation
- Refund amount verification
- A seven-day refund-request window
- An automatic refund limit
- Approval checks before issuing a simulated refund
- Repeated-action detection
- Step and LLM-call limits
- Checkpointing support

The agent validates refund eligibility before attempting a simulated refund. Requests outside the allowed seven-day window are rejected. The automatic refund limit is ₹1,000; higher amounts require approval according to the implemented rules.

### Demonstration Results

**Ineligible order**

| Field | Result |
|---|---|
| Order ID | `ORD-110093` |
| Delivery time | 19 minutes |
| Outcome | `rejected` |
| Refund amount | ₹0 |

**Eligible order**

| Field | Result |
|---|---|
| Order ID | `ORD-110603` |
| Delivery time | 70 minutes |
| Order amount | ₹58 |
| Outcome | `simulated_refund_issued` |
| Tool steps | 4 |

The eligible-order demonstration satisfied the implemented late-delivery refund conditions and recorded a simulated ₹58 refund. No real money was transferred.

The demonstration shows how eligibility checks and controlled tool execution can be combined. Human approval and escalation are simulated rather than connected to a real support system.

### Main Outputs

- `outputs/policy_rag_results.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

## Automated Tests

The project includes 12 automated unit tests, and the recorded pytest run passed all 12 tests.

The refund-agent tests cover selected behaviors, including:

- Refund-request eligibility at the seven-day policy boundary
- Negative request days and invalid `NaN` input
- Delivery-time eligibility
- Invalid order amounts and delivery times
- Approval requirements before issuing a refund
- Order-ID validation
- Rejection of an eight-day request without LLM calls or agent steps

These tests validate selected rules and scenarios. They do not guarantee that every possible input, integration failure, or security issue has been tested.

## Limitations

- All datasets are synthetic and may not reflect real customer behavior.
- `mock_llm.py` simulates LLM behavior and is not a production language model.
- Ticket classifications and policy responses may be incorrect.
- TF-IDF retrieval relies on lexical similarity and may miss semantically relevant content.
- The policy evaluation contains only seven test questions.
- The lapse model's metrics depend on the dataset, label definition, features, and evaluation window.
- Temporal leakage needs to be assessed before using lapse predictions for future customer behavior.
- Customer segments and premium-customer flags require business validation before operational use.
- The retention list is illustrative and is not a fully leakage-free evaluation.
- Basic masking does not replace a comprehensive privacy and security review.
- JSON checkpointing provides basic saved progress, not production-grade crash recovery or exactly-once execution.
- Refund processing, human approval, and escalation are simulated.
- The workflow has not been validated for production deployment.

## Future Improvements

- Evaluate models using representative, permissioned data.
- Add time-based validation and stronger leakage checks for lapse prediction.
- Expand the policy evaluation set with ambiguous, edge-case, and adversarial questions.
- Route uncertain, invalid, or high-risk ticket classifications to human review.
- Add automated tests for refund eligibility, tool validation, retries, and failure handling.
- Improve checkpoint recovery, audit logging, and idempotency before any real refund integration.
- Add access control, privacy safeguards, and monitoring before production deployment.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates an end-to-end customer intelligence and support workflow covering data preparation, SQL analytics, customer segmentation, lapse prediction, support-ticket triage, policy retrieval, and simulated refund handling.

It is designed as a learning and portfolio project that demonstrates data and AI workflow design, evaluation, basic guardrails, and awareness of important production limitations. It is not intended for production customer support or real refund processing.
