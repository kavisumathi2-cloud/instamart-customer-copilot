# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data engineering, SQL analytics, customer segmentation, customer lapse prediction, support ticket triage, policy retrieval, and a simulated refund agent.

The project demonstrates an end-to-end data and AI workflow using Python, pandas, SQLite, scikit-learn, TF-IDF, and a mock LLM.

**Important:** This is an educational prototype. The datasets are synthetic, the LLM is simulated, and refund processing does not transfer real money.

## Technologies Used

- Python 3.11+
- pandas and NumPy
- SQLite
- scikit-learn
- TF-IDF and n-gram similarity
- Joblib
- Unit testing with `unittest`
- Git and GitHub for version control and project submission

## Project Setup and Reproduction

### Requirements

- Python 3.11 or newer
- VS Code or another Python IDE
- A virtual environment
- The supplied `generate_data.py` and `mock_llm.py` files

### 1. Activate the virtual environment

Run this command in PowerShell from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install pandas numpy scikit-learn joblib
```

### 3. Generate the synthetic dataset

```powershell
python generate_data.py
```

### 4. Run the project scripts

Run the following commands from the repository root, in order:

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

The scripts create cleaned datasets, a SQLite database, customer features, trained model artifacts, evaluation outputs, ticket triage results, policy retrieval results, and refund-agent demonstration outputs.

**Note:** Running the scripts again may regenerate synthetic data and overwrite generated outputs. Use the same project environment and run the scripts in the documented order.

## Part A — Data Engineering, SQL and EDA

### Work completed

- Generated synthetic customer, product, order, and support-ticket datasets.
- Cleaned and validated the source data.
- Saved cleaned CSV files under `data/clean/`.
- Built a SQLite database at `data/instamart.db`.
- Created customer-level features for analysis.
- Performed SQL queries and exploratory data analysis.

### Data summary

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Support tickets | 164 |

The project database was verified to contain 1,200 customers, 40 products, 12,309 orders, and 164 tickets.

The cleaned order dataset contains 11,110 completed orders and 1,199 cancelled orders, according to the project summary.

### Main outputs

- `data/clean/`
- `data/instamart.db`
- `data/clean/customer_features.csv`

## Part B — Customer Segmentation and Lapse Prediction

### Customer segmentation

Customer segmentation uses RFM-style features:

- **Recency:** Time since the customer's last completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total spending on completed orders.

K-means clustering was used to identify customer groups based on their purchasing behaviour.

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |

The premium-customer flag identifies the selected high-value segment for downstream support prioritisation.

### Lapse prediction

Two classification approaches were evaluated:

- Logistic Regression
- Random Forest

The Random Forest model was selected based on cross-validation F1 performance.

| Metric | Random Forest |
|---|---:|
| Test accuracy | 0.737 |
| F1 score for lapsed customers | 0.655 |
| Precision for lapsed customers | 0.659 |
| Recall for lapsed customers | 0.652 |

Frequency and monetary value were among the influential features in the trained Random Forest model.

Class balancing was also explored to examine the trade-off between identifying more lapsed customers and maintaining prediction precision.

These results describe the project's synthetic dataset and evaluation setup. They should not be interpreted as expected production performance.

### Main outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage

### Work completed

- Masked email addresses and phone numbers in ticket text before passing it to the mock LLM.
- Classified support tickets into billing, delivery, technical, and fraud categories.
- Added retry handling for transient failures.
- Validated model responses and category names.
- Enforced an LLM-call budget.
- Routed tickets using category and customer-segment information.
- Recorded fallback classifications and evaluation metrics.

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

These metrics describe this synthetic dataset and mock LLM only. They do not establish expected real-world production performance.

### Main outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`

## Part D — Policy RAG and Refund Agent

### Policy retrieval

The policy documents cover refunds, delivery, payments, privacy, and escalation.

The retrieval workflow uses TF-IDF and n-gram similarity to identify relevant policy sentences. Deterministic rules handle selected high-impact questions, including late-delivery refunds and payment issues.

Seven test questions returned the expected answers in the initial evaluation. This is a small test set and does not establish general policy-answering accuracy.

### Refund agent workflow

The refund agent demonstrates a ReAct-style workflow:

1. Retrieve the order details.
2. Search relevant policy documents.
3. Calculate refund eligibility using Python rules.
4. Simulate a refund or escalate for human review.

### Safety controls

The refund agent includes:

- An allowlisted tool dispatcher.
- Order-ID validation.
- Order-status checks.
- Refund amount validation.
- A seven-day refund-request window.
- An automatic refund limit.
- Approval checks before issuing a simulated refund.
- Repeated-action detection.
- Step and LLM-call limits.
- Checkpointing support.

The refund agent rejects requests outside the allowed seven-day window before making LLM calls. Boundary tests confirm that requests at day 7 are accepted by the time-window validator, while requests at day 8 are rejected.

All refund outcomes are simulated. No real payment or refund transaction is performed.

### Demonstration results

**Ineligible order**

- Order ID: `ORD-110093`
- Delivery time: 19 minutes
- Outcome: `rejected`
- Refund amount: ₹0

**Eligible order**

- Order ID: `ORD-110603`
- Delivery time: 70 minutes
- Order amount: ₹58
- Outcome: `simulated_refund_issued`
- Tool steps: 4

These examples demonstrate the implemented decision flow using synthetic order data. They do not represent real customer transactions.

### Main outputs

- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

## Automated Tests

The refund agent is covered by 12 automated unit tests. All 12 tests passed in the recorded test run.

Run the tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The tests cover:

- Refund-request eligibility at the seven-day policy boundary.
- Negative request days and invalid `NaN` input.
- Delivery-time eligibility.
- Invalid order amounts and delivery times.
- Approval requirements before issuing a refund.
- Order-ID validation.
- Rejection of an eight-day request without LLM calls or agent steps.

These tests validate selected rules and scenarios. They do not guarantee that every possible input, integration failure, or security issue has been tested.

## Limitations

- The datasets are synthetic and do not represent real Instamart customer data.
- The LLM is mocked and does not demonstrate the behaviour of a live language model.
- Policy retrieval uses lexical similarity and may miss relevant information or retrieve an incomplete match.
- The policy evaluation contains only seven test questions.
- Model metrics depend on the synthetic data, feature construction, labels, and evaluation split.
- Customer behaviour and lapse predictions should not be treated as production-ready decisions.
- Refund processing is simulated; no payment gateway or real refund API is connected.
- Human approval and escalation are demonstrated in the workflow rather than through a real support system.
- JSON checkpointing is a basic demonstration and may need stronger persistence and recovery mechanisms for production use.
- Automated tests cover selected refund rules but not every possible failure scenario.

## Future Improvements

- Expand automated tests to cover more agent failure and recovery scenarios.
- Increase the policy RAG evaluation set and test ambiguous or conflicting policy questions.
- Improve monitoring and reporting for ticket-triage fallbacks.
- Evaluate model performance with additional data splits and appropriate validation.
- Improve checkpoint recovery and duplicate-action protection.
- Integrate a real support platform only after adding authentication, authorisation, audit logging, and production-grade safeguards.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates how data engineering, SQL analytics, machine learning, text retrieval, and rule-based agent workflows can be combined in a single educational project.

It includes customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and a simulated refund agent with explicit eligibility checks and automated tests.

The project is intended for learning and demonstration, not for production customer support or real refund processing.