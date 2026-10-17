# Instamart Customer Intelligence & Support Copilot

## Overview

The **Instamart Customer Intelligence & Support Copilot** is an end-to-end data analytics and AI workflow built using synthetic Instamart-style e-commerce data.

The project demonstrates how data engineering, SQL analytics, machine learning, text classification, policy retrieval, and rule-based safeguards can support customer intelligence and customer service workflows.

The pipeline covers four major areas:

- **Part A:** Data engineering, database construction, SQL analysis, and exploratory data analysis
- **Part B:** Customer segmentation and lapse prediction
- **Part C:** Support ticket classification, validation, and routing
- **Part D:** Policy retrieval and a simulated refund agent

> **Important:** This is an educational prototype. The datasets are synthetic, the LLM behavior is simulated, and refund processing does not transfer real money.

## Tech Stack

- **Language:** Python 3.11+
- **Data processing:** pandas, NumPy
- **Database:** SQLite
- **Machine learning:** scikit-learn
- **Text retrieval:** TF-IDF and n-gram similarity
- **Model persistence:** joblib
- **Workflow state:** JSON checkpointing
- **Version control:** Git and GitHub

## Project Structure

```text
instamart-customer-copilot/
├── data/
│   └── policies/
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
├── generate_data.py
├── mock_llm.py
└── README.md
```

The scripts generate the datasets, database, model artifacts, and evaluation outputs when executed. Generated files may appear in additional directories after the pipeline runs.

## Getting Started

### Requirements

- Python 3.11 or later
- Git
- Windows PowerShell or another terminal with a compatible Python environment

### 1. Clone the repository

```powershell
git clone https://github.com/kavisumathi2-cloud/instamart-customer-copilot.git
cd instamart-customer-copilot
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, run the following command in the current terminal and try activating the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 3. Install dependencies

```powershell
python -m pip install pandas numpy scikit-learn joblib
```

### 4. Generate synthetic data

```powershell
python generate_data.py
```

The data-generation script creates synthetic customer, product, order, and support-ticket datasets.

### 5. Run the pipeline

Execute the following commands from the repository root, in order:

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

The pipeline prepares the data, builds the SQLite database, creates customer features, evaluates customer segments and lapse predictions, processes support tickets, retrieves policy information, and runs a refund-agent demonstration.

## Part A — Data Engineering, SQL & EDA

### Implementation

- Generated synthetic customer, product, order, and support-ticket datasets.
- Cleaned and validated the source datasets.
- Removed duplicate order records during cleaning.
- Built a SQLite database for structured analysis.
- Created customer-level features using completed-order history.
- Performed SQL aggregations and exploratory analysis.

### Dataset Summary

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Support tickets | 164 |

The cleaned order dataset contains 11,110 completed orders and 1,199 cancelled orders.

### SQL Analysis

The analysis includes:

- Order counts grouped by status
- Total order amount
- Order frequency by customer
- Highest-spending customers
- Order amounts grouped by status

These queries demonstrate how relational data can be transformed into useful operational summaries.

### Main Outputs

- `data/clean/`
- `data/instamart.db`
- `data/clean/customer_features.csv`

## Part B — Customer Segmentation & Lapse Prediction

### Customer Segmentation

Customer-level features represent three common RFM concepts:

- **Recency:** Time since the customer's most recent completed order
- **Frequency:** Number of completed orders
- **Monetary:** Total spending on completed orders

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

The resulting segment labels provide a descriptive view of customer behavior. The premium-customer flag supports downstream prioritization, but segment definitions would require business validation before production use.

### Lapse Prediction

Two classification approaches were evaluated:

- Logistic Regression
- Random Forest

The Random Forest model was selected based on cross-validation F1 performance.

### Random Forest Evaluation

| Metric | Result |
|---|---:|
| Test accuracy | 0.737 |
| F1 score for lapsed customers | 0.655 |
| Precision for lapsed customers | 0.659 |
| Recall for lapsed customers | 0.652 |

Frequency and monetary value were among the influential features in the trained model. Class balancing was also explored to assess the trade-off between precision and recall.

These results apply to the selected synthetic dataset, label definition, and evaluation setup. They should not be interpreted as evidence of expected production performance.

**Potential limitation:** Temporal leakage must be reviewed carefully. Features used to predict future lapse should contain only information available at the prediction date. Features derived from a full observation window may reveal information that would not have been available at that time.

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
| Accuracy on valid classifications | 84.89% |
| Accuracy including fallback predictions | 76.22% |
| LLM calls used | 184 of 500 |

These results demonstrate the workflow's validation and fallback behavior on synthetic data. They do not establish real-world classification accuracy, privacy compliance, or production readiness.

### Main Outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`

## Part D — Policy RAG & Refund Agent

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

Safety controls include:

- An allowlisted tool dispatcher
- Order-status validation
- Refund amount verification
- An automatic refund limit
- Repeated-action detection
- Step and tool-call limits

### Demonstration Results

**Ineligible order**

| Field | Result |
|---|---|
| Order | `ORD-110093` |
| Delivery time | 19 minutes |
| Outcome | `rejected` |
| Refund amount | ₹0 |

**Eligible order**

| Field | Result |
|---|---|
| Order | `ORD-110603` |
| Delivery time | 70 minutes |
| Order amount | ₹58 |
| Outcome | `simulated_refund_issued` |
| Tool steps | 4 |

The demonstration shows how eligibility checks and controlled tool execution can be combined. No real money was transferred, and human approval is simulated rather than connected to an actual approval system.

### Main Outputs

- `outputs/policy_rag_results.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

## Limitations

- All datasets are synthetic and may not reflect real customer behavior.
- `mock_llm.py` simulates LLM behavior and is not a production language model.
- Ticket classifications and policy responses may be incorrect.
- TF-IDF retrieval relies on lexical similarity and may miss semantically relevant content.
- The lapse model's metrics depend on the dataset, label definition, and evaluation window.
- Temporal leakage needs to be assessed before using lapse predictions for future customer behavior.
- Customer segments and premium-customer flags require business validation before operational use.
- JSON checkpointing provides basic saved progress, not production-grade crash recovery or exactly-once execution.
- Refund processing, human approval, and escalation are simulated.
- Basic masking does not replace a comprehensive privacy and security review.
- The workflow has not been validated for production deployment.

## Next Steps

- Evaluate models using representative, permissioned data.
- Add time-based validation and stronger leakage checks for lapse prediction.
- Expand the policy evaluation set with ambiguous, edge-case, and adversarial questions.
- Route uncertain, invalid, or high-risk ticket classifications to human review.
- Add automated tests for refund eligibility, tool validation, retries, and failure handling.
- Improve checkpoint recovery, audit logging, and idempotency before any real refund integration.
- Add access control, privacy safeguards, and monitoring before production deployment.

## Conclusion

This project demonstrates an end-to-end customer intelligence and support workflow, covering data preparation, SQL analytics, customer segmentation, lapse prediction, support-ticket triage, policy retrieval, and simulated refund handling.

It is designed as a learning and portfolio project that demonstrates data and AI workflow design, evaluation, basic guardrails, and awareness of important production limitations.
