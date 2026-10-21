# Instamart Customer Intelligence & Support Copilot

## Overview

This project is an educational capstone that combines data engineering, SQL analytics, customer segmentation, lapse prediction, privacy-aware support ticket triage, policy retrieval, and a guarded refund-agent workflow.

The project uses synthetic Instamart-style data. It does not contain real customer data and does not issue real refunds.

### Project objectives

- Build and validate a structured customer, product, and order database.
- Clean and analyse order data using Python, pandas, and SQLite.
- Create customer-level features and customer segments.
- Train and evaluate a model to predict customer lapse.
- Classify support tickets while applying privacy and token-budget controls.
- Retrieve fictional policy information using TF-IDF.
- Demonstrate a refund agent with validation, approval checks, execution limits, and checkpoint evidence.

### Technology stack

- Python 3.11+
- SQLite 3.25+
- pandas
- NumPy
- scikit-learn
- joblib
- Python standard library
- pytest for automated testing

The project uses the provided `generate_data.py` and `mock_llm.py` files. These files must remain unmodified.

## Reproduce

Run the following commands from the repository root in PowerShell.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this command in the same terminal and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Generate the synthetic data

```powershell
python generate_data.py
```

The generator creates synthetic customers, products, orders, support tickets, policy documents, and evaluation data.

### 4. Clean and inspect the data

```powershell
python src/inspect_data.py
python src/clean_data.py
```

### 5. Build and validate the database

```powershell
python src/build_database.py
python src/check_database.py
```

### 6. Run SQL analysis

```powershell
python src/sql_analysis.py
```

The SQL queries are documented in `sql/queries.sql`.

### 7. Build customer features and segments

```powershell
python src/customer_features.py
python src/customer_segmentation.py
```

### 8. Train and evaluate lapse prediction

```powershell
python src/lapse_prediction.py
```

### 9. Run support ticket triage

```powershell
python src/ticket_triage.py
```

### 10. Run policy retrieval evaluation

```powershell
python src/policy_rag.py
```

### 11. Run refund-agent demonstrations

Follow the usage and demonstration instructions implemented in `src/refund_agent.py`. The agent uses synthetic order data and simulated refund actions.

### 12. Run automated tests

```powershell
python -m pytest -q
```

Run commands from the repository root and execute them in order where one step generates inputs required by a later step. Generated outputs can change when scripts are rerun.

## Part A: Data Foundation

### Data generation

The project uses synthetic Instamart-style data to demonstrate a complete analytics workflow.

The generated datasets include:

- Customers
- Products
- Orders
- Support tickets
- Agent scenarios
- Policy evaluation questions
- Fictional policy documents

The data generator originally reported:

| Dataset | Generated rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Raw orders | 13,084 |
| Support tickets | 164 |

These are generated-data reference counts. Later cleaning and database steps can produce different row counts in the cleaned order data.

### Data cleaning

The cleaning process standardises and validates the generated data, handles duplicates, and writes cleaned datasets for downstream processing.

The cleaned dataset used in the latest local validation contained:

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

The cleaned order dataset contains 12,309 distinct order records. In the recorded validation, the order statuses included 11,110 completed orders and 1,199 cancelled orders.

### Database

The project stores relational data in SQLite at `data/instamart.db`.

The database workflow creates and loads the customers, products, and orders tables. Validation checks cover table availability, record counts, identifiers, order statuses, customer references, and numeric order values.

### SQL analysis

The SQL analysis demonstrates customer-level and order-level aggregations, including:

- Order counts by customer
- Total customer spending
- Order counts by status
- Revenue totals by status
- Highest-spending customers
- Customers with frequent orders

The SQL statements are maintained in `sql/queries.sql`.

### Part A outputs

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

## Part B: Customer Segmentation and Lapse Prediction

### Customer feature engineering

The project builds customer-level features from order history, including order counts, completed orders, completed spending, and the latest recorded order timestamp.

These features support customer segmentation and lapse prediction.

### Customer segmentation

The segmentation workflow groups customers according to their behavioural features and produces customer-level segment assignments.

The recorded segment distribution was:

| Segment | Customers |
|---|---:|
| Regular Customers | 436 |
| High Value | 419 |
| Low Spend | 183 |
| Top Value | 137 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The K-means evaluation recorded a silhouette score of approximately 0.525 for K=4. The final customer labels also include a separate group for customers with no completed orders.

A premium-customer flag is used to identify selected high-value customers for support prioritisation. Segment names are descriptive labels for this synthetic dataset; they are not verified business classifications.

### Lapse prediction

The lapse-prediction workflow constructs customer features and labels, trains candidate classification models, compares their performance, and saves the selected model and predictions.

The recorded modelling setup used:

- A prediction cutoff of 1 September 2025
- A September 2025 label window
- Random state 42
- Cross-validation F1 to select the candidate model

The eligible cohort in the recorded evaluation contained 1,156 customers.

### Recorded model metrics

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.728 | 0.737 |
| Lapsed-class precision | 0.64 | 0.66 |
| Lapsed-class recall | 0.65 | 0.65 |
| Lapsed-class F1 | 0.65 | 0.66 |

Random Forest was selected based on the recorded cross-validation F1 comparison. Metrics depend on the synthetic dataset, label construction, feature processing, and evaluation split.

### Part B outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C: Support Ticket Triage with Guardrails

### Ticket preparation and privacy

The ticket-triage workflow prepares support-ticket text and applies privacy-aware processing before classification. A redacted ticket dataset is produced for downstream evaluation.

The goal is to demonstrate classification and safe handling of support text using synthetic examples, not to establish production readiness.

### Classification approaches

The project compares zero-shot and few-shot mock-LLM classification approaches. The evaluation records predicted categories and compares them with the synthetic ground-truth labels.

In the recorded evaluation:

| Measurement | Zero-shot | Few-shot |
|---|---:|---:|
| Classification accuracy | 85.7% | 100% |
| Recorded token usage | 11,584 | 22,660 |

The recorded token-budget demonstration used a budget of 6,000 tokens and reported 5,951 tokens across 80 calls.

These results apply only to the evaluated synthetic tickets and mock-LLM behaviour. They should not be interpreted as evidence of equivalent performance with real customer tickets or a live language model.

### Guardrails

The workflow demonstrates:

- Redaction of selected sensitive text
- Controlled classification categories
- Comparison against known synthetic labels
- Token-budget tracking
- A budget demonstration for evaluating constrained execution

### Part C outputs

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

### Refund-agent scenario evaluation

The seven recorded refund-agent scenarios matched their expected outcomes and refund amounts.

| Scenario | Expected outcome | Actual outcome | Expected refund (INR) | Actual refund (INR) | Match |
|---|---|---|---:|---:|---|
| S1 | REFUND | REFUND | 128.25 | 128.25 | Yes |
| S2 | REFUND | REFUND | 135.00 | 135.00 | Yes |
| S3 | ESCALATE | ESCALATE | 1999.00 | 1999.00 | Yes |
| S4 | DENIED | DENIED | 0.00 | 0.00 | Yes |
| S5 | DENIED | DENIED | 0.00 | 0.00 | Yes |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 0.00 | Yes |
| S7 | FALLBACK | FALLBACK | 0.00 | 0.00 | Yes |

**Result:** 7 of 7 scenarios matched both the expected outcome and expected refund amount in the recorded scenario output.

### Stop-safely demonstrations

| Demonstration | Recorded status | Observed behaviour |
|---|---|---|
| Token-limit demonstration | `FALLBACK_TOKEN_LIMIT` | Stopped before the next LLM call when the token budget was exhausted. Recorded 2 steps, 2 LLM calls, and 196 tokens. |
| Timeout demonstration | `FALLBACK_TIMEOUT` | Stopped safely with a zero time limit. Recorded 0 steps, 0 LLM calls, and 0 tokens. |

These entries document the recorded safety demonstrations. They do not establish behaviour for every possible runtime failure.

### Crash/resume checkpoint evidence

The recorded checkpoint run for order `ORD-110603` completed with status `REFUND` and a simulated refund amount of INR 58.00. The recorded action sequence was:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The run recorded four steps, five LLM calls, and 948 tokens. The checkpoint evidence is stored in `state/refund_agent_CRASH_RESUME_CHECK_001.json`; the corresponding run is recorded in `outputs/agent_runs.jsonl`.

A second recorded checkpoint scenario, `CRASH_RESUME_S1_CHECK`, completed with status `REFUND` and a simulated refund amount of INR 128.25 for order `ORD-100048`. Its recorded action sequence was also `get_order`, `search_policy`, `calculate_refund`, and `issue_refund`.

The logs document the recorded action sequence and checkpoint evidence. They should not, by themselves, be treated as proof of a complete process restart unless the restart-and-resume procedure was separately verified.

### Part D outputs

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_summary.json`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`
- `outputs/refund_agent_scenarios.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `state/` checkpoint files, including the named crash/resume evidence files

## Validation Summary

The latest reported local validation included:

- Cleaned-data validation passed
- Database validation passed
- SQL query checks passed
- Automated test run: 14 tests passed

Run the validation commands again against the current checkout before submission if project files have changed.

The recorded checks provide evidence for the tested data and workflows; they do not prove that every possible input or production failure has been tested.

## Limitations and Next Steps

- The datasets are synthetic and do not represent real Instamart customers.
- The LLM is mocked; behaviour may differ from a live model.
- Model metrics depend on synthetic labels, feature construction, preprocessing, and the test split.
- Feature importance and model coefficients describe model behaviour; they do not prove causation.
- Ticket-triage accuracy and token counts apply only to the evaluated mock-LLM runs.
- TF-IDF retrieval can miss relevant context or return incomplete passages.
- The RAG evaluation set is small and does not establish broad real-world reliability.
- Refund actions are simulated; no payment transaction occurs.
- Automated tests cover selected cases, not every possible production failure.
- Before production use, the system would require stronger authentication, authorisation, monitoring, transactional controls, privacy reviews, and evaluation with appropriately governed real-world data.

## Conclusion

This project combines data engineering, SQL, customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and a guarded refund-agent workflow in one educational repository.

The repository demonstrates how data validation, machine-learning evaluation, retrieval, deterministic business rules, and execution guardrails can work together in a synthetic customer-support setting.
