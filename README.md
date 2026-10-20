# Instamart Customer Intelligence & Support Copilot

## 1. Overview

This project builds an end-to-end customer intelligence and support copilot using synthetic Instamart-style e-commerce data.

The project combines data cleaning, SQLite database engineering, SQL analytics, customer segmentation, lapse prediction, privacy-aware support ticket triage, policy retrieval, and a simulated refund agent.

The implementation is organised into four parts:

- **Part A — Data Foundation:** Clean, validate, store, and analyse the source data.
- **Part B — Customer Intelligence:** Segment customers and predict customer lapse.
- **Part C — Support Ticket Triage:** Protect personal information, classify tickets, handle errors, and apply token-budget guardrails.
- **Part D — Policy RAG and Refund Agent:** Retrieve policy information and demonstrate a guarded refund workflow.

**Important:** All project data is synthetic and fictional. The supplied `mock_llm.py` provides simulated language-model responses. Refund actions are demonstrations only; no real money is transferred.

## 2. Reproduce

### Requirements

- Python 3.11 or newer
- SQLite 3.25 or newer
- pandas
- NumPy
- scikit-learn
- joblib
- The supplied `generate_data.py` and `mock_llm.py`

The supplied `generate_data.py` and `mock_llm.py` files must remain unmodified.

Run the following commands from the repository root in PowerShell.

### Step 1 — Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2 — Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn joblib pytest
```

### Step 4 — Generate synthetic data

```powershell
python generate_data.py
```

The generator produces synthetic customers, products, orders, tickets, policy-evaluation questions, agent scenarios, and policy documents.

Expected final generator message:

```text
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

### Step 5 — Clean the data

```powershell
python src/clean_data.py
```

### Step 6 — Build the SQLite database

```powershell
python src/build_database.py
```

### Step 7 — Run SQL analysis

```powershell
python src/sql_analysis.py
```

### Step 8 — Create customer features

```powershell
python src/customer_features.py
```

### Step 9 — Run customer segmentation

```powershell
python src/customer_segmentation.py
```

### Step 10 — Run lapse prediction

```powershell
python src/lapse_prediction.py
```

### Step 11 — Run support ticket triage

```powershell
python src/ticket_triage.py
```

### Step 12 — Run policy retrieval evaluation

```powershell
python src/policy_rag.py
```

### Step 13 — Run the simulated refund agent

```powershell
python src/refund_agent.py
```

### Step 14 — Run data and SQL checks

```powershell
python check_clean_data.py
python src/check_database.py
python check_sql_queries.py
python check_order_stats.py
```

### Step 15 — Run automated tests

```powershell
python -m pytest -v
```

The scripts generate cleaned CSV files, a SQLite database, customer features, segmentation and lapse-prediction outputs, ticket-triage results, policy-retrieval evaluations, and refund-agent demonstration files.

Running the data generator again may overwrite the generated datasets. Run the pipeline in the documented order when reproducing the results.

## 3. Part A — Data Foundation

### 3.1 Data generation and validation

The project starts with synthetic data representing customers, products, orders, and support tickets.

The raw generator produces 13,084 order rows. Cleaning and validation create the datasets used by the remaining parts.

### 3.2 Final dataset summary

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Cleaned orders | 12,309 |
| Support tickets | 164 |

The cleaned order dataset contains 11,110 completed orders and 1,199 cancelled orders. The cleaning pipeline removes pending orders from the final cleaned order dataset.

### 3.3 Cleaning and validation

The data-cleaning workflow checks for issues such as:

- Duplicate records
- Missing and invalid values
- Inconsistent categorical values
- Invalid order amounts
- Invalid delivery-time values
- Missing values that need appropriate handling

The cleaning process writes the resulting datasets under `data/clean/`.

Validation commands:

```powershell
python check_clean_data.py
python src/check_database.py
```

The recorded database verification passed the following checks:

- 1,200 customers
- 40 products
- 12,309 orders
- 164 tickets
- Unique customer, product, and order IDs
- Valid order statuses
- No missing order customer IDs
- No negative order amounts

### 3.4 SQLite database

The database is stored at:

`data/instamart.db`

It provides a consistent source for SQL queries and downstream analysis.

### 3.5 SQL analysis

The project includes seven SQL analysis queries in `sql/queries.sql`.

The queries demonstrate customer-level spending, category and city analysis, ranking, window functions, and cumulative totals.

The SQL validation script reported all seven queries passing.

Example verified findings:

| Analysis | Result |
|---|---:|
| Total completed-order revenue | INR 2,998,668.80 |
| Bengaluru Electronics revenue | INR 233,909.45 |
| Bengaluru total order amount in city summary | INR 523,643.40 |
| Mumbai total order amount in city summary | INR 479,058.50 |

The highest-spending customer in each city can be identified using SQL ranking functions. `RANK()` and `ROW_NUMBER()` differ when multiple rows have tied values.

### 3.6 Order statistics

The recorded order-amount statistics include:

| Statistic | Value |
|---|---:|
| Completed orders | 11,110 |
| Mean order amount | INR 269.91 |
| Median order amount | INR 135.00 |
| First quartile | INR 64.80 |
| Third quartile | INR 297.00 |
| Maximum order amount | INR 3,998.00 |
| Skewness | 4.61 |
| Upper outliers | 897 |

The mean being higher than the median, together with positive skewness, indicates that a relatively small number of large orders influence the average.

Correlation results describe associations in the synthetic data and do not establish causation.

### Main Part A outputs

- `data/raw/`
- `data/clean/`
- `data/policies/`
- `data/instamart.db`
- `data/clean/customer_features.csv`
- `sql/queries.sql`

## 4. Part B — Customer Segmentation and Lapse Prediction

### 4.1 Customer segmentation

Customer segmentation groups customers according to their purchasing behaviour.

The RFM-style features are:

- **Recency:** Time since the customer's last completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total spending on completed orders.

K-means clustering is used to identify customer groups. Customers without completed orders are handled separately.

### 4.2 Segment distribution

| Segment | Customers |
|---|---:|
| Regular Customers | 436 |
| High Value | 419 |
| Low Spend | 183 |
| Top Value | 137 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The premium-customer flag identifies the selected highest-mean-monetary segment for downstream support prioritisation.

### 4.3 K-means evaluation

The saved `outputs/kmeans_scores.csv` records inertia and silhouette scores for different cluster counts.

| Number of clusters (k) | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.7 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.0 | 0.525 |
| 5 | 617.5 | 0.478 |
| 6 | 503.2 | 0.483 |
| 7 | 408.5 | 0.416 |
| 8 | 366.2 | 0.412 |

Among these evaluated values, `k=4` has the highest silhouette score of 0.525. Inertia generally decreases as the number of clusters increases, so inertia should not be considered in isolation.

### 4.4 Principal Component Analysis

The first two principal components explain the following proportion of variance:

| Component | Explained variance ratio |
|---|---:|
| PC1 | 0.634 (63.4%) |
| PC2 | 0.218 (21.8%) |
| **Combined** | **0.852 (85.2%)** |

These values describe how much of the feature variance is represented by the first two components.

### 4.5 Lapse prediction

The lapse-prediction task uses:

- Cutoff date: 1 September 2025
- Label window: September 2025
- Random state: 42

The eligible lapse cohort contains 1,156 customers.

| Lapse label | Customers |
|---|---:|
| Not Lapsed | 713 |
| Lapsed | 443 |
| **Total** | **1,156** |

The lapsed proportion is approximately 38.32%.

The data is divided into 924 training rows and 232 testing rows using a stratified split.

### 4.6 Feature construction and leakage prevention

The model features include historical recency, frequency, monetary value, discount, cancellation rate, delivery time, product category, city, age, gender, and customer tenure.

The implementation follows these safeguards:

- No orders dated 1 September 2025 or later are used to construct the model features.
- Preprocessing is fitted on the training split only.
- Categorical variables are one-hot encoded.
- Numerical features are scaled.
- The segment labels are not used as lapse-model features.
- The test split is kept separate from model fitting and preprocessing.

### 4.7 Model comparison

An always-not-lapsed baseline achieves approximately 61.6% accuracy but has an F1 score of zero for the lapsed class.

Logistic Regression and Random Forest are evaluated. Random Forest is selected because it has the higher cross-validation F1 score.

| Model | Test accuracy | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.728 | 0.64 | 0.65 | 0.65 |
| Random Forest | 0.737 | 0.66 | 0.65 | 0.66 |

### 4.8 Cross-validation and selected parameters

| Model | Best parameters | Best CV F1 |
|---|---|---:|
| Logistic Regression | `C=100` | 0.651 |
| Random Forest | `max_depth=8`, `min_samples_leaf=10` | 0.681 |

The selected Random Forest configuration uses `max_depth=8` and `min_samples_leaf=10`.

### 4.9 Random Forest confusion matrix

```text
[[113, 30],
 [ 31, 58]]
```

Rows represent actual labels, and columns represent predicted labels.

- 113 customers were correctly predicted as not lapsed.
- 30 customers were predicted as lapsed but were not lapsed in the label window.
- 31 lapsed customers were incorrectly predicted as not lapsed.
- 58 lapsed customers were correctly identified.

False negatives can cause retention opportunities to be missed. False positives can result in unnecessary retention outreach.

### 4.10 Oversampling experiment

The training set was oversampled to balance the classes. The test set remained unchanged.

| Evaluation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.644 | 0.652 | 0.648 |
| Oversampled training distribution | 0.589 | 0.854 | 0.697 |

Oversampling improves recall in this experiment but reduces precision. The appropriate balance depends on the cost of missed lapses compared with unnecessary retention outreach.

### 4.11 Feature importance

The top five Random Forest features are:

| Feature | Importance |
|---|---:|
| Frequency | 0.2796 |
| Monetary value | 0.2712 |
| Recency | 0.1152 |
| Average delivery minutes | 0.0612 |
| Cancellation rate | 0.0525 |

These values describe the fitted model's relative feature importance. They do not prove that any feature causes a customer to lapse.

### 4.12 Logistic Regression coefficients

| Feature | Coefficient |
|---|---:|
| Recency | +0.0392 |
| Frequency | -1.6973 |
| Monetary value | +0.2834 |

The signs describe associations with predicted lapse odds, holding other model features constant. They should not be interpreted as causal effects.

### 4.13 Retention list

`outputs/retention_list.csv` contains up to 10 eligible premium customers from the test set, ranked by predicted lapse probability. The recorded run identified one customer in this list who was actually lapsed.

**Leakage limitation:** The premium flag is derived from whole-window segmentation, which can include September information used to define the lapse label. Therefore, the retention list is illustrative and is not a leakage-free evaluation, even though the lapse model's own features use pre-cutoff information.

### Main Part B outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

The implementation also verifies that predictions remain consistent after the saved model is reloaded.

## 5. Part C — Support Ticket Triage with Guardrails

### 5.1 Input validation

The pipeline processes 164 synthetic support tickets in their input order.

Before a ticket is sent to the mock LLM:

- Empty or whitespace-only text is assigned `skipped_empty`.
- Text exceeding the configured 400-token input limit is assigned `flagged_long`.
- Guardrail-rejected tickets are not sent to the model for classification.

The dataset contains four empty tickets and four long tickets. The long tickets exceed the configured limit.

### 5.2 PII redaction

The pipeline redacts personal information before constructing model prompts.

The configured substitutions are:

| Information | Redaction placeholder | Substitutions |
|---|---|---:|
| Email address | `[EMAIL]` | 55 |
| Phone number | `[PHONE]` | 91 |
| Order ID | `[ORDER_ID]` | 132 |
| **Total** | | **278** |

Redaction is applied in this order:

1. Email addresses
2. Phone numbers
3. Order IDs

The implementation checks the redacted text again using the same patterns. The recorded validation found zero remaining matches.

The message supplied to the mock model is the redacted ticket text, without additional ticket content being appended to that user message.

### 5.3 Classification prompts

The pipeline evaluates zero-shot and few-shot prompting approaches.

- **Zero-shot:** The model receives instructions and the classification task without labelled examples.
- **Few-shot:** The prompt also includes examples intended to guide the model's category selection.

The classification pipeline validates model output before accepting a category.

### 5.4 Resilience and retries

The pipeline includes retry handling for transient model errors.

The retry workflow records retry information and uses a wait with jitter before attempting another call. Invalid or unusable model responses are handled through the configured fallback path.

The project records 19 transient errors that were recovered during the relevant retry evaluation. Retry attempts and model-call counts depend on the run and should not be treated as interchangeable metrics.

### 5.5 Output validation and fallback handling

The pipeline checks model output and distinguishes successful JSON classifications, keyword-based classifications, invalid responses, and other guarded outcomes.

The recorded classification outputs include:

| Output type | Count |
|---|---:|
| JSON classifications | 135 |
| Keyword-based classifications | 12 |
| Invalid model outputs | 9 |
| **Total eligible-ticket outcomes in this breakdown** | **156** |

The 156 eligible-ticket outcomes exclude the four empty tickets and four long tickets that are stopped by input guardrails.

Invalid model output is not silently treated as a valid model classification.

### 5.6 Token-budget guardrail

The pipeline enforces a configured model-token budget. When the remaining budget is insufficient for another model call, the ticket is handled through the appropriate budget-exhaustion path.

The separate budget-demonstration output records these statuses:

| Status | Count |
|---|---:|
| `classified_json` | 70 |
| `classified_keyword` | 6 |
| `flagged_invalid_output` | 4 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |
| `budget_exhausted` | 76 |
| **Total** | **164** |

These counts describe the budget demonstration and should not be combined with metrics from a different triage run.

### 5.7 Zero-shot and few-shot evaluation

The recorded evaluation reports the following results:

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets in the comparison | 147 | 147 |
| Accuracy | 85.7% | 100% |
| Total tokens | 11,584 | 22,660 |
| Tokens per classified ticket | 78.80 | 154.15 |

The few-shot prompt achieved higher accuracy in this recorded comparison while using more tokens per classified ticket. These results apply only to the evaluated synthetic tickets and the supplied mock LLM.

### 5.8 Customer-aware routing

The ticket-triage pipeline uses customer segment information from Part B as part of support prioritisation.

The pipeline also records classification results, fallback outcomes, and evaluation data for review.

### Main Part C outputs

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`
- `outputs/triage_results.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_confusion_zero_shot.csv`
- `outputs/triage_budget_demo.csv`

## 6. Part D — Policy RAG and Refund Agent

### 6.1 Policy documents and retrieval

The policy collection contains five documents covering:

- Refunds
- Delivery
- Payments
- Privacy
- Escalation

The policy RAG implementation uses TF-IDF and n-gram similarity to retrieve relevant policy text. It evaluates retrieved context against the supplied policy-evaluation questions.

Retrieval quality depends on the wording of the question, the selected chunks, and the retrieval configuration.

### 6.2 Retrieval evaluation

The recorded overlap comparison found that the expected document or fact was retrieved for 11 of 12 evaluation questions in both the overlap and no-overlap configurations at the evaluated top-K setting.

The evaluation also recorded:

- 12 of 12 answerable questions correctly identified as answerable.
- 2 of 2 unanswerable questions correctly identified as unanswerable.

These are results for the available evaluation set, not a guarantee of general policy-answering accuracy.

### 6.3 Example policy rules

The synthetic policy documents and agent workflow demonstrate rules such as:

- A late-delivery refund requires a completed order delivered in more than 45 minutes.
- The late-delivery refund request must be made within seven days of delivery.
- A delivery time of 45 minutes or less does not satisfy the late-delivery threshold.
- The failed-payment refund workflow uses a 48-hour request window.
- Approved refunds are described as returning to the original payment method within five working days.
- Payment-card numbers are not stored by the workflow.

These are fictional project policies, not actual Instamart policy statements.

### 6.4 Refund agent workflow

The refund agent demonstrates a guarded, ReAct-style workflow:

1. Validate the request and order identifier.
2. Retrieve order details from the SQLite database.
3. Retrieve relevant policy information.
4. Evaluate eligibility using deterministic Python rules.
5. Validate the proposed refund amount and required approval.
6. Simulate an eligible refund, deny an ineligible request, or escalate the case.
7. Record the execution result and checkpoint information.

The LLM helps coordinate the workflow, while important eligibility and financial decisions are validated using Python rules.

### 6.5 Safety controls

The implementation includes:

- An allowlisted tool dispatcher
- Order-ID validation
- Order-status checks
- Refund-amount validation
- A seven-day refund-request window
- An automatic refund limit
- Approval checks before issuing a simulated refund
- Repeated-action detection
- Step and LLM-call limits
- Checkpointing and recovery support
- Guarded fallback outcomes

The automatic refund limit is INR 1,000. The configured late-delivery threshold is more than 45 minutes.

Requests outside the allowed seven-day window are rejected by the eligibility validator before the agent makes LLM calls. Boundary tests cover the seven-day and eight-day cases.

### 6.6 Agent scenario evaluation

The scenario evaluation compares expected and actual outcomes and refund amounts.

| Scenario | Expected outcome | Actual outcome | LLM calls | Tokens |
|---|---|---|---:|---:|
| S1 | REFUND | REFUND | 5 | 953 |
| S2 | REFUND | REFUND | 5 | 953 |
| S3 | ESCALATE | ESCALATE | 5 | 943 |
| S4 | DENIED | DENIED | 4 | 661 |
| S5 | DENIED | DENIED | 3 | 406 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 154 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 | 1,296 |

All seven recorded scenarios matched their expected outcome labels. The recorded refund amounts also matched the expected refund amounts.

S7 reached the configured maximum-iteration fallback rather than completing a normal terminal decision.

### 6.7 Refund examples

**Eligible late-delivery refund demonstration**

- Order ID: `ORD-110603`
- Customer ID: `C1029`
- Order amount: INR 58
- Delivery time: 70 minutes
- Request age: 3 days
- Result: `simulated_refund_issued`
- Agent steps: 4
- LLM calls: 5
- Recorded token count: 968

This example demonstrates an eligible synthetic order. It does not initiate a real financial transaction.

**Ineligible order demonstration**

- Order ID: `ORD-110093`
- Delivery time: 19 minutes
- Result: rejected
- Refund amount: INR 0

The order does not satisfy the configured late-delivery threshold.

### 6.8 Token-limit and timeout demonstrations

The agent also records guarded fallback behaviour:

| Demonstration | Outcome | LLM calls | Tokens |
|---|---|---:|---:|
| Token-limit demonstration | `FALLBACK_TOKEN_LIMIT` | 2 | 196 |
| Timeout demonstration | `FALLBACK_TIMEOUT` | 0 | 0 |

These cases demonstrate that the workflow can stop rather than continue indefinitely when a configured limit or timeout condition is reached.

### 6.9 Crash and resume demonstration

The repository contains crash/resume state examples for the refund workflow.

The recorded recovery examples completed with a REFUND outcome, four steps, five LLM calls, and 948 tokens for each named run.

The saved state and execution logs provide evidence for the demonstrated recovery path. These aggregate counts alone do not prove exactly-once execution for every possible crash timing or external side effect.

### 6.10 Refund-agent limitations

All refund actions are simulated. No payment gateway, live customer account, or real refund API is connected.

The project demonstrates validation, approval checks, tool restrictions, execution limits, and checkpointing in an educational environment. A production deployment would require stronger transactional guarantees, persistent audit records, authentication, authorisation, and duplicate-action protection.

### Main Part D outputs

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_summary.json`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`
- `outputs/refund_agent_scenarios.csv`
- `outputs/refund_agent_CRASH_RESUME_CHECK_001.json`
- `outputs/refund_agent_CRASH_RESUME_S1_CHECK.json`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

## 7. Limitations and Next Steps

### Limitations

- All datasets are synthetic and do not represent real Instamart customer data.
- The supplied LLM is mocked; results may differ substantially from those of a live language model.
- Model metrics depend on synthetic labels, feature construction, preprocessing, and the evaluation split.
- The retention list is illustrative and is not a leakage-free evaluation because the premium flag is derived from whole-window segmentation.
- Feature importance and coefficients describe model behaviour and do not establish causation.
- Ticket-triage accuracy and token counts apply to the evaluated mock-LLM runs, not to real-world support traffic.
- Policy retrieval uses lexical similarity and may retrieve incomplete or irrelevant context for ambiguous questions.
- The RAG evaluation set is small and does not establish broad policy-answering reliability.
- Refund processing is simulated; no real payment or refund transaction occurs.
- Checkpoint and crash/resume demonstrations do not guarantee exactly-once execution under every failure condition.
- Automated tests cover selected rules and scenarios, not every possible input or production failure.

### Next steps

- Expand data-quality checks and automated tests.
- Evaluate segmentation stability across different samples and seeds.
- Validate lapse prediction on independent data and monitor precision-recall trade-offs.
- Expand ticket-triage evaluation with additional labelled examples and difficult edge cases.
- Improve token-budget monitoring and fallback reporting.
- Expand RAG evaluation with more answerable, unanswerable, ambiguous, and conflicting-policy questions.
- Strengthen refund-agent recovery, audit logging, and duplicate-action protection before any real integration.
- Add authentication, authorisation, and human approval workflows before connecting to a production support system.

## Conclusion

The Instamart Customer Intelligence & Support Copilot brings together data engineering, SQL analysis, machine learning, text retrieval, and a guarded agent workflow in one reproducible educational project.

The repository demonstrates customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and simulated refund decisions. Its results and safeguards are documented for learning and evaluation, not for production customer support or real financial transactions.
