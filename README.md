# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data engineering, SQL analytics, customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and a simulated refund agent.

The project is divided into four parts:

- **Part A — Data Foundation:** Clean and validate data, create a SQLite database, run SQL queries, and perform exploratory data analysis.
- **Part B — Customer Intelligence:** Generate RFM features, segment customers using K-Means, predict customer lapse, evaluate models, and create a retention list.
- **Part C — Support Ticket Triage:** Redact personally identifiable information (PII), classify tickets using zero-shot and few-shot prompting, validate responses, and enforce token-budget and routing guardrails.
- **Part D — Policy RAG Assistant & Refund Agent:** Retrieve relevant policy information, evaluate retrieval and fallback behavior, and run a simulated tool-using refund agent with checkpoint and resume support.

All data is synthetic. The refund agent simulates refund decisions and records; it does not transfer real money.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- Python packages listed in `requirements.txt`
- Git and GitHub for version control

### Setup and execution

Open PowerShell in the repository root directory.

**1. Create a virtual environment**

```powershell
python -m venv .venv
```

**2. Activate the virtual environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```powershell
python -m pip install -r requirements.txt
```

**4. Generate synthetic data**

```powershell
python generate_data.py
```

**5. Clean the data**

```powershell
python src/clean_data.py
```

**6. Build the SQLite database**

```powershell
python src/build_database.py
```

**7. Validate the database**

```powershell
python src/check_database.py
```

Expected final validation message:

```text
ALL CHECKPOINTS PASSED
```

**8. Inspect the data**

```powershell
python src/inspect_data.py
```

**9. Run SQL analysis**

```powershell
python src/sql_analysis.py
```

**10. Generate customer features**

```powershell
python src/customer_features.py
```

**11. Run customer segmentation**

```powershell
python src/customer_segmentation.py
```

**12. Run lapse prediction**

```powershell
python src/lapse_prediction.py
```

**13. Run support ticket triage**

```powershell
python src/ticket_triage.py
```

**14. Run policy RAG evaluation**

```powershell
python src/policy_rag.py
```

**15. Run the refund agent**

```powershell
python src/refund_agent.py
```

**16. Run automated tests**

```powershell
python -m pytest -v
```

Run the commands from the repository root and in the order shown. Some scripts generate or overwrite output files and checkpoint artifacts.

### Important files and outputs

- `generate_data.py` — synthetic data generation.
- `mock_llm.py` — mock language-model behavior.
- `data/raw/` — generated raw datasets.
- `data/clean/` — cleaned datasets and customer features.
- `data/policies/` — policy documents used by the RAG assistant.
- `data/instamart.db` — SQLite database.
- `sql/queries.sql` — SQL analysis queries.
- `outputs/customer_segments.csv` — customer segment assignments.
- `outputs/kmeans_scores.csv` — K-Means evaluation scores.
- `outputs/lapse_model.joblib` — saved lapse prediction model.
- `outputs/retention_list.csv` — retention candidate list.
- `outputs/tickets_redacted.csv` — redacted ticket data.
- `outputs/triage_results.csv` — final ticket triage results.
- `outputs/triage_zero_shot.csv` — zero-shot classification results.
- `outputs/triage_few_shot.csv` — few-shot classification results.
- `outputs/triage_budget_demo.csv` — token-budget demonstration.
- `outputs/ticket_triage_summary.json` — triage summary.
- `outputs/policy_rag_results.csv` — policy evaluation results.
- `outputs/policy_rag_overlap_comparison.csv` — chunk-overlap comparison.
- `outputs/policy_rag_summary.json` — policy evaluation summary.
- `outputs/refund_agent_demo.json` — refund agent demonstration.
- `outputs/refund_agent_checkpoint.json` — saved agent checkpoint.
- `outputs/refund_agent_scenarios.csv` — scenario outcomes.
- `outputs/refund_ledger.jsonl` — refund tool execution records.
- `outputs/agent_runs.jsonl` — agent run logs.
- `state/` — checkpoint evidence used for crash/resume testing.

## Part A — Data Foundation

### Data generation

The project generates synthetic customer, product, order, ticket, agent-scenario, and policy-evaluation data. The original `generate_data.py` and `mock_llm.py` files are used without modification.

The raw datasets initially contained 1,200 customers, 40 products, 13,084 order rows, and 164 tickets.

### Data cleaning

The cleaning pipeline standardizes data types and values, removes duplicate orders, handles missing values, validates identifiers, and prepares datasets for analysis.

The cleaning rules include:

- Standardizing city and gender values.
- Handling missing or invalid customer attributes.
- Removing exact duplicate order records.
- Keeping the latest record for each order ID.
- Converting order amounts and delivery times to numeric values.
- Handling missing order amounts using available product, quantity, and discount information.
- Treating negative delivery times as invalid.
- Imputing missing delivery times for completed orders using product-category medians.
- Keeping delivery time missing for cancelled orders where a delivery did not take place.
- Excluding pending orders from the cleaned analytical order dataset.

### Cleaned data results

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Tickets | 164 |

The cleaned orders contain:

| Order status | Count |
|---|---:|
| Completed | 11,110 |
| Cancelled | 1,199 |
| Pending | 0 |
| **Total** | **12,309** |

Completed-order revenue is INR 2,998,668.80.

| Data-quality check | Result |
|---|---:|
| Missing order amounts | 0 |
| Completed orders with missing delivery time | 0 |
| Cancelled orders with missing delivery time | 1,199 |
| Duplicate order IDs | 0 |
| Invalid order statuses | 0 |
| Missing customer references in orders | 0 |

Cancelled orders have missing delivery times because a completed delivery duration is not available for those orders.

### SQLite database

The cleaned data is loaded into `data/instamart.db`.

The database validation script checks the expected row counts, unique customer/product/order identifiers, valid order statuses, missing customer references, and negative order amounts.

Validation result:

```text
ALL CHECKPOINTS PASSED
```

### SQL analysis

The project includes seven SQL queries in `sql/queries.sql`.

| Query | Purpose | Result |
|---|---|---:|
| Q1 | Customer spending analysis | 30 rows |
| Q2 | Completed revenue by city and category | 30 rows |
| Q3 | Products priced above the average product price | 11 rows |
| Q4 | Customers without completed orders | 25 rows |
| Q5 | Customers above average completed spending | 550 rows |
| Q6 | City-level spending ranks compared with row numbers | 8 differing rows |
| Q7 | Daily and cumulative completed revenue | 183 days |

The highest-revenue city/category combination is Bengaluru/Electronics, with INR 233,909.45.

The final cumulative revenue from Q7 is INR 2,998,668.80.

The Q6 query demonstrates the difference between rank-based ranking and unique row numbering. When multiple customers have tied spending, `RANK()` can assign the same rank, while `ROW_NUMBER()` assigns a different sequence number to each row.

### Exploratory data analysis

Completed-order amounts have these descriptive statistics:

| Metric | Value |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| First quartile | INR 64.80 |
| Third quartile | INR 297.00 |
| Skewness | 4.61 |
| Upper outlier fence | INR 645.30 |
| Orders above upper fence | 897 |

The distribution is strongly right-skewed. A relatively small number of high-value orders increase the mean above the median.

The project also compares SQL aggregations with pandas aggregations to cross-check analytical results.

Correlation analysis helps identify relationships between numerical variables. Correlation does not establish causation, and weak correlations should not be interpreted as proof that one variable causes changes in another.

## Part B — Customer Segments & Lapse Prediction

### RFM feature engineering

The project calculates recency, frequency, and monetary value from completed-order history.

- **Recency:** Days since the customer's most recent completed order.
- **Frequency:** Number of completed orders.
- **Monetary value:** Total spending on completed orders.

The as-of date is 1 October 2025. Customers without completed orders are retained separately rather than being assigned misleading RFM values.

The RFM cohort contains 1,175 customers. The remaining 25 customers have no completed orders.

| RFM metric | Value |
|---|---:|
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 completed orders |
| Maximum frequency | 33 completed orders |
| Mean monetary value | INR 2,552.06 |
| Maximum monetary value | INR 14,766.55 |

### K-Means segmentation

RFM features are standardized before clustering. K-Means is evaluated for K values from 2 to 8 using `n_init=10` and `random_state=42`.

The actual scores recorded in `outputs/kmeans_scores.csv` are:

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.7 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.0 | 0.525 |
| 5 | 617.5 | 0.478 |
| 6 | 503.2 | 0.483 |
| 7 | 408.5 | 0.416 |
| 8 | 366.2 | 0.412 |

Inertia decreases as K increases because additional clusters can reduce the distance between observations and their cluster centres. Therefore, the lowest inertia alone does not identify the best number of clusters.

K=4 has the highest silhouette score among the evaluated configurations. K=4 was selected based on the evaluation results and the interpretability of the resulting customer segments.

### Final customer segments

The segmentation output contains all 1,200 customers.

| Segment | Customers |
|---|---:|
| Regular Customers | 436 |
| High Value | 419 |
| Low Spend | 183 |
| Top Value | 137 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The premium-customer flag identifies the Top Value segment for support prioritization. Cluster IDs are algorithmic identifiers and do not inherently represent a business ranking.

The output is saved in `outputs/customer_segments.csv`.

### Lapse prediction

The lapse prediction task uses a cutoff of 1 September 2025 and a September 2025 label window.

The eligible lapse cohort contains 1,156 customers.

| Lapse label | Customers |
|---|---:|
| Not Lapsed | 713 |
| Lapsed | 443 |
| **Total** | **1,156** |

The lapsed proportion is approximately 38.32%.

The dataset is split into 924 training rows and 232 testing rows. Stratification and `random_state=42` are used.

Features are constructed from information available before the cutoff. They include historical recency, frequency, monetary value, discount, cancellation rate, delivery time, product category, city, age, gender, and customer tenure.

Post-cutoff orders and September outcomes are excluded from the model features. Preprocessing is fitted on the training data only. Categorical variables are one-hot encoded, and numerical features are scaled.

### Model comparison

An always-not-lapsed baseline achieves approximately 61.6% accuracy but an F1 score of zero for the lapsed class.

Logistic Regression and Random Forest are evaluated. Random Forest is selected based on the model-selection results.

| Model | Test accuracy | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.728 | 0.64 | 0.65 | 0.65 |
| Random Forest | 0.737 | 0.66 | 0.65 | 0.66 |

The selected Random Forest configuration uses `max_depth=8` and `min_samples_leaf=10`.

The Random Forest confusion matrix is:

```text
[[113, 30],
 [ 31, 58]]
```

Rows represent actual labels, and columns represent predicted labels. The first row represents customers who were not lapsed; the second row represents customers who were lapsed.

False negatives are lapsed customers predicted as not lapsed. False positives are customers predicted as lapsed who did not lapse in the label window. Both types of error affect the cost and effectiveness of retention campaigns.

### Oversampling experiment

The training set is oversampled to balance the classes. The test set remains unchanged.

| Evaluation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.644 | 0.652 | 0.648 |
| Oversampled training distribution | 0.589 | 0.854 | 0.697 |

Oversampling improves recall in this experiment but reduces precision. This trade-off matters when deciding how many customers to contact in a retention campaign.

The model is saved and reloaded, and predictions are checked for consistency after reloading.

The top Random Forest features include frequency, monetary value, recency, average delivery time, and cancellation rate.

### Retention list

`outputs/retention_list.csv` contains the top 10 eligible test-set premium customers ranked by predicted lapse probability. One of these customers was actually lapsed in the test labels.

The list is illustrative. The premium flag comes from the full-window segmentation output, which may use information from the label period. Therefore, the retention-list selection is not fully leakage-free, even though the lapse model features use pre-cutoff information.

## Part C — Support Ticket Triage with Guardrails

### Input validation

The pipeline processes 164 tickets.

Four empty or whitespace-only tickets are skipped, and four tickets exceeding the configured 400-token limit are flagged. These tickets are not sent to the model for classification.

### PII redaction

The pipeline redacts email addresses, Indian mobile phone numbers, and order IDs before model calls.

| PII type | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total substitutions** | **278** |

The audit found zero matching email addresses, phone numbers, or order IDs remaining after redaction.

The mock model receives the redacted ticket text rather than the original text. Regex-based redaction is a practical safeguard for this synthetic dataset, but it cannot guarantee detection of every possible personal-information format.

### Classification and response validation

The permitted ticket categories are:

- `billing`
- `delivery`
- `fraud`
- `technical`

The pipeline evaluates zero-shot and few-shot prompts. Both prompts require a JSON-only response containing an allowed category. Few-shot prompting also provides examples to guide the classification task.

Model responses are validated before routing.

| Status | Meaning |
|---|---|
| `classified_json` | Valid JSON with an allowed category |
| `classified_keyword` | Invalid JSON, but one allowed category can be extracted |
| `flagged_invalid_output` | Output cannot be safely interpreted |
| `skipped_empty` | Ticket text is empty |
| `flagged_long` | Ticket exceeds the input limit |
| `budget_exhausted` | Token budget prevents another model call |

The retry policy permits at most three attempts, with exponential backoff and jitter. The mock model is reset before each complete experimental run.

### Zero-shot and few-shot evaluation

There are 156 eligible tickets for model classification after input guardrails.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Model calls reported | 156 | 156 |
| Total tokens | 11,584 | 22,660 |
| Valid JSON classifications | 135 | 135 |
| Keyword classifications | 12 | 12 |
| Classified tickets | 147 | 147 |
| Invalid outputs | 9 | 9 |
| Empty tickets | 4 | 4 |
| Overlength tickets | 4 | 4 |
| Accuracy on classified tickets | 85.7% | 100% |
| Tokens per classified ticket | 78.80 | 154.15 |

The zero-shot confusion matrix is:

```text
                 Predicted
Actual       billing delivery fraud technical
billing          38       0      0        0
delivery          0      36      0        0
fraud             11       0     23        0
technical         10       0      0       29
```

Zero-shot errors mainly involve fraud and technical tickets being classified as billing. Few-shot examples improve accuracy in this synthetic evaluation but use more tokens.

Accuracy is calculated over the 147 classified tickets, not all 164 tickets.

### Token-budget demonstration

The token-budget demonstration uses a limit of 6,000 tokens. The pipeline checks the running token count before making another model call.

The recorded budget demonstration used 5,951 tokens and exercised the budget guardrail. The output is saved in `outputs/triage_budget_demo.csv`.

### Ticket routing

The final triage output is saved in `outputs/triage_results.csv`.

Routing rules are:

- Empty, overlength, invalid-output, and budget-exhausted tickets are routed to `human_review` with normal priority.
- Fraud tickets are routed to `security_team` with high priority.
- Premium-customer tickets are routed to `premium_desk` with high priority.
- Other tickets are routed to `support_queue` with normal priority.
- Missing customer-segment information is treated as non-premium.

Routing uses validated model output and customer-segment information rather than blindly trusting raw model responses.

## Part D — Policy RAG Assistant & Refund Agent

### Policy documents and retrieval

The policy assistant loads five text documents from `data/policies/`.

Documents are split into 40-word chunks with 10-word overlap, producing 15 chunks. TF-IDF vectors represent the chunks and questions. Cosine similarity ranks chunks by relevance, and the default configuration retrieves the top three chunks.

TF-IDF is a lightweight retrieval method based on word importance. Cosine similarity compares the direction of the question and chunk vectors. This approach is not a semantic embedding model and may be sensitive to wording and chunk boundaries.

### Policy evaluation

The policy evaluation contains 12 answerable questions and two unanswerable questions.

| Evaluation | Result |
|---|---|
| Answerable questions passed | 12/12 |
| Unanswerable questions correctly handled | 2/2 |
| Overall evaluation | All checks passed |

The answerable questions cover late-delivery refunds, failed payments, fraud escalation, privacy, payment methods, delivery fees, account deletion, and related policy rules.

When the retrieved context does not support an answer, the assistant uses a fallback response:

> I could not find sufficient information in the policy documents. Please contact a human support agent.

This prevents the assistant from inventing policy information when evidence is insufficient.

### Chunk-overlap comparison

The project compares retrieval with 10-word overlap against retrieval without overlap, using top-K values of 3, 5, and 8.

At top-K 3, both approaches retrieved the expected document and expected fact for 11 of 12 questions. Without overlap, the failed-payment refund question did not retrieve the expected document or the expected 48-hour fact.

At top-K 5, the expected payment document was retrieved, but the required fact was still absent from the retrieved context. Finding the correct document does not guarantee that the specific answer is present in the selected chunks.

The comparison demonstrates that overlap alone does not improve every question. Retrieval performance also depends on query wording, chunk boundaries, and ranking.

Results are saved in:

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_summary.json`
- `outputs/policy_rag_overlap_comparison.csv`

### Refund-agent tools

The simulated refund agent uses the following tools:

- `get_order` — retrieve order details.
- `search_policy` — retrieve relevant policy context.
- `calculate_refund` — check eligibility and calculate the simulated refund.
- `issue_refund` — record a simulated refund.
- `escalate_to_human` — hand off cases that require human review.

The agent enforces configured policy guardrails, including:

- Late-delivery eligibility requires delivery time greater than 45 minutes.
- Refund requests must fall within the seven-day request window.
- Refunds above INR 1,000 require human approval.

Tool execution is recorded in `outputs/refund_ledger.jsonl`. Agent runs are logged in `outputs/agent_runs.jsonl`.

The demonstration returns a simulated refund of INR 58 for an order delivered in 70 minutes, with a request age of three days. It completes four tool steps and reports five LLM calls. No real money is transferred.

### Refund-agent scenario results

The scenario output covers seven test scenarios.

| Scenario | Expected outcome |
|---|---|
| S1 — Eligible late-delivery refund | Refund issued |
| S2 — Eligible refund | Refund issued |
| S3 — Refund above automatic approval limit | Escalation |
| S4 — Delivery within 45 minutes | Denied |
| S5 — Order not completed | Denied |
| S6 — Unknown order | Not found |
| S7 — Iteration limit reached | Fallback after maximum iterations |

The expected refund amounts for S1 and S2 are INR 128.25 and INR 135.00, respectively.

Additional demonstrations exercise timeout and token-limit guardrails. The agent returns `FALLBACK_TIMEOUT` when the timeout limit is reached and `FALLBACK_TOKEN_LIMIT` when the token limit is reached.

The scenario artifacts are saved in `outputs/refund_agent_scenarios.csv` and related JSON outputs.

### Checkpointing and resume

The agent stores checkpoint information under `state/` and writes demo checkpoint artifacts under `outputs/`.

The crash/resume demonstration records the following tool sequence for the S1 refund:

```text
get_order → search_policy → calculate_refund → issue_refund
```

The completed refund result is INR 128.25.

The intended resume behavior is to restore saved state and continue from the last completed tool observation instead of repeating completed actions. Exactly-once execution should be confirmed by comparing the tool-execution logs from the interrupted and resumed runs.

### ReAct versus plan-and-execute

The ReAct-style workflow alternates between reasoning, selecting a tool, observing its result, and deciding what to do next. This supports conditional decisions based on order status, policy retrieval, refund eligibility, and unexpected tool results.

A plan-and-execute workflow creates a plan first and then executes the planned steps. This can make a workflow easier to inspect, but it may need additional handling when a tool result differs from the plan.

The ReAct approach supports dynamic decisions and checkpointing, but may require multiple model calls and additional tokens. Iteration, token, and time limits are therefore important guardrails.

The implementation is a synthetic demonstration, not a production payment system.

## Limitations and Next Steps

- The data and policies are synthetic and do not establish real-world performance.
- Customer segments and lapse predictions depend on the chosen as-of date, label window, and feature definitions.
- The retention list is illustrative because the premium flag uses full-window segmentation output.
- Lapse predictions may produce false positives and false negatives.
- PII redaction uses regex patterns and does not guarantee detection of every sensitive-data format.
- Few-shot classification uses more tokens than zero-shot classification in the measured experiment.
- TF-IDF retrieval can miss facts because of query wording, chunk boundaries, and ranking.
- The policy rules and refund limits are configured for this capstone and must not be treated as actual Instamart policies.
- Refund issuance is simulated; no real payment is made.
- Checkpoint and append-only log files may contain records from repeated test runs. The final artifacts should be checked for consistency and reproducibility.

Before submission, verify the public GitHub repository, run the project from a fresh clone, confirm the validation output, run automated tests, and ensure that generated artifacts match the documented results.