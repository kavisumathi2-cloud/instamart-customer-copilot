# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data engineering, SQL analytics, customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and a simulated refund agent.

The project is organized into four parts:

- **Part A — Data Foundation:** Clean and validate customer, product, order, and ticket data; build a SQLite database; perform SQL analysis and exploratory data analysis.
- **Part B — Customer Intelligence:** Create RFM features, segment customers using K-Means, predict customer lapse, compare models, and prepare a retention list.
- **Part C — Support Ticket Triage:** Redact personally identifiable information, classify tickets using zero-shot and few-shot prompts, validate model responses, enforce token budgets, and route tickets.
- **Part D — Policy RAG and Refund Agent:** Retrieve policy information from documents, evaluate retrieval and fallback behavior, and run a simulated tool-using refund agent with checkpoints and resume support.

All customer, product, order, ticket, and policy-evaluation data are synthetic. The refund agent records simulated refunds only; it does not transfer real money.

## Reproduce

### Requirements

- Python 3.11 or later
- SQLite 3.25 or later
- Python packages listed in `requirements.txt`
- Git and GitHub for repository version control

### Setup

Clone or download this repository, then open a terminal in the project root.

Create and activate a virtual environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Generate and prepare the data

Run the data-generation and cleaning scripts in the order supported by the repository:

```powershell
python generate_data.py
python src/clean_data.py
python src/build_database.py
python src/check_database.py
```

The validation script should finish with:

```text
ALL CHECKPOINTS PASSED
```

### Run the analysis and models

Run the project scripts from the repository root:

```powershell
python src/inspect_data.py
python src/sql_analysis.py
python src/customer_features.py
python src/customer_segmentation.py
python src/lapse_prediction.py
python src/ticket_triage.py
python src/policy_rag.py
python src/refund_agent.py
```

Run automated tests:

```powershell
python -m pytest
```

The project was previously tested with **12 passing tests**. Run the tests again after a fresh clone to verify the current repository state.

### Main generated artifacts

Important generated files include:

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/tickets_redacted.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_confusion_zero_shot.csv`
- `outputs/ticket_triage_summary.json`
- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/policy_rag_summary.json`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`

The `state/` directory stores agent checkpoint files used for resuming interrupted runs. Some generated files are created or overwritten when scripts are rerun.

## Part A — Data Foundation

### Data generation and cleaning

The project uses synthetic customer, product, order, ticket, agent-scenario, and policy-evaluation data.

The original data-generation and mock-LLM scripts are used without modification.

The raw orders dataset initially contained 13,084 rows. Cleaning removed 119 exact duplicate rows and retained the latest order record for each order ID, producing 12,309 cleaned orders.

The cleaned dataset contains:

| Dataset | Rows |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Tickets | 164 |

### Data quality checks

The cleaning and validation process checks identifiers, duplicate records, numeric amounts, order statuses, missing values, delivery times, customer attributes, and database integrity.

Customer cleaning includes:

- Standardizing city values.
- Standardizing gender values and representing unknown gender explicitly.
- Handling missing ages and restricting cleaned ages to 18–60.

Order cleaning includes:

- Removing exact duplicate records.
- Keeping the latest record per order ID.
- Converting order amounts and delivery times to numeric values.
- Filling missing amounts using product price, quantity, and discount information where applicable.
- Treating negative delivery times as missing.
- Imputing missing delivery times for completed orders using product-category medians.
- Keeping delivery time missing for cancelled orders and removing pending orders from the cleaned analytical dataset.

### Cleaned data results

The cleaned orders contain 11,110 completed orders and 1,199 cancelled orders.

| Check | Result |
|---|---:|
| Cleaned orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Missing order amounts | 0 |
| Completed orders with missing delivery time | 0 |
| Cancelled orders with missing delivery time | 1,199 |
| Completed-order revenue | ₹2,998,668.80 |

Customer city counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, and Mumbai 184.

The cleaned gender counts are female 610, male 560, and unknown 30.

### SQLite database

The project builds `data/instamart.db` with customer, product, and order tables. The validation script checks the expected row counts, identifier uniqueness, valid order statuses, missing customer identifiers, and negative order amounts.

The validation output ends with `ALL CHECKPOINTS PASSED`.

### SQL analysis

The seven SQL queries in `sql/queries.sql` cover customer spending, city/category revenue, product prices, customers without completed orders, above-average spending, city-level ranking, and cumulative daily revenue.

| Query | Result |
|---|---|
| Q1 — Customers with completed orders above ₹500 and qualifying total spending | 30 customers |
| Q2 — Completed revenue by city and product category | 30 city/category rows |
| Q3 — Products priced above the average product price | 11 products |
| Q4 — Customers without completed orders | 25 customers |
| Q5 — Customers above average completed-order spending | 550 customers |
| Q6 — City-level customer spending rankings | 8 rows where rank and row number differ |
| Q7 — Daily completed revenue and cumulative revenue | 183 days |

The total revenue in Q2 and the final cumulative revenue in Q7 both equal ₹2,998,668.80.

The highest-revenue city/category combination is Bengaluru/Electronics at ₹233,909.45.

Window functions demonstrate the difference between ranking with ties and assigning unique row numbers.

### Exploratory data analysis

Completed-order amounts have a mean of approximately ₹269.91 and a median of ₹135. The distribution is strongly right-skewed, with skewness of approximately 4.61.

The first quartile is ₹64.80 and the third quartile is ₹297. Using the 1.5-IQR rule, the upper outlier fence is ₹645.30, with 897 completed orders above that threshold.

The correlations between order amount and discount percentage, and between order amount and quantity, are relatively weak in this dataset. Correlation alone does not establish causation.

A pandas aggregation independently cross-checks the SQL city/category revenue results.

## Part B — Customer Segments & Lapse Prediction

### RFM features

The project calculates recency, frequency, and monetary value using an as-of date of 1 October 2025. Customers with completed-order history are used for RFM clustering, while customers without completed orders are retained separately.

The active RFM cohort contains 1,175 customers. The remaining 25 customers are labelled `No Completed Orders`.

Observed RFM statistics include:

| Metric | Result |
|---|---:|
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 completed orders |
| Maximum frequency | 33 completed orders |
| Mean monetary value | ₹2,552.06 |
| Maximum monetary value | ₹14,766.55 |

### K-Means segmentation

RFM features are standardized using `StandardScaler`. K-Means models are evaluated for K values from 2 to 8, with `n_init=10` and `random_state=42`.

The selected K-Means configuration uses K=4. The choice considers inertia, silhouette score, and business interpretability rather than selecting the largest K solely because it produces lower inertia.

The final segment output contains 1,200 customers.

| Segment | Customers |
|---|---:|
| Regular Customers | 436 |
| High Value | 419 |
| Low Spend | 183 |
| Top Value | 137 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The premium-customer flag identifies the Top Value segment for support prioritization. Cluster IDs are algorithmic labels; they do not have an inherent business ranking.

The generated `outputs/customer_segments.csv` includes customer ID, recency, frequency, monetary value, segment ID, segment name, and premium flag.

### Lapse prediction

The lapse model uses a cutoff of 1 September 2025. The label window covers September 2025. A customer is labelled lapsed when they have no completed orders during the label window and belong to the eligible historical cohort.

The lapse cohort contains 1,156 customers:

| Label | Customers |
|---|---:|
| Not Lapsed | 713 |
| Lapsed | 443 |

The lapsed proportion is approximately 38.32%.

The data is split into 924 training rows and 232 test rows using stratification and `random_state=42`.

Features are constructed from information available before the cutoff. They include historical recency, frequency, monetary value, discount, cancellation rate, delivery time, product category, city, age, gender, and customer tenure. Orders after the cutoff and September outcomes are excluded from the model features.

Preprocessing is fitted on training data only. Categorical features are one-hot encoded, and numerical features are scaled.

### Baseline and model comparison

An always-not-lapsed baseline achieves approximately 61.6% accuracy but has an F1 score of zero for the lapsed class.

Logistic Regression and Random Forest are evaluated using five-fold cross-validation with F1 as the scoring metric. Random Forest is selected based on the model-selection results.

| Model | Test accuracy | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Tuned Logistic Regression | 0.728 | 0.64 | 0.65 | 0.65 |
| Tuned Random Forest | 0.737 | 0.66 | 0.65 | 0.66 |

The selected Random Forest configuration uses `max_depth=8` and `min_samples_leaf=10`.

The Random Forest test confusion matrix is:

```text
[[113, 30],
 [ 31, 58]]
```

The first row represents customers who were not lapsed, and the second row represents customers who were lapsed. The columns represent predicted not-lapsed and predicted lapsed labels, respectively.

False negatives are lapsed customers who are missed by the model. False positives are customers predicted to lapse who do not lapse in the label window. Both matter for retention campaigns because missed customers may be lost, while false positives can waste outreach effort.

### Oversampling experiment

The training data is oversampled to balance the two classes. The test data remains unchanged.

| Logistic Regression evaluation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.644 | 0.652 | 0.648 |
| Oversampled training distribution | 0.589 | 0.854 | 0.697 |

Oversampling improves recall in this experiment but reduces precision. This trade-off is important when deciding how many customers to target in a retention campaign.

The model is saved and reloaded, and the project checks that predictions match after reloading.

The top Random Forest features include frequency, monetary value, recency, average delivery time, and cancellation rate.

### Retention list

`outputs/retention_list.csv` contains the top 10 eligible test-set premium customers ranked by predicted lapse probability. One of the 10 customers was actually lapsed in the test labels.

This list is illustrative. The premium flag is derived from the full-window segmentation output, which can include September information. Consequently, the retention-list selection is not fully leakage-free even though the lapse model features themselves use only pre-cutoff information.

## Part C — Support Ticket Triage with Guardrails

### Input validation

The ticket triage pipeline processes 164 tickets.

The input guardrails skip empty or whitespace-only tickets and flag tickets exceeding the configured 400-token limit. Four tickets are skipped as empty and four are flagged as too long.

These tickets are not sent to the model for classification.

### PII redaction

The pipeline redacts email addresses, Indian mobile phone numbers, and order IDs before model calls.

| PII type | Redactions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total substitutions** | **278** |

The audit found zero matching email addresses, phone numbers, or order IDs remaining after redaction.

The model's user message contains the redacted ticket text only. The redaction patterns are practical guardrails for the synthetic dataset and are not a guarantee that every possible form of personal information will be detected.

### Classification and response validation

The four permitted ticket categories are billing, delivery, fraud, and technical.

Both zero-shot and few-shot prompts specify the allowed categories and require a JSON-only response schema. The few-shot prompt also includes examples covering the classification task, including a fraud example.

The pipeline validates model output and records the response status:

- `classified_json` — valid JSON containing an allowed category.
- `classified_keyword` — the response is not valid JSON, but exactly one allowed category can be extracted.
- `flagged_invalid_output` — the output cannot be safely interpreted.
- `skipped_empty` — empty input.
- `flagged_long` — ticket exceeds the input limit.
- `budget_exhausted` — the token budget prevents another model call.

The retry policy allows at most three attempts with exponential backoff and jitter. The mock model is reset before each complete experimental run.

### Zero-shot and few-shot evaluation

There are 156 eligible tickets for model classification after input guardrails.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Model calls | 156 | 156 |
| Total tokens | 11,584 | 22,660 |
| Valid JSON classifications | 135 | 135 |
| Keyword classifications | 12 | 12 |
| Total classified | 147 | 147 |
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

The zero-shot errors mainly involve fraud and technical tickets being classified as billing. In this synthetic evaluation, few-shot examples improve classification accuracy at the cost of additional tokens.

Accuracy is calculated on the 147 classified tickets, not on all 164 tickets.

### Token budget demonstration

The budget demonstration uses a 6,000-token limit. The model is not called once the running total has reached the configured budget.

The generated budget demonstration used 5,951 tokens and reported the budget guardrail in its output. The corresponding results are saved in `outputs/triage_budget_demo.csv`.

### Final ticket routing

The final few-shot output is saved as `outputs/triage_results.csv`.

Routing follows these rules:

- Empty, overlength, invalid-output, and budget-exhausted tickets go to `human_review` with normal priority.
- Fraud tickets go to `security_team` with high priority.
- Premium-customer tickets go to `premium_desk` with high priority.
- Other tickets go to `support_queue` with normal priority.
- Missing customer-segment information is treated as non-premium.

The final routing decision uses the validated category and customer segment rather than blindly trusting raw model output.

## Part D — Policy RAG Assistant & Refund Agent

### Policy documents and chunking

The policy assistant loads five text documents from `data/policies/`.

Documents are split into 40-word chunks with 10-word overlap. The resulting index contains 15 chunks.

TF-IDF vectors are used to represent the chunks and questions. Cosine similarity ranks chunks by their relevance to a question. The top three retrieved chunks are used for the default retrieval configuration.

TF-IDF emphasizes terms that are frequent within a document but relatively distinctive across the document collection. Cosine similarity compares the direction of the question and chunk vectors. This is a lightweight retrieval approach, not a semantic embedding model.

### Policy evaluation and fallback behavior

The evaluation contains 12 answerable questions and two questions that cannot be answered from the policy documents.

| Evaluation | Result |
|---|---:|
| Answerable questions passed | 12/12 |
| Unanswerable questions correctly handled | 2/2 |
| All evaluations passed | True |

The answerable cases cover late-delivery refunds, failed payments, fraud escalation, privacy, payment methods, delivery fees, account deletion, and other policy rules.

For questions without sufficient relevant policy context, the assistant returns:

> I could not find sufficient information in the policy documents. Please contact a human support agent.

The fallback avoids inventing a policy answer when the indexed documents do not support one.

### Chunk overlap comparison

The project compares retrieval using 10-word overlap with retrieval using no overlap. The comparison tests 12 answerable questions at top-K values of 3, 5, and 8.

At top-K 3, both strategies retrieved the expected document and expected fact for 11 of 12 questions. Without overlap, the failed-payment refund question did not retrieve the expected document or the expected `48 hours` fact.

At top-K 5, the expected payment document was retrieved, but the expected fact was still absent from the retrieved context. This demonstrates that retrieving a relevant document does not necessarily mean the required answer is present in the selected chunks.

Overlap alone did not improve every question. Retrieval quality also depends on chunk boundaries, query wording, and ranking.

The detailed comparison is saved in `outputs/policy_rag_overlap_comparison.csv`. The main evaluation and summary are saved in `outputs/policy_rag_results.csv` and `outputs/policy_rag_summary.json`.

### Refund-agent tools

The refund agent uses a tool-based workflow. Its tool specification and tool mapping define the available actions:

- `get_order` — retrieve order details.
- `search_policy` — retrieve relevant policy context.
- `calculate_refund` — check refund eligibility and calculate the simulated refund.
- `issue_refund` — record a simulated refund.
- `escalate_to_human` — hand off cases that require human review.

Tool execution is logged in `outputs/refund_ledger.jsonl`. Agent runs are recorded in `outputs/agent_runs.jsonl`.

The refund agent enforces the configured policy guardrails, including the late-delivery threshold of more than 45 minutes, the seven-day request window, and human approval for refunds above ₹1,000.

The refund demo returns a simulated refund of ₹58 for an order delivered in 70 minutes, with a request age of three days. It completes four tool steps and reports five LLM calls. The final response explicitly states that no real money was transferred.

### Agent scenario results

The logged agent scenarios include the following expected outcomes:

| Scenario | Expected result |
|---|---|
| S1 — Eligible late-delivery refund | `REFUND: INR 128.25 issued.` |
| S2 — Eligible refund | `REFUND: INR 135.0 issued.` |
| S3 — Refund above automatic approval limit | `ESCALATE` |
| S4 — Delivery within 45 minutes | `DENIED` |
| S5 — Order not completed | `DENIED` |
| S6 — Unknown order | `NOT_FOUND` |
| S7 — Iteration limit reached | `FALLBACK_MAX_ITERATIONS` |

The agent also demonstrates safe fallback behavior for a zero-second timeout and a depleted token budget:

- `FALLBACK_TIMEOUT`
- `FALLBACK_TOKEN_LIMIT`

The agent stops when a configured limit is reached rather than continuing indefinitely.

### Checkpointing and resume

The agent saves checkpoint state under `state/` and writes demo checkpoint artifacts under `outputs/`.

The crash/resume demonstration verifies a completed S1 refund result of ₹128.25 and records the tool sequence:

`get_order → search_policy → calculate_refund → issue_refund`

The intended resume behavior is to restore the saved state and continue from the last completed tool observation instead of executing completed actions again. The tool sequence in the logged record is consistent with the expected workflow; confirming exactly-once execution also requires comparing tool executions across the crash and resumed portions of the run.

### ReAct versus plan-and-execute

The ReAct-style agent alternates between reasoning, selecting a tool, observing its result, and deciding what to do next. This lets it adapt to order status, policy results, refund eligibility, and unexpected tool responses.

A plan-and-execute approach creates a plan first and then performs the planned actions. It can make the overall workflow easier to inspect, but a fixed plan may need extra handling when a tool returns an unexpected result.

For this project, the tool-driven ReAct workflow supports conditional decisions and checkpointing. Its trade-offs include multiple model calls, token usage, and the need for strict iteration, token, and time limits. The implementation is a simulated demonstration, not a production payment system.

## Limitations and Next Steps

- The datasets and policies are synthetic. Metrics do not establish real-world performance.
- The segmentation and lapse results depend on the selected as-of date, label window, and feature definitions.
- The retention list is illustrative because the premium-customer flag is derived from the full-window segmentation output.
- Lapse prediction can produce false positives and false negatives. Retention decisions should account for outreach cost and customer experience.
- Regex-based PII redaction does not guarantee detection of every sensitive-data format.
- Few-shot classification uses more tokens than zero-shot classification in the measured experiment.
- TF-IDF retrieval is sensitive to wording and chunk boundaries. A relevant source document can be retrieved without the exact required fact appearing in the retrieved context.
- Policy rules and refund limits are configured for this synthetic capstone and must not be treated as actual Instamart policy.
- Refund issuance is simulated. No real payments are made.
- Checkpoints and append-only logs may contain repeated records from multiple test runs; final submitted artifacts should be checked for consistency and reproducibility.
- Before final submission, run the project from a fresh clone, rerun the tests, verify the generated outputs, and ensure the public GitHub repository contains the required source files and documentation.
