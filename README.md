# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an end-to-end customer intelligence and support copilot using synthetic Instamart-style e-commerce data.

The project combines:

- Data cleaning and validation
- SQLite database engineering and SQL analytics
- RFM-based customer segmentation
- Customer lapse prediction
- Privacy-aware support ticket triage
- Policy retrieval using TF-IDF
- A guarded, simulated refund agent

The implementation is organised into four parts:

- **Part A — Data Foundation:** Clean, validate, store, and analyse the source data.
- **Part B — Customer Intelligence:** Segment customers and predict customer lapse.
- **Part C — Support Ticket Triage:** Redact personal information, classify tickets, validate outputs, and enforce token-budget guardrails.
- **Part D — Policy RAG and Refund Agent:** Retrieve policy information and demonstrate a guarded refund workflow.

**Important:** All project data is synthetic and fictional. The supplied `mock_llm.py` provides simulated language-model responses. Refund actions are demonstrations only; no real money is transferred.

### Technologies Used

- Python 3.11+
- SQLite 3.25+
- pandas
- NumPy
- scikit-learn
- joblib
- pytest

The supplied `generate_data.py` and `mock_llm.py` files must remain unmodified.

### Repository Structure

```text
instamart-customer-copilot/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”œâ”€â”€ clean/
â”‚   â”œâ”€â”€ policies/
â”‚   â””â”€â”€ instamart.db
â”œâ”€â”€ outputs/
â”œâ”€â”€ sql/
â”‚   â””â”€â”€ queries.sql
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ build_database.py
â”‚   â”œâ”€â”€ check_database.py
â”‚   â”œâ”€â”€ clean_data.py
â”‚   â”œâ”€â”€ customer_features.py
â”‚   â”œâ”€â”€ customer_segmentation.py
â”‚   â”œâ”€â”€ inspect_data.py
â”‚   â”œâ”€â”€ lapse_prediction.py
â”‚   â”œâ”€â”€ policy_rag.py
â”‚   â”œâ”€â”€ refund_agent.py
â”‚   â”œâ”€â”€ refund_agent_backup.py
â”‚   â”œâ”€â”€ sql_analysis.py
â”‚   â””â”€â”€ ticket_triage.py
â”œâ”€â”€ state/
â”œâ”€â”€ tests/
â”œâ”€â”€ check_clean_data.py
â”œâ”€â”€ check_order_stats.py
â”œâ”€â”€ check_sql_queries.py
â”œâ”€â”€ generate_data.py
â”œâ”€â”€ mock_llm.py
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

Generated files may appear in `data/`, `outputs/`, and `state/` after running the project.

## Reproduce

Run the following commands from the repository root in Windows PowerShell.

### Step 1 — Create a virtual environment

```powershell
python -m venv .venv
```

### Step 2 — Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pytest
```

### Step 4 — Generate synthetic data

```powershell
python generate_data.py
```

The generator creates synthetic customer, product, order, support-ticket, policy-evaluation, and agent-scenario data.

The original generator run reported:

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

The database is stored at:

```text
data/instamart.db
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

### Step 14 — Run data, database, and SQL checks

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

The documented validation run completed with **14 tests passed**.

Run the pipeline in the documented order when reproducing the results. Running the data generator again may overwrite generated datasets.

## Part A — Data Foundation

### 5.1 Raw Data Audit

The synthetic generator creates customer, product, order, support-ticket, policy-evaluation, and agent-scenario data.

The initial raw order dataset contained 13,084 rows. The audit checks dataset shapes, data types, missing values, duplicate records, invalid amounts, delivery-time problems, and inconsistent categorical values.

Recorded raw-data findings:

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending orders | 656 |
| Distinct order IDs | 12,309 |
| Unparseable order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Customers with blank ages | 61 |
| Customer ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Nonblank raw gender spellings | 10 |
| Blank gender values | 30 |

These findings describe the initial generated data before cleaning.

### 5.2 Customer Cleaning

Customer cleaning standardises categorical values, handles invalid ages, and parses signup dates.

The cleaning rules include:

- Standardise city names to six canonical cities.
- Map gender values to `female`, `male`, or `unknown`.
- Convert ages outside the allowed 10–100 range to missing values.
- Fill missing ages using the median age for the customer's city.
- Parse signup dates as datetime values.

Recorded city counts:

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |
| **Total** | **1,200** |

Recorded gender counts:

| Gender | Customers |
|---|---:|
| Female | 610 |
| Male | 560 |
| Unknown | 30 |
| **Total** | **1,200** |

The cleaned customer data contains no missing ages, and the recorded cleaned age range is 18–60.

### 5.3 Order Cleaning

The order-cleaning workflow applies these rules:

1. Parse order timestamps as datetime values.
2. Remove 119 exact duplicate rows.
3. Retain the latest record for each order ID.
4. Convert order amounts to numeric values.
5. Impute missing or unparseable amounts using product price × quantity × (1 − discount percentage).
6. Convert negative delivery times to missing values.
7. Impute missing delivery times for completed orders using the completed-order category median.
8. Keep cancelled-order delivery times missing where no valid delivery time exists.
9. Remove pending orders from the final cleaned order dataset.

The final cleaned datasets contain:

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

The cleaned order dataset contains 11,110 completed orders and 1,199 cancelled orders, with no pending orders.

Recorded completed-order revenue is **INR 2,998,668.80**.

### 5.4 Database Construction and Validation

The SQLite database is stored at `data/instamart.db`.

Database validation checks include:

- Expected table and record counts
- Unique customer, product, and order IDs
- Valid order statuses
- Valid customer references
- No missing order customer IDs
- No negative order amounts

The recorded database and cleaned-data validation scripts passed.

### 5.5 SQL Analysis

Seven SQL queries are provided in `sql/queries.sql`.

They demonstrate:

- Customer-level spending with grouping and filtering
- Revenue by city and product category
- Products priced above the average
- Customers without completed orders
- CTE-based comparison with average customer spending
- Window-function rankings
- Daily running revenue

Recorded SQL findings:

| Analysis | Result |
|---|---:|
| Completed-order revenue | INR 2,998,668.80 |
| Bengaluru Electronics revenue | INR 233,909.45 |
| Bengaluru city-summary amount | INR 523,643.40 |
| Mumbai city-summary amount | INR 479,058.50 |
| Customers without completed orders | 25 |
| Customers above average completed-order spending | 550 |

The SQL validation script reported all seven queries passing.

`RANK()` gives equal ranks to tied values, whereas `ROW_NUMBER()` assigns a distinct sequence number to each row.

### 5.6 Order Statistics and Exploratory Analysis

Recorded completed-order statistics:

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

Using the Interquartile Range (IQR) method:

- IQR = Q3 − Q1.
- Upper fence = Q3 + 1.5 × IQR.
- Completed orders above the upper fence are flagged as upper outliers.

The mean being higher than the median and the positive skewness indicate that a relatively small number of large orders influence the average.

The recorded strongest absolute correlation with order amount is the relationship with discount percentage, approximately −0.095. This is a weak negative association and does not prove that discounts cause lower spending.

### Main Part A Outputs

- `data/raw/`
- `data/clean/`
- `data/policies/`
- `data/instamart.db`
- `data/clean/customer_features.csv`
- `sql/queries.sql`

## Part B — Customer Segments & Lapse Prediction

### 6.1 RFM Feature Construction

Customer segmentation uses RFM-style features:

- **Recency:** Whole number of days since the customer's last completed order.
- **Frequency:** Number of completed orders.
- **Monetary:** Total completed-order spending.

The fixed RFM snapshot date is 1 October 2025.

Customers without completed orders have no last completed-order date, so their recency is missing.

Recorded cohort:

| Measure | Result |
|---|---:|
| Total customers | 1,200 |
| Customers with completed orders | 1,175 |
| Customers without completed orders | 25 |
| Median recency | 19 days |
| Minimum recency | 0 days |
| Maximum recency | 180 days |
| Mean frequency | 9.46 |
| Mean monetary value | INR 2,552.06 |
| Maximum frequency | 33 |
| Maximum monetary value | INR 14,766.55 |

### 6.2 Customer Segmentation

The RFM features are scaled before K-means clustering. Customers without completed orders are assigned a separate `No Completed Orders` segment.

Recorded segment distribution:

| Segment | Customers |
|---|---:|
| Regular Customers | 436 |
| High Value | 419 |
| Low Spend | 183 |
| Top Value | 137 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

Segment names describe the fitted groups and are not inherent properties of customers. The premium flag identifies the selected highest-mean-monetary segment for downstream support prioritisation.

### 6.3 K-Means Evaluation

K-means was evaluated for cluster counts from 2 to 8 using `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette score |
|---:|---:|---:|
| 2 | 1820.7 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.0 | 0.525 |
| 5 | 617.5 | 0.478 |
| 6 | 503.2 | 0.483 |
| 7 | 408.5 | 0.416 |
| 8 | 366.2 | 0.412 |

Among the evaluated values, K=4 has the highest silhouette score of 0.525. Inertia generally decreases as K increases, so inertia alone should not be used to select the number of clusters.

### 6.4 Principal Component Analysis

The first two principal components explain:

| Component | Explained variance |
|---|---:|
| PC1 | 63.4% |
| PC2 | 21.8% |
| **Combined** | **85.2%** |

These ratios indicate how much of the scaled feature variance is represented by the first two components.

### 6.5 Lapse Prediction Task

The lapse-prediction configuration uses:

- Cutoff date: 1 September 2025
- Label window: September 2025
- Random state: 42
- Eligible cohort: 1,156 customers

The label identifies whether a customer has no completed order during the September label window.

| Lapse label | Customers |
|---|---:|
| Not Lapsed | 713 |
| Lapsed | 443 |
| **Total** | **1,156** |

The lapsed proportion is approximately 38.32%. The data is split into 924 training rows and 232 testing rows using a stratified split.

### 6.6 Feature Construction and Leakage Prevention

The model features include historical recency, frequency, monetary value, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender, and customer tenure.

The implementation follows these safeguards:

- No orders dated 1 September 2025 or later are used to construct the lapse-model features.
- The test split is kept separate from model fitting and preprocessing.
- Numerical features are scaled.
- Categorical variables are one-hot encoded with a dropped reference category.
- Preprocessing is fitted on training data only.
- Segment labels are not used as lapse-model features.
- Oversampling, when evaluated, is applied only to the training data.

The training scaler's mean frequency was 8.115, compared with 8.029 for the full cohort. These means can differ because the scaler is fitted only on the training split.

### 6.7 Baseline and Model Comparison

An always-not-lapsed baseline achieves 61.6% accuracy but an F1 score of zero for the lapsed class. This demonstrates why accuracy alone is insufficient for an imbalanced classification task.

Logistic Regression and Random Forest were evaluated using cross-validation with F1 scoring.

| Model | Test accuracy | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.728 | 0.64 | 0.65 | 0.65 |
| Random Forest | 0.737 | 0.66 | 0.65 | 0.66 |

The recorded Random Forest model has the higher cross-validation F1 score.

### 6.8 Hyperparameter Tuning

Logistic Regression was tuned over C values of 0.01, 0.1, 1, 10, and 100, using five-fold cross-validation and F1 scoring.

Random Forest was tuned over:

- `max_depth`: 3, 5, 8
- `min_samples_leaf`: 5, 10, 20
- `n_estimators`: 100
- `random_state`: 42
- Five-fold cross-validation with F1 scoring

Recorded results:

| Model | Best parameters | Best CV F1 |
|---|---|---:|
| Logistic Regression | `C=100` | 0.651 |
| Random Forest | `max_depth=8`, `min_samples_leaf=10` | 0.681 |

### 6.9 Confusion Matrices

The confusion matrices use rows for actual labels and columns for predicted labels, ordered as not lapsed and lapsed.

**Logistic Regression**

```text
[[111, 32],
 [ 31, 58]]
```

**Random Forest**

```text
[[113, 30],
 [ 31, 58]]
```

For the recorded Random Forest model:

- 113 customers were correctly predicted as not lapsed.
- 30 customers were predicted as lapsed but were not lapsed.
- 31 lapsed customers were incorrectly predicted as not lapsed.
- 58 lapsed customers were correctly identified.

False negatives can cause retention opportunities to be missed. False positives can result in unnecessary retention outreach.

### 6.10 Oversampling Experiment

The training set was oversampled to balance the two classes. The test set remained unchanged.

| Evaluation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Original training distribution | 0.644 | 0.652 | 0.648 |
| Oversampled training distribution | 0.589 | 0.854 | 0.697 |

Oversampling improved recall in this experiment but reduced precision. This trade-off is useful when missing a potentially lapsed customer is more costly than contacting a customer who would have remained active.

### 6.11 Feature Importance

The recorded top five Random Forest features are:

| Feature | Importance |
|---|---:|
| Frequency | 0.2796 |
| Monetary value | 0.2712 |
| Recency | 0.1152 |
| Average delivery minutes | 0.0612 |
| Cancellation rate | 0.0525 |

Feature importance describes the fitted model's relative reliance on its input features. It does not establish causation.

### 6.12 Logistic Regression Coefficients

| Feature | Coefficient |
|---|---:|
| Recency | +0.0392 |
| Frequency | −1.6973 |
| Monetary value | +0.2834 |

The coefficient signs describe associations with predicted lapse odds while holding the other model features constant. They should not be interpreted as causal effects.

### 6.13 Retention List

`outputs/retention_list.csv` contains ten eligible premium customers from the test set, ranked by predicted lapse probability. The recorded run identified one customer in this list who was actually lapsed.

**Leakage limitation:** The premium flag is derived from whole-window segmentation, which can include September information used to define the lapse label. Therefore, the retention list is illustrative and is not a leakage-free evaluation, even though the lapse model's own features use pre-cutoff information.

The fitted model is saved to `outputs/lapse_model.joblib`. Reload verification confirmed that predictions matched exactly after loading the saved model.

### Main Part B Outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage with Guardrails

### 7.1 Input Validation

The pipeline processes 164 synthetic support tickets.

Before classification:

- Empty or whitespace-only text is assigned `skipped_empty`.
- Text exceeding the configured 400-token input limit is assigned `flagged_long`.
- Guardrail-rejected tickets are not sent to the model for classification.

The dataset contains four empty tickets and four long tickets.

### 7.2 PII Redaction

Personal information is redacted before model prompts are constructed.

| Information | Placeholder | Substitutions |
|---|---|---:|
| Email address | `[EMAIL]` | 55 |
| Phone number | `[PHONE]` | 91 |
| Order ID | `[ORDER_ID]` | 132 |
| **Total** | | **278** |

The redaction order is:

1. Email addresses
2. Phone numbers
3. Order IDs

The recorded audit found zero remaining email, phone, or order-ID pattern matches after redaction.

The mock model receives the redacted ticket text as its user content. Sensitive original values are not intentionally appended to the model message.

### 7.3 Zero-Shot and Few-Shot Classification

The pipeline compares two prompting strategies:

- **Zero-shot:** The model receives classification instructions without labelled examples.
- **Few-shot:** The prompt includes labelled examples to guide the model's category selection.

Both approaches classify tickets into the allowed categories:

- `billing`
- `delivery`
- `fraud`
- `technical`

The expected response format is JSON.

### 7.4 Retry and Error Handling

The pipeline includes retry handling for transient model errors.

The retry configuration uses a maximum of three retries and exponential backoff with jitter. The base delay is 0.05 seconds.

The configured wait is:

```text
0.05 × 2^n × uniform(0.9, 1.15)
```

The retry log records the ticket identifier, retry number, and wait duration. The recorded retry evaluation recovered 19 transient failures.

Retry attempts, model calls, and completed ticket classifications are different measures and should not be treated as interchangeable.

### 7.5 Output Validation and Fallback

The pipeline validates model responses before accepting a category.

The recorded comparison contains:

| Outcome | Count |
|---|---:|
| JSON classifications | 135 |
| Keyword-based classifications | 12 |
| Invalid model outputs | 9 |
| Eligible tickets | 156 |

Four empty tickets and four long tickets are stopped by input guardrails before classification.

When JSON validation fails, the implementation attempts its configured keyword fallback. If the output cannot be mapped unambiguously to one allowed category, it is treated as invalid rather than silently accepted as a valid classification.

### 7.6 Zero-Shot Confusion Matrix

The recorded zero-shot confusion matrix uses the label order billing, delivery, fraud, technical.

```text
[[38,  0,  0,  0],
 [ 0, 36,  0,  0],
 [11,  0, 23,  0],
 [10,  0,  0, 29]]
```

The matrix shows that 11 fraud tickets and 10 technical tickets were classified as billing in this evaluation.

These errors are important because fraud and technical issues may need different handling from routine billing requests.

### 7.7 Zero-Shot versus Few-Shot Evaluation

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets in the comparison | 147 | 147 |
| Accuracy | 85.7% | 100% |
| Total tokens | 11,584 | 22,660 |
| Tokens per classified ticket | 78.80 | 154.15 |

In the recorded synthetic evaluation, few-shot prompting improved accuracy but used almost twice as many tokens per classified ticket.

Few-shot prompting can be useful for higher-risk categories when classification accuracy is especially important. Zero-shot prompting is more token-efficient when cost and throughput are priorities. These results apply to the supplied synthetic tickets and mock LLM, not to real-world support traffic.

### 7.8 Token-Budget Guardrail

The project includes a separate demonstration using a token budget of 6,000.

| Status | Count |
|---|---:|
| `classified_json` | 70 |
| `classified_keyword` | 6 |
| `flagged_invalid_output` | 4 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |
| `budget_exhausted` | 76 |
| **Total** | **164** |

The budget demonstration used 5,951 tokens across 80 model calls.

These counts belong to the budget demonstration and should not be combined with the final few-shot run's counts.

### 7.9 Customer-Aware Routing

The final ticket-routing workflow uses ticket classification, guardrail outcomes, and customer segment information.

The intended routing rules are:

- Guardrail-rejected or otherwise unsafe cases â†’ `human_review`
- Fraud â†’ `security_team`, priority `HIGH`
- Premium customer â†’ `premium_desk`, priority `HIGH`
- Other eligible tickets â†’ `support_queue`, priority `NORMAL`

The guardrail checks take precedence over ordinary automated routing. Customers without a matching premium segment are not automatically treated as premium.

### Main Part C Outputs

The following output files are produced by the relevant scripts when those runs are executed:

- `outputs/tickets_redacted.csv`
- `outputs/ticket_triage_results.csv`
- `outputs/ticket_triage_summary.json`
- `outputs/triage_results.csv`
- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_confusion_zero_shot.csv`
- `outputs/triage_budget_demo.csv`

## Part D — Policy RAG Assistant & Refund Agent

### 8.1 Policy Documents and Chunking

The policy collection contains five synthetic documents covering refunds, delivery, payments, privacy, and escalation.

The overlap configuration uses 40-word chunks with 10 words of overlap. The recorded overlap configuration contains 15 chunks; the no-overlap configuration contains 13 chunks.

Recorded chunk counts by document in the overlap configuration:

| Policy document | Chunks |
|---|---:|
| Refund | 4 |
| Delivery | 3 |
| Escalation | 3 |
| Payment | 3 |
| Privacy | 2 |
| **Total** | **15** |

Chunk overlap helps retain context that might otherwise be split across adjacent chunks. It can also increase the number of indexed chunks.

### 8.2 TF-IDF Retrieval

The retrieval system uses TF-IDF and n-gram similarity to identify relevant policy text.

- Term frequency (TF) reflects how frequently a term appears in a text chunk.
- Inverse document frequency (IDF) gives more weight to terms that are relatively uncommon across the document collection.
- Similarity ranking selects the chunks most relevant to a question.
- The configured retrieval setting uses top-K results, with K=3 in the main evaluation.

Lexical retrieval can miss a condition when relevant words are split across chunks or when the question uses wording different from the policy text. Retrieving a relevant chunk does not by itself guarantee that a generated answer includes every required condition.

### 8.3 Retrieval Evaluation

The policy evaluation contains 14 questions: 12 answerable questions and two unanswerable questions.

Recorded summary:

| Evaluation | Result |
|---|---:|
| Policy documents loaded | 5 |
| Questions | 14 |
| Answerable questions | 12 |
| Answerable questions passed in the summary evaluation | 12 |
| Unanswerable questions | 2 |
| Unanswerable questions correctly handled with fallback | 2 |

The targeted late-delivery retrieval comparison reports:

| Strategy | Top-K | Completed-order condition | More than 45 minutes | Within 7 days |
|---|---:|---|---|---|
| Overlap 10 | 3 | Found | Found | Not found |
| Overlap 10 | 5 | Found | Found | Found |
| Overlap 10 | 8 | Found | Found | Found |
| No overlap | 3 | Found | Found | Found |
| No overlap | 5 | Found | Found | Found |
| No overlap | 8 | Found | Found | Found |

This diagnostic illustrates that the seven-day condition was absent from the overlap top-3 retrieval result but was present in the no-overlap top-3 result. Increasing K to 5 retrieved all three conditions in the overlap configuration.

The targeted condition-retrieval comparison is separate from the answerability and fallback counts in the evaluation summary. These results should not be treated as interchangeable.

### 8.4 Example Policy Rules

The synthetic policies and agent workflow demonstrate rules such as:

- A late-delivery refund requires a completed order delivered in more than 45 minutes.
- The late-delivery refund request must be made within seven days of delivery.
- A delivery time of 45 minutes or less does not satisfy the late-delivery threshold.
- The failed-payment refund workflow uses a 48-hour request window.
- Approved refunds are described as returning to the original payment method within five working days.
- Payment-card numbers are not stored by the workflow.

These are fictional project policies, not actual Instamart policy statements.

### 8.5 Refund Agent Workflow

The refund agent demonstrates a guarded, ReAct-style workflow:

1. Validate the request and order identifier.
2. Retrieve order details from SQLite.
3. Retrieve relevant policy information.
4. Evaluate eligibility using deterministic Python rules.
5. Validate the proposed refund amount and required approval.
6. Simulate an eligible refund, deny an ineligible request, or escalate the case.
7. Record the execution result and checkpoint information.

The LLM helps coordinate the workflow, while important eligibility and financial decisions are validated using Python rules.

### 8.6 Safety Controls

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

### 8.7 Agent Scenario Evaluation

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

The recorded scenario table shows that the scenarios matched their expected outcome labels, with S7 reaching the configured maximum-iteration fallback instead of a normal terminal decision.

### 8.8 Refund Examples

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

This demonstrates an eligible synthetic order. It does not initiate a real financial transaction.

**Ineligible order demonstration**

- Order ID: `ORD-110093`
- Delivery time: 19 minutes
- Result: rejected
- Refund amount: INR 0

This order does not satisfy the configured late-delivery threshold.

### 8.9 Token-Limit and Timeout Demonstrations

| Demonstration | Outcome | LLM calls | Tokens |
|---|---|---:|---:|
| Token-limit demonstration | `FALLBACK_TOKEN_LIMIT` | 2 | 196 |
| Timeout demonstration | `FALLBACK_TIMEOUT` | 0 | 0 |

These demonstrations show that the workflow can stop when a configured limit or timeout condition is reached.

### 8.10 Checkpoint and Crash-Resume Demonstration

The repository contains crash/resume state examples for the refund workflow in the `state/` directory.

The recorded recovery examples completed with a REFUND outcome, four steps, five LLM calls, and 948 tokens for each named run.

The saved state and execution logs provide evidence for the demonstrated recovery path. Aggregate counts alone do not prove exactly-once execution for every possible crash timing or external side effect.

Tools recorded in the crash/resume execution:
1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
These tools appear in the saved `state/refund_agent_CRASH_RESUME_CHECK_001.json` execution evidence. Refund actions are simulated; no real money is transferred.

### 8.11 ReAct and Plan-and-Execute Considerations

A ReAct-style agent interleaves tool calls and observations. This allows later decisions to use information returned by earlier tools, such as order status or policy eligibility.

A plan-and-execute design creates a plan first and then executes the planned steps. It can be useful for predictable workflows, but a fixed plan may need replanning when tool observations differ from expectations.

For this project, the ReAct-style workflow supports sequential order lookup, policy retrieval, deterministic eligibility checks, and guarded refund actions. The implementation also limits iterations and model calls to prevent unbounded execution.

A separate measured plan-and-execute benchmark, with independently recorded call and token counts, should only be reported if that implementation and its results are available.

### 8.12 Refund-Agent Limitations

All refund actions are simulated. No payment gateway, live customer account, or real refund API is connected.

The project demonstrates validation, approval checks, tool restrictions, execution limits, and checkpointing in an educational environment. A production deployment would require stronger transactional guarantees, persistent audit records, authentication, authorisation, and duplicate-action protection.

### Main Part D Outputs

The following output files are produced by the relevant scripts when those runs are executed:

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_summary.json`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_checkpoint.json`
- `outputs/refund_agent_scenarios.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

Crash/resume state examples are stored under `state/`, not `outputs/`.

### Validation and Testing

The project includes validation scripts for data quality, database integrity, SQL query results, order statistics, and automated tests.

Run the validation commands from the repository root:

```powershell
python check_clean_data.py
python src/check_database.py
python check_sql_queries.py
python check_order_stats.py
python -m pytest -v
```

The recorded validation results include:

- Cleaned data checks passed.
- Database validation checks passed.
- All seven SQL query checks passed.
- The recorded automated test run completed with 14 tests passed.

Results can vary if generated data, project code, configuration, or dependencies change.

## Limitations and Next Steps

- All datasets are synthetic and do not represent real Instamart customer data.
- The supplied LLM is mocked; results may differ substantially from those of a live language model.
- Model metrics depend on synthetic labels, feature construction, preprocessing, and the evaluation split.
- The retention list is illustrative and is not a leakage-free evaluation because the premium flag is derived from whole-window segmentation.
- Feature importance and coefficients describe model behaviour and do not establish causation.
- Ticket-triage accuracy and token counts apply to the evaluated mock-LLM runs, not to real-world support traffic.
- Policy retrieval uses lexical similarity and may retrieve incomplete or irrelevant context for ambiguous questions.
- The RAG evaluation set is small and does not establish broad policy-answering reliability.
- The targeted retrieval diagnostic and overall answerability/fallback evaluation measure different aspects of the RAG workflow.
- Refund processing is simulated; no real payment or refund transaction occurs.
- Checkpoint and crash/resume demonstrations do not guarantee exactly-once execution under every failure condition.
- Automated tests cover selected rules and scenarios, not every possible input or production failure.
- A separate measured plan-and-execute comparison should not be claimed unless it has actually been run and recorded.

### Future Improvements

- Expand data-quality checks and automated tests.
- Evaluate segmentation stability across different samples and random seeds.
- Validate lapse prediction on independent data and monitor precision-recall trade-offs.
- Expand ticket-triage evaluation with additional labelled examples and difficult edge cases.
- Improve token-budget monitoring and fallback reporting.
- Expand RAG evaluation with answerable, unanswerable, ambiguous, and conflicting-policy questions.
- Compare ReAct and plan-and-execute using the same scenarios and report measured call and token counts.
- Strengthen refund-agent recovery, audit logging, and duplicate-action protection before any real integration.
- Add authentication, authorisation, and human approval workflows before connecting to a production support system.

### Conclusion

The Instamart Customer Intelligence & Support Copilot brings together data engineering, SQL analysis, machine learning, text retrieval, and a guarded agent workflow in one reproducible educational project.

The repository demonstrates customer segmentation, lapse prediction, privacy-aware ticket triage, policy retrieval, and simulated refund decisions. Its results and safeguards are documented for learning and evaluation, not for production customer support or real financial transactions.



