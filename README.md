# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot using synthetic Instamart-style e-commerce data. It combines data engineering, SQL analytics, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a simulated refund agent.

The project demonstrates how analytics and machine learning can help an e-commerce business understand customer behaviour, identify customers who may lapse, prioritize support tickets, and handle refund requests using policy-based guardrails.

**Important:** This project uses synthetic data. The refund agent simulates refund decisions and does not transfer real money.

## Reproduce

Run all commands from the repository root using Python 3.11 or later and Windows PowerShell.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Generate, clean, and validate the data

```powershell
python generate_data.py
python src/clean_data.py
python src/build_database.py
python src/check_database.py
```

### 5. Run data inspection and SQL analysis

```powershell
python src/inspect_data.py
python src/sql_analysis.py
```

The SQL queries are documented in `sql/queries.sql`.

### 6. Run customer feature engineering, segmentation, and lapse prediction

```powershell
python src/customer_features.py
python src/customer_segmentation.py
python src/lapse_prediction.py
```

### 7. Run support ticket triage and policy retrieval

```powershell
python src/ticket_triage.py
python src/policy_rag.py
```

### 8. Run the refund-agent demonstration

```powershell
python src/refund_agent.py
```

### 9. Run automated tests

```powershell
python -m pytest -q
```

The latest recorded test run completed with **12 passed**.

**Reproduction note:** Generated outputs and evaluation metrics can change if the data or configuration changes. Re-run the relevant checks after regenerating the data. The provided `generate_data.py` and `mock_llm.py` files are project dependencies and should remain unmodified.

## Part A — Data Engineering, SQL and EDA

### Data Foundation

The project generates synthetic customer, product, order, support-ticket, agent-scenario, and policy-evaluation data. Python and pandas are used for data preparation, and SQLite is used for structured storage and SQL analysis.

### Raw Data Audit

The recorded raw-data audit identified the following issues:

| Audit item | Finding |
|---|---:|
| Raw order rows | 13,084 |
| Duplicate order records removed | 119 |
| Pending orders in raw data | 656 |
| Distinct order IDs in cleaned orders | 12,309 |
| Nonnumeric raw order amounts | 120 |
| Negative raw delivery times | 106 |
| Blank raw delivery times | 2,541 |
| Customers with blank ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Nonblank raw gender spellings | 10 |
| Customers with blank genders | 30 |

These are raw-data audit findings. The cleaning workflow prepares the data for downstream analysis by handling duplicate records, invalid values, and inconsistent data. Missing delivery times are expected for some cancelled orders because those orders may not have been delivered.

### Cleaned Data Validation

The latest recorded cleaned-data counts were:

| Dataset | Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |
| Support tickets | 164 |

The cleaned order dataset contained 11,110 completed orders and 1,199 cancelled orders. Pending orders and duplicate order records were excluded from the cleaned orders.

The recorded completed-order spending total was **₹2,998,668.80**.

Database validation checked record counts, unique identifiers, order statuses, missing customer identifiers, and invalid order amounts.

### SQL Analysis

The project includes seven SQL analyses covering customer spending, order counts, product performance, customers without completed orders, aggregations, ranking differences, and daily cumulative spending.

| Query | Recorded result |
|---|---|
| Q1 — Top-spending customers | 30 rows |
| Q2 — Completed-order revenue analysis | 30 rows; total completed revenue ₹2,998,668.80 |
| Q3 — Product analysis | 11 products returned |
| Q4 — Customers without completed orders | 25 customers |
| Q5 — Aggregated analysis | 550 rows |
| Q6 — Ranking comparison | 8 rows where `RANK()` differs from `ROW_NUMBER()` |
| Q7 — Daily cumulative revenue | 183 days; final cumulative total ₹2,998,668.80 |

The five highest recorded customer spending totals were:

| Customer ID | Completed spending (₹) |
|---|---:|
| C0693 | 14,766.55 |
| C0291 | 13,742.60 |
| C0488 | 13,187.50 |
| C0082 | 12,856.20 |
| C0743 | 12,730.05 |

The queries demonstrate grouping, aggregation, ranking, filtering, and window functions. The cumulative revenue analysis also demonstrates how daily totals can be used to study revenue over time.

### Exploratory Data Analysis

Recorded customer counts by city were:

| City | Customers |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |

Recorded customer gender counts were 610 female, 560 male, and 30 unknown. Cleaned customer ages ranged from 18 to 60.

These checks describe the customer population and identify data-quality issues before segmentation and predictive modelling.

### Part A Outputs

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

## Part B — Customer Segmentation and Lapse Prediction

### Customer Feature Engineering

Customer features summarize purchasing behaviour, including order frequency, completed spending, and recency. The segmentation workflow uses the cleaned order data and a snapshot date of **2025-10-01**.

Of 1,200 customers, 1,175 had completed-order history and were eligible for RFM segmentation. The remaining 25 customers had no completed orders.

### RFM Summary

| Statistic | Recency (days) | Frequency | Monetary (₹) |
|---|---:|---:|---:|
| Mean | 33.93 | 9.46 | 2,552.06 |
| Median | 19.45 | 5 | 2,286.50 |
| Minimum | 0.05 | 1 | 12.00 |
| Maximum | 180.64 | 33 | 14,766.55 |

Recency measures how long it has been since the most recent completed order. Frequency measures completed-order activity, while monetary value summarizes completed spending.

### K-Means Evaluation

K-Means clustering was evaluated with silhouette scores for candidate values of `k`.

| Number of clusters (k) | Silhouette score |
|---:|---:|
| 2 | 0.465 |
| 3 | 0.445 |
| 4 | 0.525 |
| 5 | 0.478 |
| 6 | 0.483 |
| 7 | 0.416 |
| 8 | 0.412 |

Among the evaluated candidates, **k = 4** achieved the highest silhouette score of **0.525**. A higher silhouette score indicates better separation and cohesion under this metric for the tested data.

### PCA

The recorded PCA explained-variance ratios were:

| Component | Explained variance |
|---|---:|
| PC1 | 0.634 |
| PC2 | 0.218 |
| Combined | 0.852 |

The first two components explained approximately **85.2%** of the variation in the transformed feature data.

### Customer Segments

The recorded segmentation run produced:

| Segment | Customers |
|---|---:|
| Top Value | 137 |
| High Value | 419 |
| Regular Customers | 436 |
| Low Spend | 183 |
| No Completed Orders | 25 |
| **Total** | **1,200** |

The premium-customer flag identifies customers selected by the configured segmentation rules for additional support prioritisation. Segment names are project-specific labels rather than universal customer categories.

### Lapse Prediction

The lapse-prediction task uses a cutoff date of **2025-09-01** and a September label window from **2025-09-01 to 2025-09-30**.

The recorded cohort contained 1,156 customers:

| Label | Customers |
|---|---:|
| Not Lapsed | 713 |
| Lapsed | 443 |
| **Total** | **1,156** |

The train/test split contained 924 training rows and 232 test rows. No missing feature values were reported before preprocessing.

### Model Comparison

The baseline model and candidate models were evaluated using cross-validation and held-out test data.

| Model | Recorded result |
|---|---|
| Baseline | Accuracy 0.616; lapsed-class F1 0.000 |
| Tuned Logistic Regression | Best recorded C = 100; cross-validation F1 = 0.651; test accuracy = 0.728; lapsed-class F1 ≈ 0.650 |
| Tuned Random Forest | `max_depth=8`, `min_samples_leaf=10`; cross-validation F1 = 0.681; test accuracy = 0.737; lapsed-class F1 = 0.655 |
| Oversampled Logistic Regression | Lapsed-class precision = 0.589; recall = 0.854; F1 = 0.697 |

The Random Forest was selected based on the recorded cross-validation F1 comparison. Oversampling improved the Logistic Regression model's recorded recall, but also changed the precision-recall balance.

### Selected Random Forest Evaluation

| Metric | Score |
|---|---:|
| Accuracy | 0.737 |
| Lapsed-class precision | 0.659 |
| Lapsed-class recall | 0.652 |
| Lapsed-class F1 | 0.655 |

Recorded Random Forest confusion matrix, with rows representing actual labels and columns representing predicted labels in the order Not Lapsed, Lapsed:

| Actual label | Predicted Not Lapsed | Predicted Lapsed |
|---|---:|---:|
| Not Lapsed | 113 | 30 |
| Lapsed | 31 | 58 |

The model's feature-importance ranking included frequency (0.2796), monetary value (0.2712), recency (0.1152), average delivery minutes (0.0612), and cancellation rate (0.0525).

The saved model was reloaded and its predictions matched the recorded predictions in the verification run.

### Retention List and Limitations

The project generates a retention list to help prioritize customers who may be at risk of lapsing. In the recorded run, the list contained 10 customers, of whom one was actually lapsed in the evaluation data.

**Important limitation:** The premium-customer flag is derived from whole-window RFM segmentation. The retention list is illustrative and is not a fully leakage-free production targeting evaluation. A stricter temporal evaluation should construct every feature and prioritisation label only from information available at the prediction cutoff.

### Part B Outputs

- `outputs/customer_segments.csv`
- `outputs/kmeans_scores.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C — Support Ticket Triage with Guardrails

### Ticket Classification

The triage workflow classifies support tickets, validates model outputs, applies keyword-based fallback handling, and tracks invalid outputs, empty tickets, long inputs, model calls, and token consumption.

The project evaluates zero-shot and few-shot approaches using synthetic support tickets.

### PII Redaction

The recorded redaction audit reported:

| PII type | Detected before redaction | Detected after redaction |
|---|---:|---:|
| Email addresses | 55 | 0 |
| Phone numbers | 91 | 0 |
| Order IDs | 132 | 0 |

The audit detected 278 occurrences across these categories, and the recorded post-redaction counts were zero. This validates the implemented detection patterns on the tested data, not every possible form of sensitive information.

### Zero-Shot Evaluation

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| Model calls | 156 |
| Tokens used | 11,584 |
| JSON classifications | 135 |
| Keyword fallback classifications | 12 |
| Valid classifications | 147 |
| Invalid outputs | 9 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Budget exhausted | 0 |
| Accuracy on valid classifications | 0.8571 |
| Tokens per valid classification | 78.80 |

### Few-Shot Evaluation

| Metric | Result |
|---|---:|
| Tickets processed | 164 |
| Model calls | 156 |
| Tokens used | 22,660 |
| JSON classifications | 135 |
| Keyword fallback classifications | 12 |
| Valid classifications | 147 |
| Invalid outputs | 9 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Budget exhausted | 0 |
| Accuracy on valid classifications | 1.0000 |
| Tokens per valid classification | 154.15 |

The recorded few-shot evaluation achieved 100% accuracy on the 147 valid classifications in this synthetic test. It used more tokens per valid classification than the zero-shot workflow. This result should not be interpreted as a guarantee of real-world performance.

### Token Budget Demonstration

A separate run tested a token budget of 6,000.

| Metric | Result |
|---|---:|
| Token budget | 6,000 |
| Tokens used | 5,951 |
| Model calls | 80 |
| JSON classifications | 70 |
| Keyword fallback classifications | 6 |
| Valid classifications | 76 |
| Invalid outputs | 4 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Budget-exhaustion count | 76 |
| Accuracy on valid classifications | 0.8421 |
| Tokens per valid classification | 78.30 |

The budget-limited run demonstrates the trade-off between available processing budget and the number of valid classifications produced.

### Routing and Guardrails

The triage workflow applies configured classification and priority rules to route support tickets. It also tracks invalid model responses, empty inputs, long messages, keyword fallbacks, and token-budget exhaustion.

PII redaction is performed before ticket processing. Redaction and output validation are safeguards, but neither guarantees perfect detection or classification.

### Part C Outputs

- `outputs/tickets_redacted.csv`
- `outputs/triage_results.csv`
- `outputs/ticket_triage_summary.json`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_confusion_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_zero_shot.csv`

## Part D — Policy RAG Assistant and Refund Agent

### Policy Retrieval

The policy assistant loads policy documents, divides them into word chunks, and retrieves relevant passages using TF-IDF similarity. It uses configured policy logic for recognized questions and returns a fallback response when the available policy information does not support an answer.

The recorded evaluation used five policy documents and 15 indexed chunks.

| Setting | Result |
|---|---:|
| Policy documents | 5 |
| Indexed chunks | 15 |
| Chunk size | 40 words |
| Chunk overlap | 10 words |
| Default retrieved chunks | 3 |
| Evaluation questions | 14 |
| Answerable questions passed | 12/12 |
| Fallback questions passed | 2/2 |
| Overall evaluation | 14/14 |

The policies include late-delivery conditions, failed-payment handling, and escalation guidance.

### Retrieval Comparison

The recorded comparison showed that retrieval quality depends on chunking and the number of retrieved chunks.

| Strategy | Top-K | All required conditions retrieved? |
|---|---:|---|
| 10-word overlap | 3 | No — the 7-day request condition was missed |
| 10-word overlap | 5 | Yes |
| 10-word overlap | 8 | Yes |
| No overlap | 3 | Yes |
| No overlap | 5 | Yes |
| No overlap | 8 | Yes |

In this experiment, increasing retrieval from top-3 to top-5 resolved the missing 7-day condition for the overlap strategy. The no-overlap strategy retrieved all three conditions even with top-3 retrieval. These results apply to the recorded test data and do not establish that one chunking strategy will always perform better than another.

### Refund Agent Guardrails

The refund agent follows a controlled workflow that uses order lookup, policy retrieval, refund calculation, and simulated refund or escalation actions.

Configured rules include:

- Late-delivery eligibility requires a completed order delivered more than 45 minutes after the relevant order timing.
- The refund request must be within the configured 7-day request window.
- The automatic refund limit is ₹1,000.
- Ineligible requests can be denied according to the policy checks.
- Requests above the automatic refund limit may be escalated.
- Workflow checkpoints support recovery after interruption.

The agent does not connect to a payment gateway. All refund outcomes are simulated.

### Successful Refund Demonstration

| Field | Recorded result |
|---|---|
| Order ID | ORD-110603 |
| Order amount | ₹58.00 |
| Delivery time | 70 minutes |
| Request age | 3 days |
| Agent status | `simulated_refund_issued` |
| Workflow steps | 4 |
| LLM calls | 5 |
| Final answer | `REFUND: INR 58.0 issued.` |

A separate check rejected the same demonstration order when the request age was 8 days, exceeding the configured 7-day window.

### Seven-Scenario Evaluation

All seven recorded scenarios matched their expected outcome and expected refund amount. The model-call and token counts below are matched to each scenario's run ID in `outputs/agent_runs.jsonl`.

| Scenario | Expected outcome | Actual outcome | Expected refund (₹) | Actual refund (₹) | Model calls | Tokens used | Match |
|---|---|---|---:|---:|---:|---:|---|
| S1 | REFUND | REFUND | 128.25 | 128.25 | 5 | 953 | Yes |
| S2 | REFUND | REFUND | 135.00 | 135.00 | 5 | 953 | Yes |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 1,999.00 | 5 | 943 | Yes |
| S4 | DENIED | DENIED | 0.00 | 0.00 | 4 | 661 | Yes |
| S5 | DENIED | DENIED | 0.00 | 0.00 | 3 | 406 | Yes |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 0.00 | 2 | 154 | Yes |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 0.00 | 6 | 1,296 | Yes |

The evaluation covers successful refunds, escalation for a refund above the automatic limit, denied requests, a missing order, and a maximum-iteration fallback. All refund amounts are simulated; no real money was transferred.

### Token and Timeout Guardrails

The recorded token-limit test returned `FALLBACK_TOKEN_LIMIT` after two LLM calls. The agent stopped safely before the next LLM call because the token budget was exhausted.

The recorded zero-second timeout test returned `FALLBACK_TIMEOUT` with zero LLM calls. The agent stopped safely rather than continuing execution beyond the configured time limit.

### Crash and Resume

A simulated crash-and-resume test used order `ORD-100048` and verified that the agent continued from its saved checkpoint.

The tool sequence across the crash and resume was:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The final status was `REFUND`, the final answer was `REFUND: INR 128.25 issued.`, and the refund amount matched the expected ₹128.25. The initial `get_order` call was not repeated after resuming from the checkpoint. The workflow demonstrated recovery without repeating the already completed tool execution.

### ReAct Versus Plan-and-Execute

The implemented agent uses a ReAct-style workflow: the model selects a tool, observes its result, and then decides what to do next. In the recorded scenarios, S1 used 5 model calls and 953 tokens, S5 used 3 calls and 406 tokens, and S6 used 2 calls and 154 tokens.

A plan-and-execute design could ask the model to generate a complete tool plan once, execute the steps, and ask for a revised plan only when an unexpected result occurs. This could reduce model calls for straightforward requests, although the actual token savings would need to be measured in a separate implementation.

ReAct can respond step by step to observations such as a missing order, a cancelled order, or a failed eligibility check. A fixed plan may require replanning when these observations differ from expectations. For this refund workflow, ReAct is useful because each tool result can inform the next action, while explicit policy checks and limits constrain the workflow. Neither design replaces deterministic validation, checkpointing, or safe handling of unexpected results.

### Part D Outputs

- `outputs/policy_rag_results.csv`
- `outputs/policy_rag_overlap_comparison.csv`
- `outputs/policy_rag_summary.json`
- `outputs/refund_agent_demo.json`
- `outputs/refund_agent_scenarios.csv`
- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- Checkpoint files under `state/`

### Safety Boundary

Refunds are simulated only. No real money is transferred. The agent's results demonstrate workflow behaviour and guardrails, not production payment processing.

## Limitations and Next Steps

- The project uses synthetic data and does not represent actual Instamart customer records.
- SQL totals and model metrics depend on the generated data and evaluation configuration.
- Customer segments depend on the selected features and clustering rules.
- Lapse predictions are estimates and should not be treated as certain outcomes.
- The retention list is illustrative and does not establish the real-world impact of retention campaigns.
- The premium-customer flag is based on whole-window RFM information; a stricter temporal evaluation should avoid leakage from information unavailable at prediction time.
- Ticket classification results may not generalize to real customer support messages.
- PII redaction relies on implemented detection patterns and may not identify every sensitive value.
- Policy retrieval depends on the available documents, chunking strategy, and retrieval settings.
- The refund agent simulates decisions and does not issue actual refunds.
- The automated tests focus on refund-agent behaviour and do not comprehensively test every project component.
- Production deployment would require stronger security, monitoring, policy validation, model evaluation, and integration testing.

Possible next steps include broader automated testing, temporal validation of retention predictions, additional retrieval evaluation, stronger privacy checks, and monitoring model performance on new data.

## Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates an end-to-end workflow combining data preparation, SQL analytics, customer segmentation, lapse prediction, support ticket triage, policy retrieval, and a guarded refund simulation.

The project highlights how analytics and machine learning can support customer intelligence and customer service while emphasizing validation, evaluation, reproducibility, and safety boundaries.