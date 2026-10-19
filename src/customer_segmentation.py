
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"

INPUT_PATH = CLEAN_DIR / "customer_features.csv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "customer_segments.csv"
SCORES_PATH = OUTPUT_DIR / "kmeans_scores.csv"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")
RANDOM_STATE = 42


# ==================================================
# 2. LOAD AND VALIDATE CUSTOMER FEATURES
# ==================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing {INPUT_PATH}. Run src/customer_features.py first."
    )

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "customer_id",
    "completed_orders",
    "completed_spend",
    "last_order_ts",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in customer_features.csv: {missing_columns}"
    )

if df["customer_id"].isna().any():
    raise ValueError("Missing customer_id values found.")

if df["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer_id values found.")

df["last_order_ts"] = pd.to_datetime(
    df["last_order_ts"], errors="coerce"
)

df["completed_orders"] = pd.to_numeric(
    df["completed_orders"], errors="coerce"
).fillna(0)

df["completed_spend"] = pd.to_numeric(
    df["completed_spend"], errors="coerce"
).fillna(0)

if (df["completed_orders"] < 0).any():
    raise ValueError("Negative completed_orders values found.")

if (df["completed_spend"] < 0).any():
    raise ValueError("Negative completed_spend values found.")


# ==================================================
# 3. SEPARATE ELIGIBLE AND NO-ORDER CUSTOMERS
# ==================================================

# Only customers with completed-order history enter K-Means.
active = df[df["completed_orders"] > 0].copy()

active["recency"] = (
    SNAPSHOT_DATE - active["last_order_ts"]
).dt.total_seconds() / 86400

active["frequency"] = active["completed_orders"]
active["monetary"] = active["completed_spend"]

feature_columns = ["recency", "frequency", "monetary"]

# Customers with invalid RFM data cannot enter K-Means.
valid_mask = (
    active[feature_columns].notna().all(axis=1)
    & (active["recency"] >= 0)
)

eligible = active.loc[valid_mask].copy()
invalid_rfm = active.loc[~valid_mask].copy()

no_orders = df[df["completed_orders"] <= 0].copy()

if len(eligible) < 8:
    raise ValueError(
        "At least 8 eligible customers are needed to evaluate K=2 to K=8."
    )

X = eligible[feature_columns].copy()

print("=" * 55)
print("CUSTOMER SEGMENTATION")
print("=" * 55)

print("RFM snapshot date:", SNAPSHOT_DATE.date())
print("Total source customers:", len(df))
print("Customers with completed orders:", len(active))
print("Eligible customers for clustering:", len(eligible))
print("Customers with no completed orders:", len(no_orders))
print("Customers with invalid RFM data:", len(invalid_rfm))

print("\nRFM summary:")
print(
    eligible[feature_columns]
    .agg(["mean", "median", "min", "max"])
    .round(2)
    .to_string()
)


# ==================================================
# 4. SCALE RFM FEATURES
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================
# 5. EVALUATE K-MEANS FROM K=2 TO K=8
# ==================================================

score_rows = []

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    score_rows.append({
        "k": k,
        "inertia": round(float(model.inertia_), 1),
        "silhouette": round(
            float(silhouette_score(X_scaled, labels)), 3
        ),
    })

scores = pd.DataFrame(score_rows)
scores.to_csv(SCORES_PATH, index=False)

print("\nK-Means evaluation:")
print(scores.to_string(index=False))


# ==================================================
# 6. FINAL K-MEANS MODEL: K=4
# ==================================================

final_model = KMeans(
    n_clusters=4,
    random_state=RANDOM_STATE,
    n_init=10,
)

eligible["segment_id"] = final_model.fit_predict(X_scaled)

# Name clusters according to ascending average monetary value.
spending_order = (
    eligible.groupby("segment_id")["monetary"]
    .mean()
    .sort_values()
)

segment_names = [
    "Low Spend",
    "Regular Customers",
    "High Value",
    "Top Value",
]

name_mapping = {
    cluster_id: segment_names[index]
    for index, cluster_id in enumerate(spending_order.index)
}

eligible["segment_name"] = eligible["segment_id"].map(name_mapping)

# Premium customers belong to the highest-spending cluster.
eligible["is_premium"] = eligible["segment_name"].eq("Top Value")


# ==================================================
# 7. PCA EXPLAINED VARIANCE
# ==================================================

pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA explained variance ratio:")
print("PC1:", round(float(pca.explained_variance_ratio_[0]), 3))
print("PC2:", round(float(pca.explained_variance_ratio_[1]), 3))
print(
    "Combined:",
    round(float(pca.explained_variance_ratio_.sum()), 3),
)


# ==================================================
# 8. BUILD OUTPUT FOR ALL CUSTOMERS
# ==================================================

output_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]

# Customers assigned to K-Means clusters.
clustered_result = eligible[output_columns].copy()

# Customers without completed orders still appear in the output.
no_orders_result = no_orders[["customer_id"]].copy()
no_orders_result["recency"] = float("nan")
no_orders_result["frequency"] = 0
no_orders_result["monetary"] = 0
no_orders_result["segment_id"] = pd.NA
no_orders_result["segment_name"] = "No Completed Orders"
no_orders_result["is_premium"] = False

# Customers with completed orders but invalid RFM features.
invalid_result = invalid_rfm[["customer_id"]].copy()
invalid_result["recency"] = invalid_rfm["recency"]
invalid_result["frequency"] = invalid_rfm["frequency"]
invalid_result["monetary"] = invalid_rfm["monetary"]
invalid_result["segment_id"] = pd.NA
invalid_result["segment_name"] = "Insufficient RFM Data"
invalid_result["is_premium"] = False

# Combine all groups into one customer-level output.
result = pd.concat(
    [
        clustered_result,
        no_orders_result[output_columns],
        invalid_result[output_columns],
    ],
    ignore_index=True,
)

result["segment_id"] = result["segment_id"].astype("Int64")
result["is_premium"] = result["is_premium"].fillna(False).astype(bool)

result = result.sort_values(
    "customer_id"
).reset_index(drop=True)


# ==================================================
# 9. VALIDATE AND SAVE OUTPUT
# ==================================================

if len(result) != len(df):
    raise AssertionError(
        f"Output has {len(result)} rows; expected {len(df)} customers."
    )

if result["customer_id"].duplicated().any():
    raise AssertionError("Duplicate customer IDs found in output.")

if result["customer_id"].nunique() != df["customer_id"].nunique():
    raise AssertionError("Some customers are missing from the output.")

result.to_csv(OUTPUT_PATH, index=False)

print("\nCustomer segmentation completed!")
print("Total source customers:", len(df))
print("Customers included in output:", len(result))
print("Customers assigned to K-Means:", len(eligible))

print("\nCustomers in each segment:")
print(result["segment_name"].value_counts(dropna=False).to_string())

print("\nAverage RFM behaviour by clustered segment:")
print(
    eligible.groupby("segment_name")[
        ["recency", "frequency", "monetary"]
    ].mean().round(2).to_string()
)

print("\nPremium customers:", int(result["is_premium"].sum()))

print("\nOutput columns:")
print(result.columns.tolist())

print("\nSaved customer segments:", OUTPUT_PATH)
print("Saved K-Means scores:", SCORES_PATH)

print("\nALL SEGMENTATION CHECKS PASSED")
