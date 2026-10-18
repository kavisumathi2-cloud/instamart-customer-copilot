
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
# 2. LOAD CUSTOMER FEATURES
# ==================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Missing {INPUT_PATH}. Run customer_features.py first."
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

df["last_order_ts"] = pd.to_datetime(
    df["last_order_ts"], errors="coerce"
)

df["completed_orders"] = pd.to_numeric(
    df["completed_orders"], errors="coerce"
).fillna(0)

df["completed_spend"] = pd.to_numeric(
    df["completed_spend"], errors="coerce"
).fillna(0)

# Only customers with completed-order history are eligible.
active = df[df["completed_orders"] > 0].copy()

# ==================================================
# 3. CREATE RFM FEATURES
# ==================================================

active["recency"] = (
    SNAPSHOT_DATE - active["last_order_ts"]
).dt.total_seconds() / 86400

active["frequency"] = active["completed_orders"]
active["monetary"] = active["completed_spend"]

feature_columns = ["recency", "frequency", "monetary"]

# Customers without a valid last order cannot be clustered.
active = active.dropna(subset=feature_columns).copy()

if len(active) < 8:
    raise ValueError(
        "At least 8 eligible customers are needed to evaluate K=2 to K=8."
    )

if (active["recency"] < 0).any():
    raise ValueError(
        "Negative recency found. Check the snapshot date and order timestamps."
    )

X = active[feature_columns].copy()

print("=" * 55)
print("CUSTOMER SEGMENTATION")
print("=" * 55)

print("RFM snapshot date:", SNAPSHOT_DATE.date())
print("Customers with completed orders:", len(active))

print("\nRFM summary:")
print(
    active[feature_columns]
    .agg(["mean", "median", "min", "max"])
    .round(2)
    .to_string()
)

# ==================================================
# 4. SCALE FEATURES
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================================================
# 5. EVALUATE K FROM 2 TO 8
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
# 6. FINAL K-MEANS MODEL: K = 4
# ==================================================

final_model = KMeans(
    n_clusters=4,
    random_state=RANDOM_STATE,
    n_init=10,
)

active["segment_id"] = final_model.fit_predict(X_scaled)

# Name clusters by ascending average monetary value.
spending_order = (
    active.groupby("segment_id")["monetary"]
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

active["segment_name"] = active["segment_id"].map(name_mapping)

# Premium means membership in the highest-spending cluster.
active["is_premium"] = active["segment_name"].eq("Top Value")

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
# 8. PREPARE REQUIRED OUTPUT
# ==================================================

# Keep only eligible customers in the RFM segmentation output.
# Customers with no completed orders remain in the source data,
# but they are not assigned to a K-Means cluster.

result = active[[
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium",
]].copy()

result["segment_id"] = result["segment_id"].astype("Int64")
result["is_premium"] = result["is_premium"].astype(bool)

result = result.sort_values(
    "customer_id"
).reset_index(drop=True)

# ==================================================
# 9. SAVE AND PRINT RESULTS
# ==================================================

result.to_csv(OUTPUT_PATH, index=False)

print("\nCustomer segmentation completed!")
print("Customers included in clustering:", len(active))
print("Total customers in output:", len(result))

print("\nCustomers in each segment:")
print(result["segment_name"].value_counts().to_string())

print("\nAverage RFM behaviour by segment:")
print(
    active.groupby("segment_name")[
        ["recency", "frequency", "monetary"]
    ].mean().round(2).to_string()
)

print("\nPremium customers:", int(result["is_premium"].sum()))

print("\nOutput columns:")
print(result.columns.tolist())

print("\nSaved customer segments:", OUTPUT_PATH)
print("Saved K-Means scores:", SCORES_PATH)
