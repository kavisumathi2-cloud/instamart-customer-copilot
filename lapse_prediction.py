
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=FutureWarning)

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DIR = DATA_DIR / "clean"
if not CLEAN_DIR.exists():
    CLEAN_DIR = DATA_DIR / "cleaned"

CUSTOMERS_PATH = CLEAN_DIR / "customers.csv"
ORDERS_PATH = CLEAN_DIR / "orders.csv"
PRODUCTS_PATH = CLEAN_DIR / "products.csv"

SEGMENTS_PATH = OUTPUT_DIR / "customer_segments.csv"
if not SEGMENTS_PATH.exists():
    SEGMENTS_PATH = CLEAN_DIR / "customer_segments.csv"

MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"
RETENTION_PATH = OUTPUT_DIR / "retention_list.csv"

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SEPTEMBER_END = pd.Timestamp("2025-10-01 00:00:00")
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "average_discount_pct",
    "cancellation_rate",
    "average_delivery_minutes",
    "age",
    "tenure_days",
]

CATEGORICAL_FEATURES = [
    "city",
    "gender",
    "favourite_category",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==================================================
# 2. LOAD DATA
# ==================================================

def load_data():
    customers = pd.read_csv(CUSTOMERS_PATH)
    orders = pd.read_csv(ORDERS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)

    required_customer_columns = {
        "customer_id", "signup_date", "city", "age", "gender"
    }
    required_order_columns = {
        "order_id", "customer_id", "product_id", "order_ts",
        "status", "order_amount", "discount_pct", "delivery_minutes"
    }
    required_product_columns = {"product_id", "category"}

    if not required_customer_columns.issubset(customers.columns):
        raise ValueError(
            f"Missing customer columns: "
            f"{required_customer_columns - set(customers.columns)}"
        )

    if not required_order_columns.issubset(orders.columns):
        raise ValueError(
            f"Missing order columns: "
            f"{required_order_columns - set(orders.columns)}"
        )

    if not required_product_columns.issubset(products.columns):
        raise ValueError(
            f"Missing product columns: "
            f"{required_product_columns - set(products.columns)}"
        )

    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], errors="coerce"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

    for column in ["order_amount", "discount_pct", "delivery_minutes"]:
        orders[column] = pd.to_numeric(
            orders[column], errors="coerce"
        )

    return customers, orders, products


# ==================================================
# 3. BUILD COHORT, LABELS AND FEATURES
# ==================================================

def build_lapse_dataset(customers, orders, products):
    valid_statuses = ["completed", "cancelled"]

    # Only pre-cutoff final-status orders can be features.
    before_cutoff = orders[
        (orders["order_ts"] < CUTOFF)
        & orders["status"].isin(valid_statuses)
    ].copy()

    completed_before = before_cutoff[
        before_cutoff["status"] == "completed"
    ].copy()

    # Cohort: customers with at least one completed order
    # strictly before September 1, 2025.
    cohort_ids = completed_before["customer_id"].unique()

    cohort = customers[
        customers["customer_id"].isin(cohort_ids)
    ].copy()

    # Label: no completed order during September means lapsed.
    september_completed = orders[
        (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < SEPTEMBER_END)
        & (orders["status"] == "completed")
    ]

    september_customer_ids = set(
        september_completed["customer_id"]
    )

    cohort["is_lapsed"] = (
        ~cohort["customer_id"].isin(september_customer_ids)
    ).astype(int)

    # Completed-order features before cutoff.
    completed_features = completed_before.groupby("customer_id").agg(
        last_completed_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_discount_pct=("discount_pct", "mean"),
        average_delivery_minutes=("delivery_minutes", "mean"),
    )

    completed_features["recency"] = (
        CUTOFF - completed_features["last_completed_order"]
    ).dt.days

    # Cancellation rate = cancelled / all final-status orders
    # before cutoff.
    all_counts = before_cutoff.groupby("customer_id").agg(
        total_final_orders=("order_id", "nunique")
    )

    cancelled_counts = (
        before_cutoff[before_cutoff["status"] == "cancelled"]
        .groupby("customer_id")["order_id"]
        .nunique()
        .rename("cancelled_orders")
    )

    cancellation_features = all_counts.join(cancelled_counts)
    cancellation_features["cancelled_orders"] = (
        cancellation_features["cancelled_orders"].fillna(0)
    )

    cancellation_features["cancellation_rate"] = (
        cancellation_features["cancelled_orders"]
        / cancellation_features["total_final_orders"].replace(0, np.nan)
    )

    # Favourite category from completed orders before cutoff.
    category_orders = completed_before.merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
    )

    category_counts = (
        category_orders.groupby(["customer_id", "category"])
        .size()
        .reset_index(name="order_count")
        .sort_values(
            ["customer_id", "order_count", "category"],
            ascending=[True, False, True],
        )
    )

    favourite_category = (
        category_counts.drop_duplicates("customer_id")
        .set_index("customer_id")["category"]
        .rename("favourite_category")
    )

    # Combine features with the customer table.
    cohort = cohort.set_index("customer_id")
    cohort = cohort.join(completed_features)
    cohort = cohort.join(
        cancellation_features[["cancellation_rate"]]
    )
    cohort = cohort.join(favourite_category)

    cohort["tenure_days"] = (
        CUTOFF - cohort["signup_date"]
    ).dt.days

    cohort = cohort.reset_index()

    # Keep numeric columns numeric. Missing numeric values are
    # intentionally left for the train-fitted imputer.
    for column in NUMERIC_FEATURES:
        cohort[column] = pd.to_numeric(
            cohort[column], errors="coerce"
        )

    # Categorical missing values are represented as Unknown.
    for column in CATEGORICAL_FEATURES:
        cohort[column] = (
            cohort[column]
            .fillna("Unknown")
            .astype(str)
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )

    # Safety checks: features must only use pre-cutoff orders.
    if not before_cutoff.empty:
        assert before_cutoff["order_ts"].max() < CUTOFF

    if not completed_before.empty:
        assert completed_before["order_ts"].max() < CUTOFF

    return cohort


# ==================================================
# 4. PREPROCESSING PIPELINE
# ==================================================

def make_pipeline(model):
    # Median imputation and scaling are fitted on training data only.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Most-frequent imputation and one-hot encoding are also
    # fitted on training data only.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n{'=' * 50}")
    print(name)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Confusion matrix [not lapsed, lapsed]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Not Lapsed", "Lapsed"],
            zero_division=0,
        )
    )

    return predictions


# ==================================================
# 5. MAIN MODEL WORKFLOW
# ==================================================

def main():
    customers, orders, products = load_data()

    data = build_lapse_dataset(customers, orders, products)

    print("=" * 55)
    print("LAPSE PREDICTION")
    print("=" * 55)
    print("Cutoff:", CUTOFF.date())
    print("September label window: 2025-09-01 to 2025-09-30")
    print("Customers in lapse cohort:", len(data))
    print("\nActual lapse-label counts:")
    print(
        data["is_lapsed"]
        .map({0: "Not Lapsed", 1: "Lapsed"})
        .value_counts()
    )

    X = data[FEATURE_COLUMNS].copy()
    y = data["is_lapsed"].astype(int).copy()

    # Diagnose missing values, but do not stop: numeric missing
    # values will be imputed inside the training pipeline.
    missing_counts = X.isna().sum()
    print("\nMissing values before preprocessing:")
    print(missing_counts[missing_counts > 0].to_string()
          if (missing_counts > 0).any()
          else "No missing values.")

    # Required feature columns must exist. Values may be missing;
    # the pipeline handles those values without dropping customers.
    assert list(X.columns) == FEATURE_COLUMNS
    assert X[CATEGORICAL_FEATURES].notna().all().all()

    X_train, X_test, y_train, y_test, id_train, id_test = (
        train_test_split(
            X,
            y,
            data["customer_id"],
            test_size=0.2,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Baseline: predict every test customer as not lapsed.
    baseline_predictions = np.zeros(len(y_test), dtype=int)

    print("\nBaseline: always predict Not Lapsed")
    print("Accuracy:", round(
        accuracy_score(y_test, baseline_predictions), 3
    ))
    print("Lapsed F1:", round(
        f1_score(y_test, baseline_predictions, zero_division=0), 3
    ))

    # Logistic Regression grid search.
    logistic_pipeline = make_pipeline(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    )

    logistic_search = GridSearchCV(
        logistic_pipeline,
        param_grid={
            "model__C": [0.01, 0.1, 1, 10, 100]
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    logistic_search.fit(X_train, y_train)

    best_logistic = logistic_search.best_estimator_

    print("\nLogistic Regression best parameters:")
    print(logistic_search.best_params_)
    print("Best CV F1:", round(logistic_search.best_score_, 3))

    # Random Forest grid search.
    forest_pipeline = make_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        )
    )

    forest_search = GridSearchCV(
        forest_pipeline,
        param_grid={
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10, 20],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    forest_search.fit(X_train, y_train)

    best_forest = forest_search.best_estimator_

    print("\nRandom Forest best parameters:")
    print(forest_search.best_params_)
    print("Best CV F1:", round(forest_search.best_score_, 3))

    logistic_predictions = evaluate_model(
        "Tuned Logistic Regression",
        best_logistic,
        X_test,
        y_test,
    )

    forest_predictions = evaluate_model(
        "Tuned Random Forest",
        best_forest,
        X_test,
        y_test,
    )

    logistic_f1 = f1_score(
        y_test, logistic_predictions, zero_division=0
    )
    forest_f1 = f1_score(
        y_test, forest_predictions, zero_division=0
    )

    # Select by the required CV F1 score; test data is reserved
    # for final evaluation, not model selection.
    if logistic_search.best_score_ >= forest_search.best_score_:
        selected_model = best_logistic
        selected_name = "Logistic Regression"
    else:
        selected_model = best_forest
        selected_name = "Random Forest"

    print("\nSelected model based on CV F1:", selected_name)
    print("Test F1:", round(
        f1_score(
            y_test,
            selected_model.predict(X_test),
            zero_division=0,
        ),
        3,
    ))

    # Scaler mean is learned from training data only.
    selected_preprocessor = selected_model.named_steps["preprocess"]
    scaler = (
        selected_preprocessor
        .named_transformers_["num"]
        .named_steps["scaler"]
    )

    frequency_index = NUMERIC_FEATURES.index("frequency")

    print("\nTraining scaler mean frequency:",
          round(scaler.mean_[frequency_index], 3))
    print("Full-cohort mean frequency:",
          round(data["frequency"].mean(), 3))
    print(
        "The means may differ because the scaler was fitted only "
        "on the training split."
    )

    # ==================================================
    # 6. OVERSAMPLE TRAINING DATA ONLY
    # ==================================================

    training_data = X_train.copy()
    training_data["target"] = y_train.to_numpy()

    class_counts = training_data["target"].value_counts()
    majority_count = int(class_counts.max())
    minority_class = int(class_counts.idxmin())

    minority_rows = training_data[
        training_data["target"] == minority_class
    ]

    extra_rows = resample(
        minority_rows,
        replace=True,
        n_samples=majority_count - len(minority_rows),
        random_state=RANDOM_STATE,
    )

    balanced_data = pd.concat(
        [training_data, extra_rows],
        ignore_index=True,
    )

    X_balanced = balanced_data[FEATURE_COLUMNS]
    y_balanced = balanced_data["target"]

    # Refit the best logistic C on balanced training data only.
    best_c = best_logistic.named_steps["model"].C

    balanced_logistic = make_pipeline(
        LogisticRegression(
            max_iter=1000,
            C=best_c,
            random_state=RANDOM_STATE,
        )
    )
    balanced_logistic.fit(X_balanced, y_balanced)

    balanced_predictions = balanced_logistic.predict(X_test)

    print("\nOversampling comparison")
    print("Original training class counts:",
          y_train.value_counts().sort_index().to_dict())
    print("Balanced training class counts:",
          y_balanced.value_counts().sort_index().to_dict())

    for label, predictions in [
        ("Original Logistic Regression", logistic_predictions),
        ("Oversampled Logistic Regression", balanced_predictions),
    ]:
        print(f"\n{label}")
        print("Precision:", round(
            precision_score(y_test, predictions, zero_division=0), 3
        ))
        print("Recall:", round(
            recall_score(y_test, predictions, zero_division=0), 3
        ))
        print("F1:", round(
            f1_score(y_test, predictions, zero_division=0), 3
        ))

    # ==================================================
    # 7. FEATURE IMPORTANCE AND COEFFICIENTS
    # ==================================================

    # Random Forest feature importances.
    forest_preprocessor = best_forest.named_steps["preprocess"]
    forest_model = best_forest.named_steps["model"]
    feature_names = forest_preprocessor.get_feature_names_out()

    importance_table = pd.DataFrame({
        "feature": feature_names,
        "importance": forest_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 5 Random Forest features:")
    print(importance_table.head(5).to_string(index=False))

    # Logistic Regression coefficient signs.
    logistic_preprocessor = best_logistic.named_steps["preprocess"]
    logistic_model = best_logistic.named_steps["model"]
    logistic_feature_names = logistic_preprocessor.get_feature_names_out()

    coefficient_table = pd.DataFrame({
        "feature": logistic_feature_names,
        "coefficient": logistic_model.coef_[0],
    })

    print("\nLogistic Regression coefficients:")
    for feature in ["recency", "frequency", "monetary"]:
        match = coefficient_table[
            coefficient_table["feature"].str.endswith(feature)
        ]
        if not match.empty:
            row = match.iloc[0]
            sign = "positive" if row["coefficient"] > 0 else "negative"
            print(f"{feature}: {sign} ({row['coefficient']:.4f})")

    # ==================================================
    # 8. SAVE AND RELOAD MODEL
    # ==================================================

    joblib.dump(selected_model, MODEL_PATH)
    loaded_model = joblib.load(MODEL_PATH)

    predictions_before_save = selected_model.predict(X_test)
    predictions_after_load = loaded_model.predict(X_test)

    assert np.array_equal(
        predictions_before_save,
        predictions_after_load,
    ), "Reloaded model predictions do not match."

    print("\nSaved model:", MODEL_PATH)
    print("Reload verification passed: predictions match exactly.")

    # ==================================================
    # 9. PREMIUM CUSTOMER RETENTION LIST
    # ==================================================

    if not SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            "Customer segments file not found. Expected:\n"
            f"{OUTPUT_DIR / 'customer_segments.csv'}\n"
            f"or {CLEAN_DIR / 'customer_segments.csv'}"
        )

    segments = pd.read_csv(SEGMENTS_PATH)

    if "is_premium" in segments.columns:
        premium_ids = set(
            segments.loc[
                segments["is_premium"].astype(str).str.lower().isin(
                    ["true", "1", "yes"]
                ),
                "customer_id",
            ]
        )
    elif (
        "segment" in segments.columns
        and "completed_spend" in segments.columns
    ):
        segment_spend = segments.groupby("segment")["completed_spend"].mean()
        premium_segment = segment_spend.idxmax()
        premium_ids = set(
            segments.loc[
                segments["segment"] == premium_segment,
                "customer_id",
            ]
        )
        print("Premium segment:", premium_segment)
    elif (
        "segment_name" in segments.columns
        and "monetary" in segments.columns
    ):
        segment_spend = segments.groupby("segment_name")["monetary"].mean()
        premium_segment = segment_spend.idxmax()
        premium_ids = set(
            segments.loc[
                segments["segment_name"] == premium_segment,
                "customer_id",
            ]
        )
        print("Premium segment:", premium_segment)
    else:
        raise ValueError(
            "Unrecognised customer segments CSV columns."
        )

    test_results = pd.DataFrame({
        "customer_id": id_test.to_numpy(),
        "lapse_probability": selected_model.predict_proba(X_test)[:, 1],
        "actually_lapsed": y_test.to_numpy(),
    })

    premium_test_results = test_results[
        test_results["customer_id"].isin(premium_ids)
    ].copy()

    retention_list = (
        premium_test_results
        .sort_values("lapse_probability", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    retention_list.to_csv(RETENTION_PATH, index=False)

    print("\nPremium customers in test set:",
          len(premium_test_results))
    print("Retention list rows:", len(retention_list))
    print("Actually lapsed in top 10:",
          int(retention_list["actually_lapsed"].sum()))
    print("Saved retention list:", RETENTION_PATH)

    print("\nReminder:")
    print(
        "The Premium flag is derived from whole-window RFM. "
        "Therefore, the retention list is illustrative and is not "
        "a fully leakage-free evaluation."
    )


if __name__ == "__main__":
    main()
