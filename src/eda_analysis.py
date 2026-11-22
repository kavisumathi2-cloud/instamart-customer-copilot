from pathlib import Path
import sqlite3
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    features = pd.read_csv(DATA_DIR / "customer_features.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    return customers, orders, features, products


def descriptive_statistics(customers, orders, features):
    completed = orders[orders["status"] == "completed"].copy()

    sections = {
        "customer_age": customers[["age"]],
        "order_metrics": orders[["quantity", "discount_pct", "order_amount"]],
        "completed_delivery": completed[["delivery_minutes"]],
        "customer_features": features[
            ["completed_orders", "completed_spend", "recency",
             "frequency", "monetary"]
        ],
    }

    results = []

    for section, frame in sections.items():
        stats = frame.describe().T
        stats.insert(0, "section", section)
        stats.insert(1, "metric", stats.index)
        results.append(stats.reset_index(drop=True))

    output = pd.concat(results, ignore_index=True)
    output.to_csv(
        OUTPUT_DIR / "eda_descriptive_statistics.csv", index=False
    )
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(output.to_string(index=False))


def city_status_pivot(customers, orders):
    merged = orders.merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    pivot = pd.pivot_table(
        merged,
        index="city",
        columns="status",
        values="order_amount",
        aggfunc="sum",
        fill_value=0,
    )

    pivot.to_csv(OUTPUT_DIR / "eda_city_order_status_pivot.csv")

    print("\n=== CITY x ORDER STATUS: ORDER AMOUNT ===")
    print(pivot.to_string())

def city_named_aggregation(customers, orders):
    merged = orders.merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    completed = merged[merged["status"] == "completed"].copy()

    city_summary = (
        completed.groupby("city")
        .agg(
            completed_orders=("order_id", "nunique"),
            total_revenue=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )

    city_summary["total_revenue"] = city_summary["total_revenue"].round(2)
    city_summary["average_order_value"] = (
        city_summary["average_order_value"].round(2)
    )

    city_summary.to_csv(
        OUTPUT_DIR / "eda_city_named_aggregation.csv",
        index=False,
    )

    print("\n=== CITY-WISE NAMED AGGREGATION ===")
    print(city_summary.to_string(index=False))

    return city_summary

def pandas_sql_cross_check(customers, orders):
    db_path = ROOT / "data" / "instamart.db"

    with sqlite3.connect(db_path) as connection:
        sql_result = pd.read_sql_query(
            """
            SELECT
                c.city,
                COUNT(DISTINCT o.order_id) AS completed_orders,
                ROUND(SUM(o.order_amount), 2) AS total_revenue,
                ROUND(AVG(o.order_amount), 2) AS average_order_value
            FROM orders AS o
            JOIN customers AS c
                ON o.customer_id = c.customer_id
            WHERE LOWER(o.status) = 'completed'
            GROUP BY c.city
            ORDER BY c.city
            """,
            connection,
        )

    merged = orders.merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    completed = merged[merged["status"] == "completed"]

    pandas_result = (
        completed.groupby("city")
        .agg(
            completed_orders=("order_id", "nunique"),
            total_revenue=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
        )
        .reset_index()
        .sort_values("city")
        .reset_index(drop=True)
    )

    pandas_result["total_revenue"] = pandas_result["total_revenue"].round(2)
    pandas_result["average_order_value"] = (
        pandas_result["average_order_value"].round(2)
    )

    sql_result = sql_result.sort_values("city").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        pandas_result,
        sql_result,
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=0.01,
    )

    print("\n=== SQLITE vs PANDAS CROSS-CHECK ===")
    print("PASS: City-wise completed orders, revenue and average order value match.")
    print(pandas_result.to_string(index=False))

    return True


def correlation_analysis(orders, features):
    completed = orders[orders["status"] == "completed"]

    order_corr = completed[
        ["quantity", "discount_pct", "order_amount", "delivery_minutes"]
    ].corr(numeric_only=True)

    feature_corr = features[
        ["completed_orders", "completed_spend", "recency",
         "frequency", "monetary"]
    ].corr(numeric_only=True)

    order_corr.to_csv(OUTPUT_DIR / "eda_order_correlation_matrix.csv")
    feature_corr.to_csv(OUTPUT_DIR / "eda_customer_feature_correlation.csv")

    print("\n=== COMPLETED ORDER CORRELATION ===")
    print(order_corr.round(3).to_string())

    print("\n=== CUSTOMER FEATURE CORRELATION ===")
    print(feature_corr.round(3).to_string())


def outlier_analysis(orders, customers, features):
    completed = orders[orders["status"] == "completed"]

    datasets = {
        "order_amount": orders["order_amount"],
        "quantity": orders["quantity"],
        "delivery_minutes_completed": completed["delivery_minutes"],
        "customer_age": customers["age"],
        "completed_spend": features["completed_spend"],
    }

    results = []

    for metric, series in datasets.items():
        values = series.dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = values[(values < lower) | (values > upper)]

        results.append({
            "metric": metric,
            "records_checked": len(values),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": len(outliers),
            "outlier_percentage": (
                len(outliers) / len(values) * 100 if len(values) else 0
            ),
        })

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_DIR / "eda_outlier_summary.csv", index=False)

    print("\n=== IQR OUTLIER SUMMARY ===")
    print(result.round(2).to_string(index=False))


def business_insights(customers, orders, features, products):
    completed = orders[orders["status"] == "completed"].copy()
    cancelled = orders[orders["status"] == "cancelled"].copy()

    city_orders = orders.merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
    )

    city_revenue = (
        city_orders[city_orders["status"] == "completed"]
        .groupby("city")["order_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    category_revenue = (
        completed.merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left",
        )
        .groupby("category")["order_amount"]
        .sum()
        .sort_values(ascending=False)
    )

    total_orders = len(orders)
    cancellation_rate = (
        len(cancelled) / total_orders * 100 if total_orders else 0
    )

    lines = [
        "INSTAMART - EXPLORATORY DATA ANALYSIS",
        "=" * 45,
        f"Total customers: {len(customers)}",
        f"Total orders: {total_orders}",
        f"Completed orders: {len(completed)}",
        f"Cancelled orders: {len(cancelled)}",
        f"Cancellation rate: {cancellation_rate:.2f}%",
        f"Completed order revenue: INR {completed['order_amount'].sum():,.2f}",
        "",
        "TOP CITY BY COMPLETED REVENUE:",
    ]

    if not city_revenue.empty:
        lines.append(
            f"{city_revenue.index[0]}: INR {city_revenue.iloc[0]:,.2f}"
        )

    lines.extend(["", "TOP PRODUCT CATEGORY BY COMPLETED REVENUE:"])

    if not category_revenue.empty:
        lines.append(
            f"{category_revenue.index[0]}: "
            f"INR {category_revenue.iloc[0]:,.2f}"
        )

    lines.extend([
        "",
        "CUSTOMER FEATURE SUMMARY:",
        f"Average completed orders per customer: "
        f"{features['completed_orders'].mean():.2f}",
        f"Average completed spend per customer: "
        f"INR {features['completed_spend'].mean():,.2f}",
        "",
        "Interpretation note:",
        "Correlation indicates association, not causation.",
        "IQR outliers are flagged for investigation, not automatically removed.",
        "Delivery analysis uses completed orders only.",
    ])

    report = "\n".join(lines)
    (OUTPUT_DIR / "eda_business_insights.txt").write_text(
        report, encoding="utf-8"
    )

    print("\n=== BUSINESS INSIGHTS ===")
    print(report)


def main():
    customers, orders, features, products = load_data()

    print("Loaded clean datasets successfully.")
    print(
        f"Customers: {len(customers)}, "
        f"Orders: {len(orders)}, "
        f"Features: {len(features)}, "
        f"Products: {len(products)}"
    )

    descriptive_statistics(customers, orders, features)
    city_status_pivot(customers, orders)
    city_named_aggregation(customers, orders)
    pandas_sql_cross_check(customers, orders)
    correlation_analysis(orders, features)
    outlier_analysis(orders, customers, features)
    business_insights(customers, orders, features, products)

    print("\nEDA completed. Reports saved in the outputs folder.")


if __name__ == "__main__":
    main()
