
-- Q1: Completed orders over ₹500, customers with total spending above ₹8,000
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spending
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spending DESC;


-- Q2: Completed order revenue by city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_amount
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Products priced above the average product price
SELECT
    product_id,
    product_name,
    category,
    ROUND(price_inr, 2) AS price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: Customers with no completed orders
SELECT
    c.customer_id
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Customers whose completed spending is above
-- the average completed spending per customer
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS completed_spending
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING SUM(order_amount) > (
    SELECT AVG(customer_total)
    FROM (
        SELECT SUM(order_amount) AS customer_total
        FROM orders
        WHERE status = 'completed'
        GROUP BY customer_id
    ) AS customer_spending
)
ORDER BY completed_spending DESC;


-- Q6: Compare RANK and ROW_NUMBER to identify ties
WITH customer_city_spending AS (
    SELECT
        c.city,
        o.customer_id,
        SUM(o.order_amount) AS total_spending
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
),
ranked AS (
    SELECT
        city,
        customer_id,
        total_spending,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC
        ) AS spending_rank,
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spending DESC, customer_id
        ) AS row_number_rank
    FROM customer_city_spending
)
SELECT
    city,
    customer_id,
    ROUND(total_spending, 2) AS total_spending,
    spending_rank,
    row_number_rank
FROM ranked
WHERE spending_rank <> row_number_rank
ORDER BY city, spending_rank, customer_id;


-- Q7: Daily completed revenue and cumulative revenue
WITH daily_spending AS (
    SELECT
        DATE(order_ts) AS order_date,
        SUM(order_amount) AS daily_amount
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_date,
    ROUND(daily_amount, 2) AS daily_amount,
    ROUND(
        SUM(daily_amount) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_amount
FROM daily_spending
ORDER BY order_date;
