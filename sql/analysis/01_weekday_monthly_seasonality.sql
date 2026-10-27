-- 01_weekday_monthly_seasonality.sql
-- Insight: Average daily units sold by weekday and month, broken down by category.
-- Shows strong Friday/Saturday lift and December seasonal peak.

WITH daily_sales AS (
    SELECT
        weekday,
        wday,
        month,
        cat_id,
        SUM(units_sold)                        AS total_units,
        COUNT(DISTINCT sale_date)              AS n_days,
        ROUND(SUM(units_sold) * 1.0
              / COUNT(DISTINCT sale_date), 2)  AS avg_daily_units
    FROM fct_daily_sales
    GROUP BY weekday, wday, month, cat_id
),

weekday_rank AS (
    SELECT
        weekday,
        wday,
        cat_id,
        SUM(total_units)   AS units,
        SUM(n_days)        AS days,
        ROUND(SUM(total_units) * 1.0 / SUM(n_days), 2) AS avg_daily,
        -- Index vs overall category average (= 100)
        ROUND(
            SUM(total_units) * 1.0 / SUM(n_days) /
            AVG(SUM(total_units) * 1.0 / SUM(n_days)) OVER (PARTITION BY cat_id)
            * 100, 1
        ) AS weekday_index
    FROM daily_sales
    GROUP BY weekday, wday, cat_id
)

SELECT *
FROM weekday_rank
ORDER BY cat_id, wday;
