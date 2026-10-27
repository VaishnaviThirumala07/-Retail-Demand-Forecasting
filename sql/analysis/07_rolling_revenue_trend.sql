-- 07_rolling_28day_revenue_trend.sql
-- Insight: 28-day rolling revenue trend using window frames.
-- Identifies periods of sustained growth or decline per store.

WITH daily_store_revenue AS (
    SELECT
        store_id,
        state_id,
        sale_date,
        SUM(revenue) AS daily_revenue
    FROM fct_daily_sales
    GROUP BY store_id, state_id, sale_date
),

rolling AS (
    SELECT
        store_id,
        state_id,
        sale_date,
        daily_revenue,
        -- 28-day rolling average
        ROUND(
            AVG(daily_revenue) OVER (
                PARTITION BY store_id
                ORDER BY sale_date
                ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            ), 2
        ) AS roll_28_avg,
        -- Year-over-year growth (rolling 28-day YoY)
        LAG(daily_revenue, 365) OVER (
            PARTITION BY store_id ORDER BY sale_date
        ) AS yoy_daily_revenue
    FROM daily_store_revenue
)

SELECT
    store_id,
    state_id,
    sale_date,
    ROUND(daily_revenue, 2)                           AS daily_revenue,
    roll_28_avg,
    ROUND(
        (roll_28_avg - LAG(roll_28_avg, 28) OVER (PARTITION BY store_id ORDER BY sale_date))
        / NULLIF(LAG(roll_28_avg, 28) OVER (PARTITION BY store_id ORDER BY sale_date), 0)
        * 100, 2
    )                                                 AS mom_growth_pct,
    ROUND(
        (daily_revenue - yoy_daily_revenue)
        / NULLIF(yoy_daily_revenue, 0) * 100, 2
    )                                                 AS yoy_growth_pct
FROM rolling
WHERE sale_date >= (SELECT MIN(sale_date) + INTERVAL '28 days' FROM daily_store_revenue)
ORDER BY store_id, sale_date;
