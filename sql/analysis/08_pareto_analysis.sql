-- 08_pareto_80_20.sql
-- Insight: Pareto analysis — what % of items drive 80% of revenue?
-- Classic 80/20 applies: top ~20% of items produce ~80% of sales.

WITH item_revenue AS (
    SELECT
        item_id,
        cat_id,
        dept_id,
        ROUND(SUM(revenue), 2) AS total_revenue
    FROM fct_daily_sales
    WHERE sell_price IS NOT NULL
    GROUP BY item_id, cat_id, dept_id
),

ranked AS (
    SELECT
        item_id,
        cat_id,
        dept_id,
        total_revenue,
        SUM(total_revenue) OVER ()                    AS grand_total,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC) AS rn,
        COUNT(*) OVER ()                               AS total_items,
        -- Cumulative revenue share
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC
                                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            / SUM(total_revenue) OVER ()              AS cumulative_share
    FROM item_revenue
)

SELECT
    -- Bucket into top 10%, 20%, 30%, 50%, rest
    CASE
        WHEN rn * 100.0 / total_items <= 10 THEN 'Top 10%'
        WHEN rn * 100.0 / total_items <= 20 THEN 'Top 20%'
        WHEN rn * 100.0 / total_items <= 30 THEN 'Top 30%'
        WHEN rn * 100.0 / total_items <= 50 THEN 'Top 50%'
        ELSE 'Bottom 50%'
    END AS item_bucket,
    COUNT(*) AS n_items,
    ROUND(SUM(total_revenue), 0) AS bucket_revenue,
    ROUND(SUM(total_revenue) / MAX(grand_total) * 100, 2) AS revenue_share_pct
FROM ranked
GROUP BY item_bucket
ORDER BY revenue_share_pct DESC;
