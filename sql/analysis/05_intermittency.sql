-- 05_intermittency_zero_sales.sql
-- Insight: Share of zero-sales days per item (intermittent demand).
-- Items with > 50% zero days require Tweedie / Croston models.

WITH item_stats AS (
    SELECT
        item_id,
        store_id,
        state_id,
        cat_id,
        dept_id,
        COUNT(*)                                       AS total_days,
        SUM(is_zero_sale)                              AS zero_days,
        ROUND(SUM(is_zero_sale) * 100.0 / COUNT(*), 1) AS zero_pct,
        ROUND(AVG(units_sold), 3)                      AS avg_daily_units,
        ROUND(STDDEV(units_sold) / NULLIF(AVG(units_sold), 0), 3) AS cv  -- Coeff of Variation
    FROM fct_daily_sales
    GROUP BY item_id, store_id, state_id, cat_id, dept_id
),

buckets AS (
    SELECT
        cat_id,
        -- Intermittency buckets
        SUM(CASE WHEN zero_pct  < 10                THEN 1 ELSE 0 END) AS regular_items,
        SUM(CASE WHEN zero_pct >= 10 AND zero_pct < 50 THEN 1 ELSE 0 END) AS erratic_items,
        SUM(CASE WHEN zero_pct >= 50                   THEN 1 ELSE 0 END) AS intermittent_items,
        COUNT(*)                                                            AS total_items,
        ROUND(AVG(zero_pct), 1)                                            AS avg_zero_pct
    FROM item_stats
    GROUP BY cat_id
)

SELECT
    cat_id,
    total_items,
    regular_items,
    erratic_items,
    intermittent_items,
    ROUND(intermittent_items * 100.0 / total_items, 1) AS intermittent_share_pct,
    avg_zero_pct
FROM buckets
ORDER BY intermittent_share_pct DESC;
