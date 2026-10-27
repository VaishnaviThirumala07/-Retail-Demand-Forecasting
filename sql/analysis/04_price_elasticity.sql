-- 04_price_elasticity_proxy.sql
-- Insight: Elasticity proxy — % change in units when price changes.
-- Uses weekly LAG to detect price changes and measure unit response.

WITH weekly_prices AS (
    SELECT
        item_id,
        store_id,
        wm_yr_wk,
        AVG(sell_price)    AS avg_price,
        SUM(units_sold)    AS total_units
    FROM fct_daily_sales
    WHERE sell_price IS NOT NULL
    GROUP BY item_id, store_id, wm_yr_wk
),

price_changes AS (
    SELECT
        item_id,
        store_id,
        wm_yr_wk,
        avg_price,
        total_units,
        LAG(avg_price)   OVER w AS prev_price,
        LAG(total_units) OVER w AS prev_units
    FROM weekly_prices
    WINDOW w AS (PARTITION BY item_id, store_id ORDER BY wm_yr_wk)
),

elasticity AS (
    SELECT
        item_id,
        store_id,
        ROUND((avg_price  - prev_price)  / NULLIF(prev_price, 0) * 100, 2) AS price_chg_pct,
        ROUND((total_units - prev_units) / NULLIF(prev_units, 0) * 100, 2) AS units_chg_pct,
        -- Point elasticity: %Δ quantity / %Δ price
        ROUND(
            ((total_units - prev_units) / NULLIF(prev_units, 0))
            / NULLIF((avg_price - prev_price) / NULLIF(prev_price, 0), 0),
            3
        ) AS point_elasticity
    FROM price_changes
    WHERE prev_price IS NOT NULL
      AND ABS((avg_price - prev_price) / NULLIF(prev_price, 0)) > 0.01  -- at least 1% price change
      AND ABS((avg_price - prev_price) / NULLIF(prev_price, 0)) < 0.5   -- cap outliers
)

SELECT
    LEFT(item_id, 6) AS category_prefix,
    COUNT(*)                                        AS n_price_changes,
    ROUND(MEDIAN(price_chg_pct), 2)                AS median_price_chg_pct,
    ROUND(MEDIAN(units_chg_pct), 2)                AS median_units_chg_pct,
    ROUND(MEDIAN(point_elasticity), 3)             AS median_elasticity,
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
          (ORDER BY point_elasticity), 3)          AS p25_elasticity,
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
          (ORDER BY point_elasticity), 3)          AS p75_elasticity
FROM elasticity
WHERE point_elasticity BETWEEN -10 AND 2  -- remove extreme outliers
GROUP BY category_prefix
ORDER BY n_price_changes DESC;
