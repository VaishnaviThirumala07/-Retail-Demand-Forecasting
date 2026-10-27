-- 06_top_bottom_items_by_store.sql
-- Insight: Top and bottom items by revenue within each store using RANK().

WITH item_revenue AS (
    SELECT
        store_id,
        state_id,
        item_id,
        dept_id,
        cat_id,
        ROUND(SUM(revenue), 2)                                    AS total_revenue,
        SUM(units_sold)                                           AS total_units,
        RANK() OVER (PARTITION BY store_id ORDER BY SUM(revenue) DESC) AS revenue_rank,
        RANK() OVER (PARTITION BY store_id ORDER BY SUM(revenue) ASC)  AS revenue_rank_asc,
        COUNT(DISTINCT store_id) OVER (PARTITION BY store_id)          AS n_items_in_store
    FROM fct_daily_sales
    WHERE sell_price IS NOT NULL
    GROUP BY store_id, state_id, item_id, dept_id, cat_id
)

-- Top 5 items per store
SELECT 'TOP' AS rank_type, *
FROM item_revenue
WHERE revenue_rank <= 5

UNION ALL

-- Bottom 5 items per store
SELECT 'BOTTOM' AS rank_type, *
FROM item_revenue
WHERE revenue_rank_asc <= 5

ORDER BY store_id, rank_type, revenue_rank;
