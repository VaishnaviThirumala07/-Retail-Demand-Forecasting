-- 02_snap_day_lift.sql
-- Insight: SNAP (Supplemental Nutrition Assistance Program) days drive a
-- measurable lift in food category sales in CA, TX, WI.
-- SNAP days are the 1st, 2nd and 3rd of each month (varies by state).

WITH snap_sales AS (
    SELECT
        state_id,
        snap_flag,
        cat_id,
        ROUND(AVG(units_sold), 4)       AS avg_units,
        COUNT(*)                         AS n_rows
    FROM fct_daily_sales
    GROUP BY state_id, snap_flag, cat_id
),

pivoted AS (
    SELECT
        state_id,
        cat_id,
        MAX(CASE WHEN snap_flag = 1 THEN avg_units END) AS snap_avg,
        MAX(CASE WHEN snap_flag = 0 THEN avg_units END) AS non_snap_avg
    FROM snap_sales
    GROUP BY state_id, cat_id
)

SELECT
    state_id,
    cat_id,
    ROUND(snap_avg, 3)                           AS snap_avg_units,
    ROUND(non_snap_avg, 3)                       AS non_snap_avg_units,
    ROUND((snap_avg - non_snap_avg)
          / NULLIF(non_snap_avg, 0) * 100, 2)   AS snap_lift_pct
FROM pivoted
WHERE non_snap_avg IS NOT NULL
  AND snap_avg     IS NOT NULL
ORDER BY snap_lift_pct DESC;
