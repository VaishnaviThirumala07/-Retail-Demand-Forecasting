-- 03_event_holiday_impact.sql
-- Insight: Effect of named events (Super Bowl, Christmas, etc.) on sales.
-- Uses a ±7-day window comparison: event day vs. same weekday non-event.

WITH event_days AS (
    SELECT
        sale_date,
        event_name_1,
        event_type_1,
        SUM(units_sold)                AS total_units,
        SUM(revenue)                   AS total_revenue
    FROM fct_daily_sales
    WHERE is_event_day = 1
    GROUP BY sale_date, event_name_1, event_type_1
),

non_event_baseline AS (
    SELECT
        weekday,
        ROUND(AVG(daily_units), 2)     AS baseline_units
    FROM (
        SELECT sale_date, weekday, SUM(units_sold) AS daily_units
        FROM fct_daily_sales
        WHERE is_event_day = 0
        GROUP BY sale_date, weekday
    ) sub
    GROUP BY weekday
),

event_with_baseline AS (
    SELECT
        e.sale_date,
        e.event_name_1,
        e.event_type_1,
        e.total_units,
        f.weekday,
        b.baseline_units,
        ROUND((e.total_units - b.baseline_units)
              / NULLIF(b.baseline_units, 0) * 100, 2) AS lift_pct
    FROM event_days e
    JOIN fct_daily_sales f  ON e.sale_date = f.sale_date
    JOIN non_event_baseline b ON f.weekday  = b.weekday
    GROUP BY e.sale_date, e.event_name_1, e.event_type_1, e.total_units,
             f.weekday, b.baseline_units
)

SELECT
    event_name_1,
    event_type_1,
    COUNT(*)                        AS occurrences,
    ROUND(AVG(total_units), 0)      AS avg_event_units,
    ROUND(AVG(baseline_units), 0)   AS avg_baseline_units,
    ROUND(AVG(lift_pct), 2)         AS avg_lift_pct
FROM event_with_baseline
WHERE event_name_1 != 'none'
GROUP BY event_name_1, event_type_1
ORDER BY avg_lift_pct DESC
LIMIT 20;
