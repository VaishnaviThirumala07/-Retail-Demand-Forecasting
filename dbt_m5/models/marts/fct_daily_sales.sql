-- fct_daily_sales.sql
-- Core fact table: daily sales with all dimensions joined
{{ config(materialized='table') }}

SELECT
    -- Grain: item × store × date
    s.item_id || '_' || s.store_id || '_' || s.sale_date::VARCHAR AS surrogate_key,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    s.sale_date,
    s.day_id,
    s.wm_yr_wk,
    -- Sales
    s.units_sold,
    s.sell_price,
    s.revenue,
    -- Calendar attributes
    c.weekday,
    c.wday,
    c.month,
    c.year,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    -- SNAP
    s.snap_flag,
    c.snap_CA,
    c.snap_TX,
    c.snap_WI,
    -- Derived flags
    CASE WHEN s.units_sold = 0 THEN 1 ELSE 0 END        AS is_zero_sale,
    CASE WHEN c.event_name_1 != 'none' THEN 1 ELSE 0 END AS is_event_day
FROM {{ ref('stg_sales') }} s
LEFT JOIN {{ ref('stg_calendar') }} c ON s.day_id = c.day_id
