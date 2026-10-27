-- stg_sales.sql
-- Staging model: type-cast and rename raw long sales
{{ config(materialized='view') }}

SELECT
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d                   AS day_id,
    units               AS units_sold,
    sale_date,
    wm_yr_wk,
    weekday,
    wday,
    month,
    year,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    snap_CA,
    snap_TX,
    snap_WI,
    CAST(sell_price AS DOUBLE)  AS sell_price,
    -- State-specific SNAP flag
    CASE state_id
        WHEN 'CA' THEN snap_CA
        WHEN 'TX' THEN snap_TX
        WHEN 'WI' THEN snap_WI
        ELSE 0
    END AS snap_flag,
    -- Revenue
    COALESCE(units * sell_price, 0) AS revenue
FROM {{ source('raw', 'raw_sales_enriched') }}
