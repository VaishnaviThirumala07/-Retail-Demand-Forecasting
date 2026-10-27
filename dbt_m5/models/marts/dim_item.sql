-- dim_item.sql
{{ config(materialized='table') }}

SELECT DISTINCT
    item_id,
    dept_id,
    cat_id,
    -- Item number within department
    CAST(REGEXP_EXTRACT(item_id, '(\d+)$') AS INTEGER) AS item_num
FROM {{ ref('stg_sales') }}
ORDER BY cat_id, dept_id, item_id
