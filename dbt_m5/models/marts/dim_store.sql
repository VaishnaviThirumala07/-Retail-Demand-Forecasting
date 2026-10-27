-- dim_store.sql
{{ config(materialized='table') }}

SELECT DISTINCT
    store_id,
    state_id,
    CAST(REGEXP_EXTRACT(store_id, '(\d+)$') AS INTEGER) AS store_num
FROM {{ ref('stg_sales') }}
ORDER BY state_id, store_id
