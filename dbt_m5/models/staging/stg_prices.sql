-- stg_prices.sql
{{ config(materialized='view') }}

SELECT
    store_id,
    item_id,
    wm_yr_wk,
    CAST(sell_price AS DOUBLE) AS sell_price
FROM {{ source('raw', 'raw_prices') }}
WHERE sell_price > 0
