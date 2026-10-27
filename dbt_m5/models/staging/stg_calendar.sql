-- stg_calendar.sql
{{ config(materialized='view') }}

SELECT
    d                                   AS day_id,
    date::DATE                          AS calendar_date,
    wm_yr_wk,
    weekday,
    wday,
    month,
    year,
    COALESCE(event_name_1, 'none')      AS event_name_1,
    COALESCE(event_type_1, 'none')      AS event_type_1,
    COALESCE(event_name_2, 'none')      AS event_name_2,
    COALESCE(event_type_2, 'none')      AS event_type_2,
    snap_CA,
    snap_TX,
    snap_WI
FROM {{ source('raw', 'raw_calendar') }}
