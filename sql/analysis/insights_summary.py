"""Compute the headline SQL insights quoted in README.md / reports/insights.md."""
from pathlib import Path

import duckdb

WAREHOUSE = Path(__file__).resolve().parents[2] / "warehouse" / "m5.duckdb"

QUERIES = {
    "snap_lift_ca_foods": """
        WITH daily AS (
            SELECT sale_date, snap_CA, SUM(units) AS units
            FROM raw_sales_enriched
            WHERE state_id = 'CA' AND cat_id = 'FOODS'
            GROUP BY sale_date, snap_CA
        )
        SELECT
            AVG(units) FILTER (WHERE snap_CA = 1) AS snap_avg,
            AVG(units) FILTER (WHERE snap_CA = 0) AS non_snap_avg,
            100 * (AVG(units) FILTER (WHERE snap_CA = 1)
                   / AVG(units) FILTER (WHERE snap_CA = 0) - 1) AS lift_pct
        FROM daily
    """,
    "weekday_index_all_states": """
        WITH daily AS (
            SELECT sale_date, weekday, SUM(units) AS units
            FROM raw_sales_enriched GROUP BY sale_date, weekday
        )
        SELECT weekday, ROUND(100 * AVG(units) / (SELECT AVG(units) FROM daily), 1) AS idx
        FROM daily GROUP BY weekday ORDER BY idx DESC
    """,
    "pareto_top20_items_revenue_share": """
        WITH rev AS (
            SELECT item_id, SUM(units * sell_price) AS revenue
            FROM raw_sales_enriched WHERE sell_price IS NOT NULL GROUP BY item_id
        ), ranked AS (
            SELECT revenue, ROW_NUMBER() OVER (ORDER BY revenue DESC) AS rn,
                   COUNT(*) OVER () AS n
            FROM rev
        )
        SELECT ROUND(100 * SUM(revenue) FILTER (WHERE rn <= 0.2 * n) / SUM(revenue), 1) AS top20_share_pct
        FROM ranked
    """,
    "intermittency": """
        WITH s AS (
            SELECT item_id, store_id, AVG(CASE WHEN units = 0 THEN 1 ELSE 0 END) AS zero_share
            FROM raw_sales_long GROUP BY item_id, store_id
        )
        SELECT COUNT(*) AS series,
               ROUND(100 * AVG(CASE WHEN zero_share > 0.5 THEN 1 ELSE 0 END), 1) AS pct_over_half_zero
        FROM s
    """,
    "christmas_and_superbowl": """
        WITH daily AS (
            SELECT sale_date, event_name_1, wday, SUM(units) AS units,
                   SUM(units) FILTER (WHERE dept_id = 'FOODS_3') AS foods3
            FROM raw_sales_enriched GROUP BY sale_date, event_name_1, wday
        ), sundays AS (
            SELECT d.sale_date, d.event_name_1, d.foods3,
                   (SELECT AVG(o.foods3) FROM daily o
                    WHERE o.wday = d.wday AND o.event_name_1 IS NULL
                      AND o.sale_date BETWEEN d.sale_date - INTERVAL 28 DAY
                                          AND d.sale_date + INTERVAL 28 DAY) AS ref
            FROM daily d WHERE d.event_name_1 = 'SuperBowl'
        )
        SELECT
            (SELECT ROUND(100 * AVG(units) / (SELECT AVG(units) FROM daily), 1)
             FROM daily WHERE event_name_1 = 'Christmas') AS christmas_day_pct_of_avg,
            (SELECT ROUND(100 * (AVG(foods3 / ref) - 1), 1) FROM sundays) AS superbowl_foods3_lift_pct
    """,
}


def main() -> None:
    con = duckdb.connect(str(WAREHOUSE), read_only=True,
                         config={"memory_limit": "1GB", "threads": 2})
    for name, sql in QUERIES.items():
        print(f"\n== {name}")
        print(con.execute(sql).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
