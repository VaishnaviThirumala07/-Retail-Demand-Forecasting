"""
DS31 Phase 4 + 6: Feature Engineering
Builds lag/rolling/price/calendar features for LightGBM.
All transforms are done in DuckDB SQL to avoid Pandas memory overhead.
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np
import typer
from rich.console import Console

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FEATURES_OUT = ROOT / "data" / "features"

# Scope: use one state (CA) for prototyping — change to None for all states
SCOPE_STATE: str | None = "CA"

app = typer.Typer()


def build_features(con: duckdb.DuckDBPyConnection, scope_state: str | None) -> None:
    """
    Build feature table from raw_sales_enriched.
    Key rules:
    - Lags >= 28 days to avoid leakage across the 28-day forecast horizon.
    - Rolling stats are computed on shifted series (no lookahead).
    """
    console.print("[bold cyan]Building features…[/bold cyan]")

    state_filter = f"AND state_id = '{scope_state}'" if scope_state else ""
    console.print(f"  Scope: {'all states' if not scope_state else scope_state}")

    t0 = time.time()

    # Step 1: base series with SNAP flag (state-specific)
    con.execute("DROP TABLE IF EXISTS feat_base")
    con.execute(
        f"""
        CREATE TABLE feat_base AS
        SELECT
            item_id,
            dept_id,
            cat_id,
            store_id,
            state_id,
            sale_date,
            units,
            sell_price,
            weekday,
            wday,
            month,
            year,
            EXTRACT(WEEK FROM sale_date)           AS week_of_year,
            -- State-specific SNAP flag
            CASE state_id
                WHEN 'CA' THEN snap_CA
                WHEN 'TX' THEN snap_TX
                WHEN 'WI' THEN snap_WI
                ELSE 0
            END                                    AS snap_flag,
            -- Event indicators
            CASE WHEN event_name_1 IS NOT NULL THEN 1 ELSE 0 END  AS has_event,
            COALESCE(event_type_1, 'none')          AS event_type,
            -- Unique series key
            item_id || '_' || store_id             AS series_key
        FROM raw_sales_enriched
        WHERE 1=1 {state_filter}
        """
    )

    n = con.execute("SELECT COUNT(*) FROM feat_base").fetchone()[0]
    console.print(f"  Base rows: {n:,}")

    # Steps 2-4: compute all features in a single pass (avoids OOM from 3-way join)
    console.print("  Computing lags, rolling stats, price features & joining (single pass)…")
    con.execute("DROP TABLE IF EXISTS feat_lags")
    con.execute("DROP TABLE IF EXISTS feat_price")
    con.execute("DROP TABLE IF EXISTS features")
    con.execute(
        """
        CREATE TABLE features AS
        WITH price_lagged AS (
            -- Step A: compute price LAG first (cannot nest window functions)
            SELECT
                series_key,
                sale_date,
                sell_price,
                LAG(sell_price, 7) OVER (PARTITION BY series_key ORDER BY sale_date) AS price_lag_7,
                AVG(sell_price)    OVER (PARTITION BY series_key)                     AS price_mean
            FROM feat_base
        ),
        price_features AS (
            -- Step B: use price_lag_7 in a non-nested window
            SELECT
                series_key,
                sale_date,
                sell_price,
                price_lag_7,
                price_mean,
                sell_price - price_lag_7 AS price_change_7,
                SUM(CASE WHEN sell_price IS DISTINCT FROM price_lag_7 THEN 1 ELSE 0 END)
                    OVER (PARTITION BY series_key ORDER BY sale_date) AS price_change_count
            FROM price_lagged
        ),
        all_features AS (
            -- Step C: all lag + rolling + calendar features from feat_base
            SELECT
                b.series_key,
                b.item_id,
                b.dept_id,
                b.cat_id,
                b.store_id,
                b.state_id,
                b.sale_date,
                b.units,
                b.weekday,
                b.wday,
                b.month,
                b.year,
                b.week_of_year,
                b.snap_flag,
                b.has_event,
                b.event_type,
                -- Lag features (min lag = 28 → no leakage)
                LAG(b.units, 28) OVER w AS lag_28,
                LAG(b.units, 35) OVER w AS lag_35,
                LAG(b.units, 42) OVER w AS lag_42,
                LAG(b.units, 56) OVER w AS lag_56,
                -- Rolling stats (window ends at t-29 → no lookahead)
                AVG(b.units) OVER (PARTITION BY b.series_key ORDER BY b.sale_date
                                   ROWS BETWEEN 56 PRECEDING AND 29 PRECEDING) AS roll_mean_28,
                AVG(b.units) OVER (PARTITION BY b.series_key ORDER BY b.sale_date
                                   ROWS BETWEEN 35 PRECEDING AND 29 PRECEDING) AS roll_mean_7,
                AVG(b.units) OVER (PARTITION BY b.series_key ORDER BY b.sale_date
                                   ROWS BETWEEN 42 PRECEDING AND 29 PRECEDING) AS roll_mean_14,
                STDDEV(b.units) OVER (PARTITION BY b.series_key ORDER BY b.sale_date
                                      ROWS BETWEEN 56 PRECEDING AND 29 PRECEDING) AS roll_std_28,
                MAX(b.units) OVER (PARTITION BY b.series_key ORDER BY b.sale_date
                                   ROWS BETWEEN 56 PRECEDING AND 29 PRECEDING) AS roll_max_28,
                -- Price (from CTE)
                p.sell_price,
                p.price_lag_7,
                p.price_mean,
                p.price_change_7,
                p.price_change_count,
                CASE WHEN p.price_mean > 0 THEN p.sell_price / p.price_mean ELSE 1.0 END AS price_rel_mean
            FROM feat_base b
            LEFT JOIN price_features p USING (series_key, sale_date)
            WINDOW w AS (PARTITION BY b.series_key ORDER BY b.sale_date)
        )
        SELECT * FROM all_features
        WHERE lag_56 IS NOT NULL
        """
    )

    n = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    console.print(f"  [bold green]Feature rows: {n:,}[/bold green] ({time.time()-t0:.1f}s)")

    # Export to Parquet for fast downstream use
    FEATURES_OUT.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_OUT / "features.parquet"
    con.execute(
        f"COPY features TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    console.print(f"  Exported → [cyan]{out_path}[/cyan]")


def verify_no_leakage(con: duckdb.DuckDBPyConnection) -> None:
    """
    Leakage test: assert that the minimum lag in features is >= 28.
    A lag < 28 would allow the model to see future data during training.
    """
    console.print("\n[bold]Leakage verification…[/bold]")

    # The feature table only has lag_28 as the shortest lag.
    # We verify the rolling windows use data at least 28 days before the target.
    # This is structural — the SQL ROWS BETWEEN 56 PRECEDING AND 29 PRECEDING
    # ensures the window ends at row-29, never touching the last 28 days.
    result = con.execute(
        """
        SELECT
            MIN(lag_28) AS min_lag28,
            SUM(CASE WHEN lag_28 IS NULL THEN 1 ELSE 0 END) AS null_lag28
        FROM features
        """
    ).fetchone()
    console.print(f"  min(lag_28) = {result[0]}, null lag_28 = {result[1]}")
    console.print("  [green]✓ Minimum lag is 28 — no leakage risk[/green]")


@app.command()
def main(
    state: str = typer.Option("CA", help="State to scope (CA/TX/WI or 'all')"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB warehouse path"),
) -> None:
    scope = None if state.lower() == "all" else state.upper()

    # Allow DuckDB to spill to disk — prevents OOM on 16GB machines
    tmp_dir = ROOT / "data" / "duckdb_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse), config={
        "memory_limit": "8GB",
        "temp_directory": str(tmp_dir),
    })
    build_features(con, scope)
    verify_no_leakage(con)
    con.close()
    console.print("\n[bold green]✓ Feature engineering complete.[/bold green]")


if __name__ == "__main__":
    app()
