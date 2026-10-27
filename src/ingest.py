"""
DS31 Phase 1: Data Ingestion
Loads M5 CSVs into DuckDB raw layer and melts wide sales to long format.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import duckdb
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"

# Expected M5 files
M5_FILES = {
    "sales_train_evaluation.csv": "raw_sales_wide",
    "calendar.csv": "raw_calendar",
    "sell_prices.csv": "raw_prices",
}

app = typer.Typer(help="DS31 Ingestion CLI")


def get_connection() -> duckdb.DuckDBPyConnection:
    WAREHOUSE.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(WAREHOUSE))


def load_raw_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Load all M5 CSVs as raw DuckDB tables."""
    for fname, table in M5_FILES.items():
        fpath = DATA_RAW / fname
        if not fpath.exists():
            console.print(f"[red]Missing file:[/red] {fpath}")
            console.print(
                "[yellow]Download from:[/yellow] https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"
            )
            sys.exit(1)

        console.print(f"Loading [cyan]{fname}[/cyan] → [green]{table}[/green]")
        t0 = time.time()
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"""
            CREATE TABLE {table} AS
            SELECT * FROM read_csv_auto('{fpath.as_posix()}', header=true)
            """
        )
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        console.print(f"  {n:,} rows loaded in {time.time()-t0:.1f}s")


def melt_sales(con: duckdb.DuckDBPyConnection) -> None:
    """
    Unpivot wide sales (d_1..d_1941) to long format using DuckDB UNPIVOT.
    Result: raw_sales_long (item_id, dept_id, cat_id, store_id, state_id, d, units)
    ~30,490 series × 1,941 days ≈ 59M rows
    """
    console.print("\nMelting wide → long format (UNPIVOT in DuckDB)…")
    t0 = time.time()

    # First get the day columns dynamically
    cols = con.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'raw_sales_wide'
          AND column_name LIKE 'd_%'
        ORDER BY CAST(REGEXP_EXTRACT(column_name, '\\d+') AS INTEGER)
        """
    ).fetchall()
    day_cols = [c[0] for c in cols]

    console.print(f"  Found {len(day_cols)} day columns ({day_cols[0]} … {day_cols[-1]})")

    unpivot_cols = ", ".join(day_cols)
    con.execute("DROP TABLE IF EXISTS raw_sales_long")
    con.execute(
        f"""
        CREATE TABLE raw_sales_long AS
        SELECT
            id,
            item_id,
            dept_id,
            cat_id,
            store_id,
            state_id,
            d,
            CAST(units AS INTEGER) AS units
        FROM raw_sales_wide
        UNPIVOT (units FOR d IN ({unpivot_cols}))
        """
    )

    n = con.execute("SELECT COUNT(*) FROM raw_sales_long").fetchone()[0]
    console.print(f"  [bold green]{n:,}[/bold green] rows in long format ({time.time()-t0:.1f}s)")
    console.print("  (30,490 series × 1,941 days ≈ 59,121,690 rows expected)")


def join_calendar_and_prices(con: duckdb.DuckDBPyConnection) -> None:
    """Create raw_sales_enriched: long sales + calendar dates + prices."""
    console.print("\nJoining calendar dates and sell prices…")
    t0 = time.time()

    con.execute("DROP TABLE IF EXISTS raw_sales_enriched")
    con.execute(
        """
        CREATE TABLE raw_sales_enriched AS
        SELECT
            s.id,
            s.item_id,
            s.dept_id,
            s.cat_id,
            s.store_id,
            s.state_id,
            s.d,
            s.units,
            c.date::DATE                         AS sale_date,
            c.wm_yr_wk,
            c.weekday,
            c.wday,
            c.month,
            c.year,
            c.event_name_1,
            c.event_type_1,
            c.event_name_2,
            c.event_type_2,
            c.snap_CA,
            c.snap_TX,
            c.snap_WI,
            p.sell_price
        FROM raw_sales_long s
        LEFT JOIN raw_calendar c  ON s.d = c.d
        LEFT JOIN raw_prices   p  ON s.store_id  = p.store_id
                                  AND s.item_id   = p.item_id
                                  AND c.wm_yr_wk  = p.wm_yr_wk
        """
    )

    n = con.execute("SELECT COUNT(*) FROM raw_sales_enriched").fetchone()[0]
    console.print(f"  [bold green]{n:,}[/bold green] enriched rows ({time.time()-t0:.1f}s)")


def run_quality_checks(con: duckdb.DuckDBPyConnection) -> None:
    """Basic data-quality assertions on raw layer."""
    console.print("\nRunning raw-layer data quality checks…")

    checks = {
        "No negative units": "SELECT COUNT(*) FROM raw_sales_long WHERE units < 0",
        "No null dates": "SELECT COUNT(*) FROM raw_sales_enriched WHERE sale_date IS NULL",
        "No null item_id": "SELECT COUNT(*) FROM raw_sales_long WHERE item_id IS NULL",
        "Price coverage (rows with price)": (
            "SELECT COUNT(*) FROM raw_sales_enriched WHERE sell_price IS NOT NULL"
        ),
    }

    for label, sql in checks.items():
        val = con.execute(sql).fetchone()[0]
        status = "[green]✓[/green]" if (label.startswith("Price") or val == 0) else "[red]✗[/red]"
        console.print(f"  {status} {label}: {val:,}")


@app.command()
def main(
    data_dir: Path = typer.Option(DATA_RAW, help="Directory with M5 CSVs"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB file path"),
) -> None:
    """Ingest M5 CSVs into DuckDB warehouse."""
    global DATA_RAW, WAREHOUSE
    DATA_RAW = data_dir
    WAREHOUSE = warehouse

    console.print("[bold cyan]DS31 — Phase 1: Ingestion[/bold cyan]\n")
    con = get_connection()

    load_raw_tables(con)
    melt_sales(con)
    join_calendar_and_prices(con)
    run_quality_checks(con)

    con.close()
    console.print("\n[bold green]✓ Ingestion complete.[/bold green]")
    console.print(f"  Warehouse: [cyan]{WAREHOUSE}[/cyan]")


if __name__ == "__main__":
    app()
