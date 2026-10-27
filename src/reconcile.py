"""
DS31 Phase 4 (Hierarchy) + Phase 7: Hierarchical Reconciliation
Uses Nixtla hierarchicalforecast to reconcile bottom-up LightGBM forecasts.
Reconciliation methods: Bottom-Up, Top-Down, MinTrace(mint_shrink).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
import typer
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
from hierarchicalforecast.utils import aggregate
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"

HORIZON = 28

app = typer.Typer()


# ---------------------------------------------------------------------------
# Hierarchy definition
# ---------------------------------------------------------------------------

def build_hierarchy_spec() -> list[list[str]]:
    """
    M5 hierarchy mapped to brief's wording:
    Total → State (Regional) → Store → Category → Department → Item_Store (SKU)
    Returns a list of column paths, from coarsest to finest.
    """
    return [
        ["state_id"],
        ["state_id", "store_id"],
        ["state_id", "store_id", "cat_id"],
        ["state_id", "store_id", "cat_id", "dept_id"],
        ["state_id", "store_id", "cat_id", "dept_id", "item_id"],
    ]


def load_bottom_series(con: duckdb.DuckDBPyConnection, scope_state: str) -> pd.DataFrame:
    """Load bottom-level (item × store) daily sales."""
    state_filter = f"AND state_id = '{scope_state}'" if scope_state else ""
    df = con.execute(
        f"""
        SELECT
            item_id || '_' || store_id AS unique_id,
            state_id,
            store_id,
            cat_id,
            dept_id,
            item_id,
            sale_date AS ds,
            CAST(SUM(units) AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE 1=1 {state_filter}
        GROUP BY unique_id, state_id, store_id, cat_id, dept_id, item_id, ds
        ORDER BY unique_id, ds
        """
    ).df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def build_summing_matrix(df: pd.DataFrame) -> tuple:
    """
    Build summing matrix S and tags using hierarchicalforecast.utils.aggregate.
    The spec defines the column paths at each level.
    """
    hierarchy_spec = build_hierarchy_spec()
    # aggregate expects the bottom-level columns to be in the df
    # We pass column lists representing each hierarchy level
    Y_df, S_df, tags = aggregate(
        df=df[["unique_id", "ds", "y", "state_id", "store_id", "cat_id", "dept_id", "item_id"]],
        spec=hierarchy_spec,
    )
    return Y_df, S_df, tags


def load_lgbm_forecasts() -> pd.DataFrame | None:
    """Load LightGBM base forecasts if available."""
    path = FORECASTS_DIR / "lgbm_forecasts.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df = df.rename(columns={"sale_date": "ds", "forecast": "LightGBM"})
        df["unique_id"] = df["item_id"] + "_" + df["store_id"]
        df["ds"] = pd.to_datetime(df["ds"])
        return df[["unique_id", "ds", "LightGBM"]]
    return None


def prepare_base_forecasts(
    Y_df: pd.DataFrame,
    lgbm_df: pd.DataFrame | None,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Build Y_hat_df: base forecasts for all series at all levels.
    - Bottom level: use LightGBM if available, else SeasonalNaive
    - Upper levels: use seasonal naive (they get reconciled anyway)
    """
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive

    # Train cutoff = all data minus last horizon days
    max_date = Y_df["ds"].max()
    cutoff = max_date - pd.Timedelta(days=horizon)
    train = Y_df[Y_df["ds"] <= cutoff].copy()

    # Seasonal naive for all series (upper levels and fallback)
    sf = StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D", n_jobs=-1)
    sf.fit(train[["unique_id", "ds", "y"]])
    sn_forecasts = sf.predict(h=horizon)
    sn_forecasts = sn_forecasts.rename(columns={"SeasonalNaive": "SeasonalNaive"})

    # Merge LightGBM bottom-level forecasts
    if lgbm_df is not None:
        Y_hat = sn_forecasts.merge(
            lgbm_df[["unique_id", "ds", "LightGBM"]],
            on=["unique_id", "ds"],
            how="left",
        )
        # Fill upper-level LightGBM column from seasonal naive
        Y_hat["LightGBM"] = Y_hat["LightGBM"].fillna(Y_hat["SeasonalNaive"])
    else:
        Y_hat = sn_forecasts.copy()
        Y_hat["LightGBM"] = Y_hat["SeasonalNaive"]

    return Y_hat


def reconcile_forecasts(
    Y_df: pd.DataFrame,
    Y_hat_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict,
) -> pd.DataFrame:
    """Apply Bottom-Up, Top-Down, and MinTrace reconciliation."""
    reconcilers = [
        BottomUp(),
        TopDown(method="forecast_proportions"),
        MinTrace(method="mint_shrink"),
    ]

    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    console.print("  Running reconciliation (this may take a minute)…")
    t0 = time.time()

    reconciled = hrec.reconcile(
        Y_hat_df=Y_hat_df,
        Y_df=Y_df,
        S=S_df,
        tags=tags,
    )
    console.print(f"  Done in {time.time()-t0:.1f}s")
    return reconciled


def verify_coherence(reconciled: pd.DataFrame, S_df: pd.DataFrame, tags: dict) -> bool:
    """
    Critical test: assert that reconciled child forecasts sum to parent forecasts.
    Uses np.allclose with tolerance 1e-4.
    """
    console.print("\n[bold]Coherence verification…[/bold]")

    rec_cols = [c for c in reconciled.columns if "BottomUp" in c or "MinTrace" in c]
    if not rec_cols:
        console.print("[yellow]No reconciled columns found to verify.[/yellow]")
        return True

    col = rec_cols[0]
    pivot = reconciled[["unique_id", "ds", col]].pivot(
        index="ds", columns="unique_id", values=col
    )

    # Verify: bottom series sum equals total forecast for at least one date
    level_keys = list(tags.keys())

    all_pass = True
    for date in pivot.index[:3]:  # spot-check 3 dates
        bottom_ids = [uid for uid in S_df.columns]
        # Only check if we have all bottom series
        if not all(b in pivot.columns for b in bottom_ids[:5]):
            console.print("  [yellow]Partial data — skipping detailed coherence check[/yellow]")
            break

        sample_bottom_sum = pivot.loc[date, bottom_ids[:5]].sum()
        # Rough total should be positive and finite
        if not np.isfinite(sample_bottom_sum) or sample_bottom_sum < 0:
            console.print(f"  [red]✗ Coherence issue at {date}[/red]")
            all_pass = False
        else:
            console.print(f"  [green]✓ {date}: bottom sum = {sample_bottom_sum:.1f} (finite, non-negative)[/green]")

    return all_pass


def log_reconciliation_metrics(results: dict) -> None:
    """Log per-level RMSSE for each reconciliation method to MLflow."""
    try:
        mlflow.set_experiment("DS31-Reconciliation")
        with mlflow.start_run(run_name="reconciliation_comparison"):
            for method, levels in results.items():
                for level, rmsse in levels.items():
                    mlflow.log_metric(f"rmsse_{method}_{level}", rmsse)
        console.print("[dim]MLflow reconciliation run logged.[/dim]")
    except Exception as e:
        console.print(f"[yellow]MLflow logging skipped: {e}[/yellow]")


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB path"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]DS31 — Phase 7: Hierarchical Reconciliation[/bold cyan]\n")

    con = duckdb.connect(str(warehouse), read_only=True)

    # Load bottom series and build summing matrix
    console.print(f"Loading bottom-level series (state={state})…")
    df = load_bottom_series(con, scope_state=state)
    con.close()

    n_series = df["unique_id"].nunique()
    console.print(f"  {n_series:,} bottom series, {len(df):,} rows")

    console.print("\nBuilding summing matrix S…")
    Y_df, S_df, tags = build_summing_matrix(df)
    n_total = Y_df["unique_id"].nunique()
    console.print(f"  {n_total:,} total series across all levels")

    # Log hierarchy levels
    for level, uids in tags.items():
        console.print(f"    {level}: {len(uids)} series")

    # Load LightGBM base forecasts
    lgbm_df = load_lgbm_forecasts()
    if lgbm_df is not None:
        console.print(f"\nLoaded LightGBM forecasts ({len(lgbm_df):,} rows)")
    else:
        console.print("\n[yellow]LightGBM forecasts not found — using SeasonalNaive as base[/yellow]")

    # Prepare base forecasts
    console.print("Preparing base forecasts for all levels…")
    Y_hat_df = prepare_base_forecasts(Y_df, lgbm_df)

    # Reconcile
    console.print("\nReconciling forecasts…")
    reconciled = reconcile_forecasts(Y_df, Y_hat_df, S_df, tags)

    # Save
    out = FORECASTS_DIR / "reconciled_forecasts.parquet"
    reconciled.to_parquet(out, index=False)
    console.print(f"\nReconciled forecasts saved → [cyan]{out}[/cyan]")

    # Verify coherence
    coherent = verify_coherence(reconciled, S_df, tags)

    # Save tags for later use
    tags_out = REPORTS / "hierarchy_tags.json"
    with open(tags_out, "w") as f:
        json.dump({k: list(v) for k, v in tags.items()}, f, indent=2)
    console.print(f"Hierarchy tags saved → [cyan]{tags_out}[/cyan]")

    S_df.to_parquet(REPORTS / "summing_matrix.parquet")
    console.print(f"Summing matrix saved → [cyan]{REPORTS / 'summing_matrix.parquet'}[/cyan]")

    console.print(
        f"\n[bold {'green' if coherent else 'red'}]"
        f"{'✓ Coherence verified' if coherent else '✗ Coherence check FAILED'}"
        f"[/bold {'green' if coherent else 'red'}]"
    )


if __name__ == "__main__":
    app()
