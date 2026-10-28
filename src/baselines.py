"""
DS31 Phase 5: Baseline Forecasts
Naive, seasonal-naive, and statsforecast AutoETS/AutoARIMA at aggregate levels.
All baselines are evaluated on the last 28 days (test set).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    HistoricAverage,
    Naive,
    SeasonalNaive,
)

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
REPORTS = ROOT / "reports"
MODELS_DIR = ROOT / "models"

HORIZON = 28
FREQ = "D"
SEASON_LENGTH = 7

app = typer.Typer()


def load_aggregated_series(
    con: duckdb.DuckDBPyConnection,
    level: str,
    scope_state: str = "CA",
) -> pd.DataFrame:
    """
    Load time series aggregated at a given hierarchy level.
    level: 'total' | 'state' | 'store' | 'category' | 'dept' | 'item_store'
    """
    group_cols = {
        "total":      "1 AS unique_id",
        "state":      "state_id AS unique_id",
        "store":      "store_id AS unique_id",
        "category":   "cat_id   AS unique_id",
        "dept":       "dept_id  AS unique_id",
        "item_store": "item_id || '_' || store_id AS unique_id",
    }

    if level not in group_cols:
        raise ValueError(f"Unknown level: {level}")

    gid = group_cols[level]
    state_filter = f"AND state_id = '{scope_state}'" if scope_state else ""

    df = con.execute(
        f"""
        SELECT
            {gid},
            sale_date AS ds,
            SUM(units) AS y
        FROM raw_sales_enriched
        WHERE 1=1 {state_filter}
        GROUP BY unique_id, ds
        ORDER BY unique_id, ds
        """
    ).df()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.rename(columns={"ds": "ds"})
    return df


def split_train_test(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based train/test split — last `horizon` days as test."""
    cutoff = df["ds"].max() - pd.Timedelta(days=horizon)
    train = df[df["ds"] <= cutoff].copy()
    test = df[df["ds"] > cutoff].copy()
    return train, test


def run_baselines(train: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Fit Naive, SeasonalNaive, HistoricAverage, AutoETS on train data."""
    models = [
        Naive(),
        SeasonalNaive(season_length=SEASON_LENGTH),
        HistoricAverage(),
        AutoETS(season_length=SEASON_LENGTH),
    ]

    sf = StatsForecast(
        models=models,
        freq=FREQ,
        n_jobs=1,
        verbose=False,
    )

    sf.fit(train)
    forecasts = sf.predict(h=horizon)
    return forecasts


def compute_rmsse(train: pd.DataFrame, test: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    RMSSE per series per model.
    RMSSE = RMSE / sqrt(mean(squared one-step naive errors on train))
    """
    # Naive benchmark error (denominator)
    naive_errors = {}
    for uid, g in train.groupby("unique_id"):
        y = g.sort_values("ds")["y"].values
        diffs = np.diff(y)
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) > 0:
            naive_errors[uid] = np.mean(diffs**2)
        else:
            naive_errors[uid] = 1.0

    results = []
    model_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds")]

    for uid, grp_f in forecasts.groupby("unique_id"):
        grp_t = test[test["unique_id"] == uid].sort_values("ds")
        grp_f = grp_f.sort_values("ds")
        denom = max(naive_errors.get(uid, 1.0), 1e-8)

        for model in model_cols:
            y_hat = grp_f[model].values
            y_true = grp_t["y"].values[: len(y_hat)]
            if len(y_true) == 0:
                continue
            mse = np.mean((y_hat - y_true) ** 2)
            rmsse = np.sqrt(mse / denom)
            results.append({"unique_id": uid, "model": model, "rmsse": rmsse})

    return pd.DataFrame(results)


def run_all_baselines(con: duckdb.DuckDBPyConnection, scope_state: str = "CA") -> dict:
    """Run baselines at each hierarchy level and collect RMSSE."""
    levels = ["total", "state", "store", "category", "dept"]
    all_results = {}

    for level in levels:
        console.print(f"\nLevel: [cyan]{level}[/cyan]")
        df = load_aggregated_series(con, level, scope_state)
        n_series = df["unique_id"].nunique()
        console.print(f"  {n_series} series, {len(df):,} rows")

        train, test = split_train_test(df)
        t0 = time.time()
        forecasts = run_baselines(train)
        elapsed = time.time() - t0

        rmsse_df = compute_rmsse(train, test, forecasts)
        summary = rmsse_df.groupby("model")["rmsse"].mean().to_dict()
        all_results[level] = summary

        tbl = Table(title=f"RMSSE @ {level}")
        tbl.add_column("Model")
        tbl.add_column("Mean RMSSE", style="green")
        for m, v in sorted(summary.items(), key=lambda x: x[1]):
            tbl.add_row(m, f"{v:.4f}")
        console.print(tbl)
        console.print(f"  Computed in {elapsed:.1f}s")

    return all_results


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB path"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(warehouse), read_only=True)
    results = run_all_baselines(con, scope_state=state)
    con.close()

    out = REPORTS / "baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[bold green]✓ Baselines saved → {out}[/bold green]")


if __name__ == "__main__":
    app()
