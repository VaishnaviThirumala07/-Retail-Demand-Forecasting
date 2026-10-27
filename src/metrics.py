"""
DS31 Phase 8: Evaluation Metrics
WRMSSE, RMSSE per level, bias, MAE.
Includes a unit-tested WRMSSE implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"
HORIZON = 28


# ---------------------------------------------------------------------------
# RMSSE
# ---------------------------------------------------------------------------

def rmsse(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Scaled Error.
    Denominator: mean squared one-step naive error on training series.
    RMSSE = RMSE(y_true, y_pred) / sqrt(mean(diff(y_train)**2))
    """
    h = len(y_true)
    mse_pred = np.mean((y_pred[:h] - y_true[:h]) ** 2)
    naive_sq_errors = np.diff(y_train) ** 2
    denom = np.mean(naive_sq_errors[naive_sq_errors >= 0]) if len(naive_sq_errors) > 0 else 1.0
    denom = max(denom, 1e-8)
    return float(np.sqrt(mse_pred / denom))


# ---------------------------------------------------------------------------
# WRMSSE
# ---------------------------------------------------------------------------

def wrmsse(
    y_train_dict: dict[str, np.ndarray],
    y_true_dict: dict[str, np.ndarray],
    y_pred_dict: dict[str, np.ndarray],
    weights_dict: dict[str, float],
) -> float:
    """
    Weighted Root Mean Squared Scaled Error (M5 metric).
    WRMSSE = sum_i(w_i * RMSSE_i) where w_i are dollar-volume weights.

    Parameters
    ----------
    y_train_dict : {series_id: train_array}
    y_true_dict  : {series_id: test_array}
    y_pred_dict  : {series_id: forecast_array}
    weights_dict : {series_id: weight}  (should sum to 1 within each level)
    """
    total = 0.0
    total_weight = 0.0
    for sid in y_true_dict:
        if sid not in y_pred_dict or sid not in y_train_dict:
            continue
        w = weights_dict.get(sid, 1.0)
        r = rmsse(y_train_dict[sid], y_true_dict[sid], y_pred_dict[sid])
        total += w * r
        total_weight += w

    return total / total_weight if total_weight > 0 else float("nan")


def compute_series_weights(
    con: duckdb.DuckDBPyConnection,
    scope_state: str,
    horizon: int = HORIZON,
) -> dict[str, float]:
    """
    Compute dollar-volume weights for each bottom series.
    Weight = (revenue over last 28 training days) / (total revenue over same period).
    """
    state_filter = f"AND state_id = '{scope_state}'" if scope_state else ""
    df = con.execute(
        f"""
        WITH last28 AS (
            SELECT
                item_id || '_' || store_id AS series_key,
                SUM(units * COALESCE(sell_price, 0)) AS revenue
            FROM raw_sales_enriched
            WHERE sale_date > (
                SELECT MAX(sale_date) - INTERVAL '{horizon + horizon} days'
                FROM raw_sales_enriched
            )
            AND sale_date <= (
                SELECT MAX(sale_date) - INTERVAL '{horizon} days'
                FROM raw_sales_enriched
            )
            {state_filter}
            GROUP BY series_key
        )
        SELECT
            series_key,
            revenue,
            revenue / SUM(revenue) OVER () AS weight
        FROM last28
        """
    ).df()
    return dict(zip(df["series_key"], df["weight"]))


# ---------------------------------------------------------------------------
# Per-level evaluation
# ---------------------------------------------------------------------------

def evaluate_level(
    forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    train_data: pd.DataFrame,
    level_col: str,
    model_col: str,
) -> dict:
    """Compute RMSSE, MAE, Bias at a given aggregation level."""
    forecast_agg = (
        forecasts.groupby([level_col, "ds"])[model_col].sum().reset_index()
    )
    actuals_agg = actuals.groupby([level_col, "ds"])["y"].sum().reset_index()
    train_agg = train_data.groupby([level_col, "ds"])["y"].sum().reset_index()

    results = []
    for uid, grp_f in forecast_agg.groupby(level_col):
        grp_a = actuals_agg[actuals_agg[level_col] == uid].sort_values("ds")
        grp_t = train_agg[train_agg[level_col] == uid].sort_values("ds")
        grp_f = grp_f.sort_values("ds")

        y_true = grp_a["y"].values
        y_pred = grp_f[model_col].values[: len(y_true)]
        y_train = grp_t["y"].values

        if len(y_true) == 0:
            continue

        r = rmsse(y_train, y_true, y_pred)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        bias = float(np.mean(y_pred - y_true))

        results.append({"id": uid, "rmsse": r, "mae": mae, "bias": bias})

    if not results:
        return {}
    df = pd.DataFrame(results)
    return {
        "mean_rmsse": float(df["rmsse"].mean()),
        "mean_mae": float(df["mae"].mean()),
        "mean_bias": float(df["bias"].mean()),
        "n_series": len(df),
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

app = typer.Typer()


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    warehouse: Path = typer.Option(WAREHOUSE, help="DuckDB path"),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging"),
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]DS31 — Phase 8: Evaluation[/bold cyan]\n")

    con = duckdb.connect(str(warehouse), read_only=True)
    state_filter = f"AND state_id = '{state}'" if state else ""

    # Load actuals (test period = last 28 days)
    actuals_df = con.execute(
        f"""
        SELECT
            item_id || '_' || store_id AS series_key,
            state_id, store_id, cat_id, dept_id, item_id,
            sale_date AS ds,
            CAST(units AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE sale_date > (SELECT MAX(sale_date) - INTERVAL '28 days' FROM raw_sales_enriched)
        {state_filter}
        """
    ).df()
    actuals_df["ds"] = pd.to_datetime(actuals_df["ds"])

    # Load train data (for RMSSE denominator)
    train_df = con.execute(
        f"""
        SELECT
            item_id || '_' || store_id AS series_key,
            state_id, store_id, cat_id, dept_id, item_id,
            sale_date AS ds,
            CAST(units AS DOUBLE) AS y
        FROM raw_sales_enriched
        WHERE sale_date <= (SELECT MAX(sale_date) - INTERVAL '28 days' FROM raw_sales_enriched)
        {state_filter}
        """
    ).df()
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    con.close()

    # Load forecasts
    recon_path = FORECASTS_DIR / "reconciled_forecasts.parquet"
    lgbm_path = FORECASTS_DIR / "lgbm_forecasts.parquet"

    all_results = {}

    # Evaluate per level x method
    levels = {
        "total":    None,
        "state":    "state_id",
        "store":    "store_id",
        "category": "cat_id",
        "dept":     "dept_id",
    }

    if recon_path.exists():
        recon = pd.read_parquet(recon_path)
        recon_methods = [c for c in recon.columns if c not in ("unique_id", "ds", "y")]

        for method in recon_methods:
            all_results[method] = {}
            for level_name, level_col in levels.items():
                if level_col is None:
                    # Total: sum everything
                    y_true = actuals_df.groupby("ds")["y"].sum()
                    y_pred_df = recon.groupby("ds")[method].sum()
                    y_train = train_df.groupby("ds")["y"].sum()
                    r = rmsse(y_train.values, y_true.values, y_pred_df.values)
                    all_results[method][level_name] = {
                        "mean_rmsse": r, "n_series": 1
                    }
                else:
                    result = evaluate_level(
                        recon.merge(actuals_df[["series_key", level_col]].drop_duplicates(),
                                    left_on="unique_id", right_on="series_key", how="left"),
                        actuals_df.rename(columns={"series_key": "unique_id"}),
                        train_df.rename(columns={"series_key": "unique_id"}),
                        level_col,
                        method,
                    )
                    all_results[method][level_name] = result

    # Print results table
    tbl = Table(title="Per-Level RMSSE Comparison")
    tbl.add_column("Method", style="cyan")
    for lvl in levels:
        tbl.add_column(lvl.capitalize(), style="green")

    for method, lvl_results in all_results.items():
        row = [method]
        for lvl in levels:
            v = lvl_results.get(lvl, {}).get("mean_rmsse", float("nan"))
            row.append(f"{v:.4f}" if np.isfinite(v) else "N/A")
        tbl.add_row(*row)
    console.print(tbl)

    # Save results
    rows = []
    for method, lvl_results in all_results.items():
        for lvl, metrics in lvl_results.items():
            rows.append({"method": method, "level": lvl, **metrics})
    results_df = pd.DataFrame(rows)
    results_df.to_csv(REPORTS / "results.csv", index=False)
    console.print(f"\nResults saved → [cyan]{REPORTS / 'results.csv'}[/cyan]")

    # MLflow logging
    if not no_mlflow:
        try:
            mlflow.set_experiment("DS31-Evaluation")
            with mlflow.start_run(run_name="per_level_rmsse"):
                for method, lvl_results in all_results.items():
                    for lvl, metrics in lvl_results.items():
                        v = metrics.get("mean_rmsse", float("nan"))
                        if np.isfinite(v):
                            mlflow.log_metric(f"rmsse_{method}_{lvl}", v)
            console.print("[dim]MLflow evaluation run logged.[/dim]")
        except Exception as e:
            console.print(f"[yellow]MLflow logging skipped: {e}[/yellow]")

    console.print("\n[bold green]✓ Evaluation complete.[/bold green]")


if __name__ == "__main__":
    app()
