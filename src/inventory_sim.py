"""
DS31 Phase 9: Inventory Cost Simulation
Compares total cost (stockout + holding) for baseline vs LightGBM vs reconciled.
All cost assumptions are stated explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from metrics import bottom_level

console = Console()
ROOT = Path(__file__).resolve().parent.parent
FORECASTS_DIR = ROOT / "data" / "forecasts"
REPORTS = ROOT / "reports"
HORIZON = 28

# ── Cost assumptions (stated explicitly) ────────────────────────────────────
# Stockout cost: lost margin per unit not sold (assumed 30% of average price)
# Holding cost: carrying cost per unit per day (assumed 0.5% of average price/day)
# Average unit price: $3.50 (approximate M5 Walmart grocery item)
AVG_PRICE = 3.50
STOCKOUT_COST_PER_UNIT = AVG_PRICE * 0.30   # = $1.05
HOLDING_COST_PER_UNIT_DAY = AVG_PRICE * 0.005  # = $0.0175/unit/day
SAFETY_STOCK_Z = 1.65  # 95th percentile coverage

app = typer.Typer()


def order_up_to_policy(
    forecast: np.ndarray,
    forecast_std: np.ndarray,
    actuals: np.ndarray,
    z: float = SAFETY_STOCK_Z,
) -> dict:
    """
    Order-up-to inventory policy.
    Order quantity = forecast + z * forecast_std (safety stock).
    Returns total stockout and holding cost over the horizon.

    Parameters
    ----------
    forecast     : point forecasts (units), length H
    forecast_std : forecast uncertainty (std dev per day), length H
    actuals      : actual demand, length H
    z            : safety stock multiplier (z-score for service level)
    """
    safety_stock = z * np.maximum(forecast_std, 0)
    order_qty = np.maximum(forecast + safety_stock, 0)

    stockout_units = np.maximum(actuals - order_qty, 0)
    overstock_units = np.maximum(order_qty - actuals, 0)

    stockout_cost = np.sum(stockout_units) * STOCKOUT_COST_PER_UNIT
    holding_cost = np.sum(overstock_units) * HOLDING_COST_PER_UNIT_DAY * HORIZON / 2

    return {
        "stockout_units": float(np.sum(stockout_units)),
        "overstock_units": float(np.sum(overstock_units)),
        "stockout_cost": float(stockout_cost),
        "holding_cost": float(holding_cost),
        "total_cost": float(stockout_cost + holding_cost),
        "fill_rate": float(
            1 - np.sum(stockout_units) / max(np.sum(actuals), 1)
        ),
    }


def load_actuals_and_forecasts() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load actuals and reconciled forecasts if available."""
    recon_path = FORECASTS_DIR / "reconciled_forecasts.parquet"
    lgbm_path = FORECASTS_DIR / "lgbm_forecasts.parquet"

    recon = bottom_level(pd.read_parquet(recon_path)) if recon_path.exists() else None
    lgbm = pd.read_parquet(lgbm_path) if lgbm_path.exists() else None
    return recon, lgbm


def simulate_inventory(
    actuals_agg: pd.Series,
    method_name: str,
    forecast_agg: pd.Series,
) -> dict:
    """Run inventory simulation for one forecast method."""
    y_true = actuals_agg.values
    y_pred = forecast_agg.values[: len(y_true)]

    # Estimate forecast std from residuals (in-sample proxy)
    resid_std = np.std(y_pred - y_true)
    forecast_std = np.full(len(y_pred), resid_std)

    result = order_up_to_policy(y_pred, forecast_std, y_true)
    result["method"] = method_name
    return result


def generate_synthetic_baseline(actuals: pd.Series) -> pd.Series:
    """SeasonalNaive (lag-7) baseline — roll last week's actuals."""
    vals = actuals.values
    return pd.Series(
        np.roll(vals, 7),
        index=actuals.index,
    )


@app.command()
def main(
    warehouse: Path = typer.Option(
        ROOT / "warehouse" / "m5.duckdb", help="DuckDB path"
    ),
) -> None:
    import duckdb

    REPORTS.mkdir(parents=True, exist_ok=True)
    console.print("[bold cyan]DS31 — Phase 9: Inventory Cost Simulation[/bold cyan]\n")

    # Load actuals (test period)
    con = duckdb.connect(str(warehouse), read_only=True)
    actuals_df = con.execute(
        """
        SELECT
            sale_date AS ds,
            SUM(units) AS y
        FROM raw_sales_enriched
        WHERE sale_date > (SELECT MAX(sale_date) - INTERVAL '28 days' FROM raw_sales_enriched)
          AND state_id = 'CA'
        GROUP BY ds
        ORDER BY ds
        """
    ).df()
    actuals_df["ds"] = pd.to_datetime(actuals_df["ds"])
    con.close()

    actuals_series = actuals_df.set_index("ds")["y"]

    recon, lgbm = load_actuals_and_forecasts()
    results = []

    # 1. Seasonal Naive baseline
    sn_forecast = generate_synthetic_baseline(actuals_series)
    results.append(simulate_inventory(actuals_series, "SeasonalNaive", sn_forecast))

    # 2. LightGBM
    if lgbm is not None:
        lgbm_agg = lgbm.groupby("sale_date")["forecast"].sum()
        lgbm_agg.index = pd.to_datetime(lgbm_agg.index)
        # align
        common_dates = actuals_series.index.intersection(lgbm_agg.index)
        if len(common_dates) > 0:
            results.append(
                simulate_inventory(
                    actuals_series.loc[common_dates],
                    "LightGBM",
                    lgbm_agg.loc[common_dates],
                )
            )

    # 3. Reconciled methods
    if recon is not None:
        recon_methods = [c for c in recon.columns if c not in ("unique_id", "ds", "y")]
        for method in recon_methods[:3]:  # limit to 3 methods
            method_agg = recon.groupby("ds")[method].sum()
            method_agg.index = pd.to_datetime(method_agg.index)
            common_dates = actuals_series.index.intersection(method_agg.index)
            if len(common_dates) > 0:
                results.append(
                    simulate_inventory(
                        actuals_series.loc[common_dates],
                        method,
                        method_agg.loc[common_dates],
                    )
                )

    # Display results
    tbl = Table(title="Inventory Cost Simulation (CA, 28-day test period)")
    tbl.add_column("Method", style="cyan")
    tbl.add_column("Stockout Units", style="red")
    tbl.add_column("Overstock Units", style="yellow")
    tbl.add_column("Stockout Cost ($)", style="red")
    tbl.add_column("Holding Cost ($)", style="yellow")
    tbl.add_column("Total Cost ($)", style="bold")
    tbl.add_column("Fill Rate", style="green")

    for r in results:
        tbl.add_row(
            r["method"],
            f"{r['stockout_units']:,.0f}",
            f"{r['overstock_units']:,.0f}",
            f"${r['stockout_cost']:,.2f}",
            f"${r['holding_cost']:,.2f}",
            f"${r['total_cost']:,.2f}",
            f"{r['fill_rate']:.1%}",
        )
    console.print(tbl)

    # Headline improvement vs seasonal naive
    if len(results) > 1:
        baseline_cost = results[0]["total_cost"]
        for r in results[1:]:
            if baseline_cost > 0:
                pct = (baseline_cost - r["total_cost"]) / baseline_cost * 100
                console.print(
                    f"  {r['method']}: total cost {'reduced' if pct > 0 else 'increased'} "
                    f"by [bold]{abs(pct):.1f}%[/bold] vs SeasonalNaive"
                )

    # Save
    sim_df = pd.DataFrame(results)
    sim_df.to_csv(REPORTS / "inventory_simulation.csv", index=False)
    console.print(f"\nSimulation saved → [cyan]{REPORTS / 'inventory_simulation.csv'}[/cyan]")

    console.print("\n[dim]Cost assumptions:[/dim]")
    console.print(f"  Avg unit price: ${AVG_PRICE}")
    console.print(f"  Stockout cost/unit: ${STOCKOUT_COST_PER_UNIT:.2f} (30% margin)")
    console.print(f"  Holding cost/unit/day: ${HOLDING_COST_PER_UNIT_DAY:.4f} (0.5%/day)")
    console.print(f"  Safety stock z-score: {SAFETY_STOCK_Z} (95% service level)")

    console.print("\n[bold green]✓ Inventory simulation complete.[/bold green]")


if __name__ == "__main__":
    app()
