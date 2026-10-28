"""
DS31 Phase 6: LightGBM Global Forecasting Model
Tweedie objective, leak-free features, time-based CV.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import LabelEncoder

console = Console()
ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE = ROOT / "warehouse" / "m5.duckdb"
FEATURES_PARQUET = ROOT / "data" / "features" / "features.parquet"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"

HORIZON = 28

FEATURE_COLS = [
    "wday", "month", "year", "week_of_year",
    "snap_flag", "has_event",
    "lag_28", "lag_35", "lag_42", "lag_56",
    "roll_mean_28", "roll_mean_7", "roll_mean_14",
    "roll_std_28", "roll_max_28",
    "sell_price", "price_lag_7", "price_rel_mean",
    "price_change_7", "price_change_count",
    # Categoricals (label encoded)
    "item_id_enc", "dept_id_enc", "cat_id_enc",
    "store_id_enc", "state_id_enc", "event_type_enc",
]

CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type"]

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 512,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
}

app = typer.Typer()


def load_features(history_days: int) -> pd.DataFrame:
    """Load the last `history_days` of features with label-encoded categoricals."""
    console.print(f"Loading features from [cyan]{FEATURES_PARQUET}[/cyan]…")
    numeric_cols = [c for c in FEATURE_COLS if not c.endswith("_enc")]
    cols = ", ".join(
        ["series_key", "sale_date", "CAST(units AS FLOAT) AS units", *CAT_COLS]
        + [f"CAST({c} AS FLOAT) AS {c}" for c in numeric_cols]
    )
    src = f"read_parquet('{FEATURES_PARQUET.as_posix()}')"
    df = duckdb.sql(
        f"""
        SELECT {cols} FROM {src}
        WHERE sale_date > (SELECT MAX(sale_date) FROM {src}) - INTERVAL '{history_days} days'
        """
    ).df()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    console.print(f"  {len(df):,} rows (last {history_days} days), {df.shape[1]} columns")

    # Label encode categoricals
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def time_based_split(df: pd.DataFrame, n_folds: int = 2) -> list[tuple]:
    """
    Rolling-origin cross-validation.
    Each fold has a train set and a 28-day validation window.
    Never shuffles — always respects temporal order.
    """
    max_date = df["sale_date"].max()
    folds = []
    for i in range(n_folds, 0, -1):
        val_end = max_date - pd.Timedelta(days=(i - 1) * HORIZON)
        val_start = val_end - pd.Timedelta(days=HORIZON - 1)
        train_end = val_start - pd.Timedelta(days=1)

        train = df[df["sale_date"] <= train_end]
        val = df[(df["sale_date"] >= val_start) & (df["sale_date"] <= val_end)]
        folds.append((train, val))
    return folds


def train_model(
    df: pd.DataFrame,
    params: dict,
    folds: list[tuple],
) -> tuple[lgb.Booster, list[float]]:
    """Train with early stopping on the last validation fold."""
    val_scores = []

    # Use last fold for early stopping
    train_df, val_df = folds[-1]
    folds.clear()

    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train_df["units"].to_numpy(dtype=np.float32)
    X_val = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_val = val_df["units"].to_numpy(dtype=np.float32)
    del train_df

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=FEATURE_COLS)

    p = params.copy()
    n_est = p.pop("n_estimators", 2000)
    early_stop = p.pop("early_stopping_rounds", 50)

    callbacks = [
        lgb.early_stopping(early_stop, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    console.print(f"\nTraining LightGBM (n_estimators≤{n_est}, early_stop={early_stop})…")
    t0 = time.time()

    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=callbacks,
    )

    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_scores.append(val_rmse)

    console.print(
        f"  Best iteration: {model.best_iteration}  |  Val RMSE: {val_rmse:.4f}  |  "
        f"{time.time()-t0:.1f}s"
    )
    return model, val_scores


def generate_forecasts(
    model: lgb.Booster,
    df: pd.DataFrame,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Direct 28-day forecast:
    Since all lags ≥ 28, we predict all horizon steps simultaneously
    using features as of the last available date.
    """
    console.print(f"\nGenerating {horizon}-day forecasts…")

    # Get the most recent lag feature row for each series
    last_date = df["sale_date"].max()
    latest = df[df["sale_date"] == last_date].copy()

    preds_list = []
    for h in range(1, horizon + 1):
        row = latest.copy()
        # Adjust calendar features for forecast date
        forecast_date = last_date + pd.Timedelta(days=h)
        row["sale_date"] = forecast_date
        row["wday"] = forecast_date.weekday()
        row["month"] = forecast_date.month
        row["year"] = forecast_date.year
        row["week_of_year"] = forecast_date.isocalendar()[1]
        # Note: snap_flag and event flags would be joined from calendar in production

        X = row[FEATURE_COLS].values
        row["forecast"] = model.predict(X)
        row["horizon_day"] = h
        preds_list.append(row[["series_key", "item_id", "store_id", "state_id",
                                "sale_date", "forecast", "horizon_day"]])

    forecasts = pd.concat(preds_list, ignore_index=True)
    forecasts["forecast"] = forecasts["forecast"].clip(lower=0)
    console.print(f"  {len(forecasts):,} forecast rows")
    return forecasts


def log_to_mlflow(params: dict, metrics: dict, model: lgb.Booster) -> None:
    """Lightweight MLflow logging — per-level RMSSE and model artifact."""
    mlflow.set_experiment("DS31-LightGBM")
    with mlflow.start_run(run_name="lgbm_tweedie"):
        mlflow.log_params(
            {
                "objective": params["objective"],
                "tweedie_power": params["tweedie_variance_power"],
                "num_leaves": params["num_leaves"],
                "learning_rate": params["learning_rate"],
                "best_iteration": model.best_iteration,
            }
        )
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "lgbm_model.txt"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))
    console.print("[dim]MLflow run logged.[/dim]")


@app.command()
def main(
    state: str = typer.Option("CA", help="State scope"),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging"),
    history_days: int = typer.Option(365, help="Days of history to train on (limits memory)"),
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df, encoders = load_features(history_days)

    # Only the final fold is used (early stopping); building more just copies data.
    model, val_scores = train_model(df, LGBM_PARAMS, time_based_split(df, n_folds=1))

    # Feature importance
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    console.print("\n[bold]Top 10 features by gain:[/bold]")
    console.print(imp.head(10).to_string(index=False))

    # Save model and encoders
    model_path = MODELS_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    with open(MODELS_DIR / "label_encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    imp.to_csv(REPORTS / "feature_importance.csv", index=False)

    # Generate forecasts
    forecasts = generate_forecasts(model, df)
    out = ROOT / "data" / "forecasts" / "lgbm_forecasts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out, index=False)
    console.print(f"\nForecasts saved → [cyan]{out}[/cyan]")

    metrics = {"val_rmse_fold1": float(val_scores[0])}

    if not no_mlflow:
        try:
            log_to_mlflow(LGBM_PARAMS, metrics, model)
        except Exception as e:
            console.print(f"[yellow]MLflow logging skipped: {e}[/yellow]")

    console.print("\n[bold green]✓ LightGBM training complete.[/bold green]")


if __name__ == "__main__":
    app()
