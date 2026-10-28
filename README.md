# Retail Demand Forecasting with Hierarchical Time Series

[![CI](https://github.com/VaishnaviThirumala07/Retail-Demand-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/VaishnaviThirumala07/Retail-Demand-Forecasting/actions/workflows/ci.yml)

28-day demand forecasts for Walmart's M5 sales data, with forecasts made coherent across the
store → category → department → SKU hierarchy (each level sums exactly to the level above).

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| Ingest | `src/ingest.py` | Loads the M5 CSVs into DuckDB and unpivots them to 59,181,090 item-store-day rows |
| Model | `dbt_m5/` | dbt staging + mart models with 23 data tests |
| Features | `src/features.py` | Lags ≥ 28 days, rolling stats, price and calendar features (California scope) |
| Baselines | `src/baselines.py` | Naive, SeasonalNaive, HistoricAverage, AutoETS per aggregate level |
| Model | `src/train_lgbm.py` | One global LightGBM model (Tweedie loss) over all SKU-store series |
| Reconcile | `src/reconcile.py` | Bottom-Up, Top-Down, MinTrace (`wls_struct`) over 12,241 series |
| Evaluate | `src/metrics.py`, `src/inventory_sim.py` | RMSSE per level, plus an inventory cost simulation |

## Results (real M5 data, California)

- **Test window:** the last 28 days (2016-04-25 → 2016-05-22), never seen during training.
- **Training:** LightGBM used the 180 days before that, with early stopping on the final 28 of them (best iteration 79).
- **Hierarchy:** 1 state, 4 stores, 12 store × category series, 28 store × department series and 12,196 SKU-store series.

**RMSSE by level** (lower is better; mean over the series at each level):

| Method | Total | Store (4) | Category (3) | Dept (7) |
|--------|-------|-----------|--------------|----------|
| SeasonalNaive | 0.495 | 0.562 | 0.545 | 0.829 |
| AutoETS | **0.458** | **0.551** | **0.508** | 0.717 |
| LightGBM, bottom-up | 0.596 | 0.697 | 0.644 | 0.765 |
| LightGBM, Top-Down | 0.495 | 0.562 | 0.545 | 0.829 |
| LightGBM, MinTrace | 0.497 | 0.571 | 0.537 | **0.709** |

- **Department level:** LightGBM + MinTrace is the most accurate method: **14.5% lower RMSSE** than seasonal naive (0.709 vs 0.829).
- **Store and total level:** AutoETS is the most accurate method, and LightGBM + MinTrace roughly matches seasonal naive.
- **LightGBM bottom-up under-forecasts in aggregate:** its mean bias is −1,165 units/day at total level, vs −815 for seasonal naive.
- **Top-Down equals seasonal naive**, because it splits the top-level forecast, which is seasonal naive.

**Inventory cost simulation** (order-up-to policy on CA daily totals over the test window; assumed costs:
$3.50 unit price, $1.05 stockout cost per unit, 0.5% per unit per day holding cost):

| Forecast | Stockout cost | Holding cost | Total cost | Fill rate |
|----------|---------------|--------------|------------|-----------|
| SeasonalNaive | $1,182 | $6,726 | **$7,908** | 99.8% |
| LightGBM, bottom-up | $5,160 | $5,853 | $11,012 | 99.1% |

LightGBM's under-forecasting leads to about 4× more stockout cost, so seasonal naive is cheaper here. MinTrace isn't included in the simulation.

Source files: [`reports/results.csv`](reports/results.csv), [`reports/baseline_results.json`](reports/baseline_results.json),
[`reports/inventory_simulation.csv`](reports/inventory_simulation.csv).

## Data insights

From [`sql/analysis/insights_summary.py`](sql/analysis/insights_summary.py), run on all 3 states over 2011–2016:

| Finding | Value |
|---------|-------|
| CA food sales on SNAP days vs other days | **+10.2%** |
| Busiest days (index, average day = 100) | Saturday **121**, Sunday **120**; Wednesday lowest at **87** |
| Revenue share of the top 20% of items | **59%** |
| Item-store series with more than 50% zero-sales days | **78%** |
| Super Bowl Sunday, FOODS_3 vs nearby Sundays | **+6.9%** |
| Christmas Day sales | **0%** (stores closed) |

## Checks

- The dbt build passes all 23 data tests on the real data.
- Reconciled forecasts are coherent for every method: store and SKU forecasts sum exactly to the total.
- 11 unit tests (`tests/`) cover the metric maths, the lag/rolling-window logic and coherence checks on toy data.
- CI runs on every push: the full pipeline on synthetic M5-format data, the tests, and a Docker build with an API `/health` check.

## How to run

Put the [M5 files](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) in `data/raw/`:
`sales_train_evaluation.csv` (116 MB), `calendar.csv`, `sell_prices.csv` (194 MB).

```bash
make setup
make all
make test
```

**On Windows, without `make`:** `scripts/run_pipeline.ps1` runs each step as a separate process and stops a step
if it exceeds its memory limit. The settings below fit a 16 GB laptop; the largest step peaked at 2.0 GB.

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -HistoryDays 180 -MaxStepGB 2 -MinFreeGB 1.5
```

**Without the M5 files:** `make sample-data` writes a small synthetic dataset in the same format. This is what CI uses.

## Limitations

- Only California is modelled. Running all states needs more memory than a 16 GB laptop has.
- LightGBM was trained on 180 days of history to fit in memory, and it under-forecasts at aggregate levels.
- The results come from a single 28-day test window.
- MinTrace uses `wls_struct`. `mint_shrink` needs more memory than is available here.
- Inventory costs are assumptions, not real Walmart figures.
- 20.8% of item-days have no sell price in the data.
- `api/main.py` only serves `/health`; its `/forecast` endpoint is a placeholder. The Streamlit dashboard (`dashboard/app.py`) and `docker compose` haven't been tested with the current outputs.
