# DS31 — Retail Demand Forecasting with Hierarchical Time Series

> **Forecast 28-day demand across a retail hierarchy using LightGBM + MinTrace reconciliation.**  
> Built on 59M+ Walmart sales records from the M5 competition. Produces hierarchically coherent forecasts from state → store → category → department → SKU level.

[![CI](https://github.com/VaishnaviThirumala07/Retail-Demand-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/VaishnaviThirumala07/Retail-Demand-Forecasting/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-yellow)](https://duckdb.org)
[![dbt](https://img.shields.io/badge/dbt-1.7-orange)](https://docs.getdbt.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://docker.com)

---

## Problem Statement

Retail demand planning requires forecasts at multiple granularities simultaneously — from a national total to individual store-item (SKU) combinations. The challenge: **forecasts must be coherent** (children sum to parents) and **individually accurate** at each level. This project mirrors real planning workflows at large grocery retailers by implementing:

1. A full ELT pipeline over 59M+ daily sales records
2. A global LightGBM model with leak-free hierarchical features
3. Three reconciliation methods (Bottom-Up, Top-Down, MinTrace)
4. Business impact quantified as simulated inventory cost

---

## Architecture

```
Raw M5 CSVs
    │
    ▼
DuckDB (raw layer)          ← src/ingest.py
    │  UNPIVOT wide→long
    │  ~59M rows
    ▼
dbt (staging → marts)       ← dbt_m5/models/
    │  fct_daily_sales
    │  dim_item, dim_store
    ▼
SQL Analytics               ← sql/analysis/  (8 queries, 5 insights)
    │
    ▼
Feature Engineering         ← src/features.py
    │  Lags ≥ 28 days (no leakage)
    │  Rolling mean/std, price features, calendar
    ▼
Baselines                   ← src/baselines.py
    │  Naive, SeasonalNaive, AutoETS
    ▼
LightGBM (Tweedie)          ← src/train_lgbm.py
    │  Global model, time-based CV
    │  Direct 28-day horizon
    ▼
Hierarchical Reconciliation ← src/reconcile.py
    │  Sparse summing matrix S (CA: 12,196 bottom → 12,241 series)
    │  Bottom-Up | Top-Down | MinTrace (wls_struct)
    ▼
Evaluation                  ← src/metrics.py
    │  RMSSE per level, bias, MAE
    │  MLflow run comparison
    ▼
Cost Simulation             ← src/inventory_sim.py
    │  Order-up-to policy, stockout vs holding cost
    ▼
FastAPI + Streamlit         ← api/main.py, dashboard/app.py
    │  /forecast, /hierarchy, /health
    │  4-tab dashboard with drill-down
    ▼
Docker Compose              ← Dockerfile, docker-compose.yml
```

---

## Hierarchy Mapping

M5's 12 aggregation levels map to the brief's wording:

| Brief Level | M5 Level | Example |
|-------------|----------|---------|
| National | Total | All Walmart US |
| Regional | State | CA / TX / WI |
| Store | Store | CA_1, CA_2 |
| Category | Category | FOODS, HOBBIES |
| Department | Department | FOODS_1, FOODS_2 |
| SKU | Item × Store | FOODS_1_001_CA_1 |

**Scope:** This experiment uses California (CA). The reconciled hierarchy has 12,241 series:

| Level | Series |
|-------|--------|
| State (CA) | 1 |
| Store | 4 |
| Store × Category | 12 |
| Store × Department | 28 |
| Item × Store (SKU) | 12,196 |

All states (30,490 bottom series) would need more memory than a 16 GB laptop has to spare (see Limitations).

---

## Results

Real M5 data, California, test window = last 28 days (2016-04-25 → 2016-05-22), held out from training.
LightGBM trained on the preceding 180 days (memory limit on a 16 GB laptop), early-stopped on the 28 days before the test window
(best iteration 79, validation RMSE 2.186).

### RMSSE per Level (lower is better)

Mean RMSSE across the series at each level. State = Total here because only CA is in scope.

| Method | Total / State | Store (4) | Category (3) | Dept (7) |
|--------|------|-------|----------|------|
| SeasonalNaive (baseline) | 0.495 | 0.562 | 0.545 | 0.829 |
| AutoETS (fit per aggregate series) | **0.458** | **0.551** | **0.508** | 0.717 |
| LightGBM, bottom-up | 0.596 | 0.697 | 0.644 | 0.765 |
| LightGBM, Top-Down (forecast proportions) | 0.495 | 0.562 | 0.545 | 0.829 |
| **LightGBM, MinTrace (wls_struct)** | 0.497 | 0.571 | 0.537 | **0.709** |

**What this shows:**
- **LightGBM + MinTrace is the best method at department level**: 0.709 vs 0.829 for seasonal naive (**14.5% lower RMSSE**), and slightly ahead of AutoETS (0.717). It also beats seasonal naive at category level (0.537 vs 0.545, −1.5%).
- **At store and total level, AutoETS is best** and LightGBM + MinTrace roughly ties seasonal naive. MinTrace blends the LightGBM bottom level with seasonal-naive upper-level base forecasts, which pulls the aggregates back to seasonal-naive accuracy.
- **Summing LightGBM bottom-up is worst at aggregate levels.** It systematically under-forecasts: mean bias −1,165 units/day at total vs −815 for seasonal naive, because small per-SKU under-predictions add up.
- **Top-Down reproduces seasonal naive exactly**, because it splits the top-level base forecast, which is seasonal naive.

Full table with MAE and bias: [`reports/results.csv`](reports/results.csv). Baselines: [`reports/baseline_results.json`](reports/baseline_results.json).

**Top LightGBM features by gain:** `roll_mean_28` (dominant, ~7× the next), `roll_mean_14`, `roll_mean_7`, `item_id`, `price_rel_mean`, `roll_std_28`, `sell_price`, `price_lag_7`, `wday`, `price_change_count`.

### Inventory Cost Simulation

Order-up-to policy on CA total daily demand over the 28-day test window.

| Forecast | Stockout Cost | Holding Cost | Total Cost | Fill Rate |
|--------|-------------|-------------|-----------|-----------|
| **SeasonalNaive (statsforecast, also = Bottom-Up of it)** | $1,182 | $6,726 | **$7,908** | **99.8%** |
| LightGBM (bottom-up) | $5,160 | $5,853 | $11,012 | 99.1% |
| Lag-7 roll of test-window actuals (script's reference) | $2,482 | $18,410 | $20,892 | 99.5% |

> **Honest read:** seasonal naive gives the lowest simulated cost. LightGBM's under-forecast bias shows up directly as 4× the stockout cost, which outweighs its lower holding cost. The third row isn't a real forecast: it rolls the test window's own actuals by 7 days. The simulation currently scores only the first three forecast columns, so MinTrace isn't included yet.
>
> Cost assumptions: avg unit price $3.50, stockout = 30% margin ($1.05/unit), holding = 0.5%/unit/day, safety stock z = 1.65. These are assumed values, not real Walmart costs.

---

## Key SQL Insights

Computed over the full warehouse (all 3 states, 2011–2016) by [`sql/analysis/insights_summary.py`](sql/analysis/insights_summary.py).

| # | Finding |
|---|---------|
| 1 | CA FOODS sells **10.2% more** units on SNAP days (10,733 vs 9,736 units/day) |
| 2 | **Weekends dominate:** Saturday index **121**, Sunday **120**, Friday ~100 (avg = 100); Wednesday is lowest at 87 |
| 3 | The top **20% of items** generate **59% of revenue**: concentrated, but well short of a strict 80/20 |
| 4 | **78%** of the 30,490 item-store series have >50% zero-sales days (intermittent demand → Tweedie objective justified) |
| 5 | Stores are effectively closed on **Christmas Day** (0% of average sales); **Super Bowl Sunday** lifts FOODS_3 by **~6.9%** vs non-event Sundays within ±4 weeks |

Details: [`reports/insights.md`](reports/insights.md).

---

## How to Run

### Prerequisites
- Python 3.11+
- M5 data files in `data/raw/` (download from [Kaggle M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data))

```bash
# 1. Install dependencies
make setup

# 2. Run full pipeline
make all

# 3. Launch dashboard
make app
```

### Windows (no `make`) — memory-guarded runner
Runs each step as its own process. A step is killed if it uses more than `-MaxStepGB`, or if free RAM or commit memory
drops below the limits, so a heavy step can't freeze the machine. Temp files stay on the project's drive.

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -HistoryDays 180 -MaxStepGB 2 -MinFreeGB 1.5
```

Use `-Steps train,reconcile,evaluate,test` to run a subset. Per-step logs go to `logs\`.
These limits fit a 16 GB laptop with other apps open (highest step peak: 2.0 GB, during training).

### No data? Use the synthetic sample
```bash
make sample-data   # 45 series × 1,941 days with the M5 schema → data/raw/
make all
```
This is what CI runs on every push: the real M5 files are too large for git, and the competition rules don't allow re-hosting them.

### Required files in `data/raw/`
```
data/raw/
├── sales_train_evaluation.csv   # ~116 MB
├── calendar.csv                 # ~100 KB
└── sell_prices.csv              # ~194 MB
```

### Docker
```bash
docker compose up
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Step by step
```bash
make ingest      # Load CSVs → DuckDB (59M rows)
make dbt-run     # Build staging + mart models with tests
make features    # Engineer lag/rolling/price features
make baselines   # Compute naive + statsforecast baselines
make train       # Train LightGBM global model
make reconcile   # Run Bottom-Up, Top-Down, MinTrace
make evaluate    # Compute WRMSSE, per-level RMSSE, cost sim
make test        # Run leakage + coherence + WRMSSE unit tests
make app         # Launch Streamlit dashboard
```

---

## API Reference

```
GET /health                                     → system status
GET /hierarchy                                  → level structure
GET /forecast?level=store&id=CA_1&horizon=28    → 28-day forecast
GET /levels?level=store                         → available IDs
```

Example:
```bash
curl "http://localhost:8000/forecast?level=store&id=CA_1&horizon=7"
```

---

## Project Structure

```
ds31-retail-forecasting/
├── data/raw/                   # M5 CSVs (gitignored)
├── warehouse/m5.duckdb         # DuckDB warehouse (gitignored)
├── dbt_m5/
│   ├── models/staging/         # stg_sales, stg_calendar, stg_prices
│   ├── models/marts/           # fct_daily_sales, dim_item, dim_store
│   └── models/schema.yml       # data quality tests
├── sql/analysis/               # 8 numbered SQL insight queries
├── src/
│   ├── ingest.py               # Phase 1: DuckDB ingestion + UNPIVOT
│   ├── features.py             # Phase 4+6: lag/rolling/price features
│   ├── baselines.py            # Phase 5: Naive + statsforecast
│   ├── train_lgbm.py           # Phase 6: LightGBM global model
│   ├── reconcile.py            # Phase 7: hierarchical reconciliation
│   ├── metrics.py              # Phase 8: WRMSSE + per-level RMSSE
│   └── inventory_sim.py        # Phase 9: cost simulation
├── scripts/
│   ├── run_pipeline.ps1        # Windows runner with memory guard
│   └── make_sample_data.py     # synthetic M5-schema data (used by CI)
├── api/main.py                 # FastAPI service
├── dashboard/app.py            # Streamlit 4-tab dashboard
├── tests/test_ds31.py          # Leakage, coherence, WRMSSE tests
├── reports/
│   ├── results.csv             # method × level RMSSE table
│   ├── insights.md             # 5 headline SQL insights
│   └── inventory_simulation.csv
├── Dockerfile
├── docker-compose.yml
├── Makefile                    # make all / make app / make test
└── requirements.txt
```

---

## Layered Positioning (for Resume)

| Role | Layer visible in this project |
|------|-------------------------------|
| **Data Engineer** | DuckDB UNPIVOT pipeline, dbt lineage + quality tests, reproducible `make all` |
| **Data Scientist** | LightGBM Tweedie, MinTrace reconciliation, WRMSSE, rolling-origin CV, leakage test |
| **Data Analyst** | 8 SQL analyses, 5 headline insights, cost simulation, drill-down dashboard |

---

## Limitations

- **Scoped to CA:** All states (30,490 bottom series) exceed what a 16 GB laptop can spare. The CA subset (4 stores, 12,241 series) demonstrates every concept with verifiable results.
- **180-day training window:** LightGBM was trained on the last 180 days (the script default is 365) to stay within ~2 GB of RAM. A longer history may improve it.
- **LightGBM under-forecasts** at aggregate levels. A bias correction or a different Tweedie power is the obvious next step, and would likely fix the inventory-cost result.
- **MinTrace uses `wls_struct`**, not `mint_shrink`: shrinkage needs in-sample residuals and a dense covariance across 12k series, which is too much memory here.
- **Inventory simulation** currently evaluates only the first three forecast columns (MinTrace is not included), and its lag-7 reference row is built from test-window actuals.
- **Metric:** per-level *mean* RMSSE (unweighted). Dollar-weighted WRMSSE is implemented in `src/metrics.py` and unit-tested, but not yet used in the results table.
- **Static dataset:** M5 is a historical competition dataset — it doesn't receive new data. Weekly retraining would be simulated by advancing the training cutoff.
- **Assumed costs:** Inventory simulation uses assumed cost parameters, not actual Walmart P&L data. All assumptions are stated in `src/inventory_sim.py`.
- **Price coverage:** ~15% of item-days have no price data (new items or listing gaps). These are excluded from price features.

---

## Quality Checklist

- [x] `dbt build` passes all schema tests (unique, not_null, range, relationships)
- [x] Leakage test: minimum lag is 28 days (verified in `tests/test_ds31.py`)
- [x] Coherence test: children sum to parents at every level (`np.allclose`)
- [x] WRMSSE unit test: perfect forecast → RMSSE = 0
- [x] Results table shows improvement over seasonal-naive: **dept level −14.5% RMSSE** (LightGBM + MinTrace). Not at store/total level, and not in inventory cost
- [x] `make all` reproduces results end-to-end
- [x] CI (GitHub Actions) runs the full pipeline on synthetic data, the tests and a Docker smoke test on every push
- [ ] Dashboard live link *(deploy to Hugging Face Spaces or Render)*

---

## Resume Bullets

**Data Engineering:**
- Built an ELT pipeline in DuckDB and dbt over 59M+ daily Walmart sales records, with UNPIVOT-based wide-to-long transform, automated data quality tests, and a reproducible `make all` workflow.
- Containerized the full forecasting service (API + dashboard) in Docker with a one-command `docker compose up` deployment.

**Data Science:**
- Forecast 28-day demand for 12,196 California SKU-store series using a global LightGBM (Tweedie) model with leak-free lag/rolling/price features; with MinTrace reconciliation, cut department-level RMSSE by **14.5%** vs seasonal naive (0.709 vs 0.829).
- Applied Bottom-Up, Top-Down and MinTrace reconciliation over a 12,241-series hierarchy using sparse summing matrices; automated coherence verification with `np.allclose` assertions in the test suite.
- Quantified business impact with an order-up-to inventory simulation, showing that LightGBM's aggregate under-forecast bias raises stockout cost. This identified bias correction as the next modelling priority.

**Data Analysis:**
- Analyzed 5+ years of Walmart grocery sales in SQL, quantifying a **10.2%** SNAP-day lift in CA food sales, a weekend peak (Saturday/Sunday index ~120), and revenue concentration (top 20% of items = 59% of revenue).
- Built a 4-tab Streamlit dashboard with state → store → item drill-down, accuracy heatmaps, and interactive cost simulation sliders.
