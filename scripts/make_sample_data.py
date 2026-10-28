"""
Generate a small synthetic dataset with the same schema as the M5 CSVs.

The real M5 files are too large for git and Kaggle's licence forbids re-hosting
them, so CI runs the full pipeline on this sample instead.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N_TRAIN_DAYS = 1941
N_CALENDAR_DAYS = 1969
START_DATE = "2011-01-29"
STORES = ["CA_1", "CA_2", "TX_1"]
CATS = {"FOODS": 0.9, "HOBBIES": 0.3, "HOUSEHOLD": 0.5}
ITEMS_PER_DEPT = 5
EVENTS = [("SuperBowl", "Sporting"), ("Easter", "Cultural"), ("Thanksgiving", "National")]


def make_calendar(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(START_DATE, periods=N_CALENDAR_DAYS, freq="D")
    week_idx = np.arange(N_CALENDAR_DAYS) // 7
    cal = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "wm_yr_wk": 11101 + week_idx,
        "weekday": dates.day_name(),
        "wday": ((dates.dayofweek + 2) % 7) + 1,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i}" for i in range(1, N_CALENDAR_DAYS + 1)],
        "event_name_1": None,
        "event_type_1": None,
        "event_name_2": None,
        "event_type_2": None,
    })
    event_days = rng.choice(N_CALENDAR_DAYS, size=30, replace=False)
    for i, day in enumerate(event_days):
        name, etype = EVENTS[i % len(EVENTS)]
        cal.loc[day, ["event_name_1", "event_type_1"]] = [name, etype]
    snap = (dates.day <= 10).astype(int)
    cal["snap_CA"] = snap
    cal["snap_TX"] = snap
    cal["snap_WI"] = snap
    return cal


def make_sales_and_prices(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = np.array([1.3, 1.25, 0.9, 0.85, 0.85, 0.9, 1.0])
    dow = np.arange(N_TRAIN_DAYS) % 7
    trend = np.linspace(0.8, 1.2, N_TRAIN_DAYS)
    n_weeks = N_CALENDAR_DAYS // 7 + 1

    sales_rows, price_rows = [], []
    for cat, base_rate in CATS.items():
        dept = f"{cat}_1"
        for k in range(1, ITEMS_PER_DEPT + 1):
            item = f"{dept}_{k:03d}"
            item_rate = base_rate * rng.uniform(0.5, 3.0)
            base_price = round(rng.uniform(1.0, 10.0), 2)
            for store in STORES:
                rate = item_rate * rng.uniform(0.7, 1.3) * weekly[dow] * trend
                units = rng.poisson(rate)
                sales_rows.append(
                    [f"{item}_{store}_evaluation", item, dept, cat, store, store[:2], *units]
                )
                prices = base_price * np.where(rng.random(n_weeks) < 0.1, 0.8, 1.0)
                for w in range(n_weeks):
                    price_rows.append([store, item, 11101 + w, round(float(prices[w]), 2)])

    day_cols = [f"d_{i}" for i in range(1, N_TRAIN_DAYS + 1)]
    sales = pd.DataFrame(
        sales_rows, columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *day_cols]
    )
    prices = pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    return sales, prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    calendar = make_calendar(rng)
    sales, prices = make_sales_and_prices(rng)

    calendar.to_csv(args.out / "calendar.csv", index=False)
    sales.to_csv(args.out / "sales_train_evaluation.csv", index=False)
    prices.to_csv(args.out / "sell_prices.csv", index=False)
    print(f"Wrote {len(sales)} series x {N_TRAIN_DAYS} days to {args.out}")


if __name__ == "__main__":
    main()
