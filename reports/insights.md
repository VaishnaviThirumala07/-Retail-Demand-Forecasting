# Key SQL Insights — DS31 Retail Demand Forecasting

Computed over the full DuckDB warehouse (all 3 states, 2011-01-29 → 2016-05-22, 59M item-store-days) by
`python sql/analysis/insights_summary.py`.

## 1. SNAP Day Sales Lift
SNAP (Supplemental Nutrition Assistance Program) days drive a measurable uplift in grocery sales.
**Finding:** CA FOODS sells **10.2% more** units on SNAP days: 10,733 units/day vs 9,736 on non-SNAP days.
Replenishment for food items should be front-loaded around the SNAP days at the start of each month.

## 2. Weekend Seasonality
Sales peak at the **weekend**, not on Friday.
**Finding:** Day-of-week sales index (average day = 100): Saturday **121**, Sunday **120**, Friday 100,
Monday 96, Tuesday 89, Thursday 88, Wednesday **87**.
Impact: stock should arrive by Friday to cover the Saturday/Sunday peak, which runs ~38% above the midweek low.

## 3. Revenue Concentration
**Finding:** The top **20% of items** generate **59% of revenue**. That's concentrated, but well short of a strict 80/20 Pareto split.
Accuracy on the top quintile still drives most of the inventory value at stake, which is the case for dollar-weighted
error metrics (WRMSSE).

## 4. Intermittent Demand
**Finding:** **78%** of the 30,490 item-store series have more than 50% zero-sales days.
(This includes days before an item launched, which M5 records as zeros.)
Such zero-heavy series justify the Tweedie objective, which models zero-inflated counts natively.

## 5. Holiday Effects
**Finding:** Stores are effectively **closed on Christmas Day** (0% of average daily sales), so that date needs special
handling in any model or replenishment plan. **Super Bowl Sunday** lifts FOODS_3 sales by **~6.9%** compared with
non-event Sundays within ±4 weeks.
These signals are captured via the `has_event` and `event_type` features in LightGBM.
