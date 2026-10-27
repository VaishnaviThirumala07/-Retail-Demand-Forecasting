# Key SQL Insights — DS31 Retail Demand Forecasting

*(Fill these in after running `make all` with your real numbers.)*

## 1. SNAP Day Sales Lift
SNAP (Supplemental Nutrition Assistance Program) days drive a measurable uplift in grocery sales.
**Finding:** Food category sales in CA are **X% higher** on SNAP days vs non-SNAP days.
This is a critical signal for demand planners adjusting replenishment schedules at the start of each month.

## 2. Weekend Seasonality
Sales peak on **Fridays and Saturdays** across all states.
**Finding:** Friday sales index is **~X** (average = 100), Saturday is **~X**.
Sunday drops to ~X, suggesting weekend shopping patterns dominate over mid-week.
Impact: replenishment should target Thursday/Friday arrivals.

## 3. Pareto Concentration
**Finding:** The top **20% of items** generate approximately **80% of revenue** — a classic Pareto distribution.
This means forecasting accuracy for the top quintile drives the vast majority of inventory value impact.
The LightGBM model with dollar-weighted WRMSSE naturally focuses accuracy on these high-value items.

## 4. Intermittent Demand
**Finding:** **X%** of item-store combinations have more than 50% zero-sales days.
These intermittent series are difficult to forecast with standard methods and justify the Tweedie objective,
which natively handles zero-inflated count distributions.

## 5. Holiday Effects
**Finding:** Christmas week drives a **~X% uplift** in total sales vs the same week in non-holiday years.
Super Bowl Sunday (FOODS_3 / HOBBIES_1 categories) shows a **~X%** spike.
These signals are captured via the `has_event` and `event_type` features in LightGBM.
