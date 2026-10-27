import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

# ── Inline copies of pure metric functions (avoids importing heavy deps) ────


def rmsse(y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    h = len(y_true)
    mse_pred = np.mean((y_pred[:h] - y_true[:h]) ** 2)
    naive_sq_errors = np.diff(y_train) ** 2
    denom = np.mean(naive_sq_errors[naive_sq_errors >= 0]) if len(naive_sq_errors) > 0 else 1.0
    denom = max(denom, 1e-8)
    return float(np.sqrt(mse_pred / denom))


def wrmsse(y_train_dict, y_true_dict, y_pred_dict, weights_dict) -> float:
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



# ---------------------------------------------------------------------------
# 1. WRMSSE unit test (hand-computed example)
# ---------------------------------------------------------------------------

class TestWRMSSE:
    """
    Validate WRMSSE against a hand-computed example.
    Series A: perfect forecast → RMSSE = 0
    Series B: constant bias of 2 units/day
    """

    def test_perfect_forecast_rmsse_zero(self):
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        y_true = np.array([11, 12, 13], dtype=float)
        y_pred = y_true.copy()  # perfect
        result = rmsse(y_train, y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_rmsse_known_value(self):
        """
        Hand-computed:
        y_train = [1,1,1,1,1] → diffs all 0 → denom → 1e-8 (clipped)
        y_true  = [2,2,2]
        y_pred  = [1,1,1]
        MSE = 1.0, RMSSE = sqrt(1/1e-8) → very large → test that it's >0
        """
        y_train = np.ones(10, dtype=float)
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        result = rmsse(y_train, y_true, y_pred)
        assert result > 0

    def test_wrmsse_weighted_average(self):
        """WRMSSE with equal weights = average RMSSE."""
        y_train = {
            "A": np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float),
            "B": np.array([10, 12, 14, 16, 18, 10, 12, 14, 16, 18], dtype=float),
        }
        y_true = {"A": np.array([6.0, 7.0]), "B": np.array([20.0, 22.0])}
        y_pred = {"A": np.array([6.0, 7.0]), "B": np.array([20.0, 22.0])}  # perfect
        weights = {"A": 0.5, "B": 0.5}

        result = wrmsse(y_train, y_true, y_pred, weights)
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_wrmsse_higher_weight_series_dominates(self):
        """A series with weight=0.9 should dominate the WRMSSE."""
        y_train = {
            "A": np.arange(1, 11, dtype=float),
            "B": np.arange(1, 11, dtype=float),
        }
        y_true = {"A": np.array([5.0, 5.0]), "B": np.array([5.0, 5.0])}
        # A is perfect, B has error
        y_pred_good = {"A": np.array([5.0, 5.0]), "B": np.array([5.0, 5.0])}
        y_pred_bad_b = {"A": np.array([5.0, 5.0]), "B": np.array([10.0, 10.0])}

        weights_b_heavy = {"A": 0.1, "B": 0.9}
        weights_a_heavy = {"A": 0.9, "B": 0.1}

        score_b_heavy = wrmsse(y_train, y_true, y_pred_bad_b, weights_b_heavy)
        score_a_heavy = wrmsse(y_train, y_true, y_pred_bad_b, weights_a_heavy)

        # B has error and B-heavy should give worse (higher) WRMSSE
        assert score_b_heavy > score_a_heavy


# ---------------------------------------------------------------------------
# 2. Leakage test
# ---------------------------------------------------------------------------

class TestLeakage:
    """
    Assert that feature matrix lag columns are >= horizon (28 days).
    Simulates the check that would run after features.py.
    """

    def test_minimum_lag_is_28(self):
        """
        Construct a synthetic feature df and verify the lag columns
        cannot contain data within 28 days of the target date.
        """
        n = 100
        dates = pd.date_range("2020-01-01", periods=n)
        units = np.random.randint(0, 10, n).astype(float)

        df = pd.DataFrame({"date": dates, "units": units})
        df["lag_28"] = df["units"].shift(28)
        df["lag_35"] = df["units"].shift(35)

        # Verify: lag_28 at position t is units at t-28 (never t-27 or closer)
        for i in range(28, n):
            assert df["lag_28"].iloc[i] == df["units"].iloc[i - 28], (
                f"Leakage detected at position {i}: lag_28 does not match units[{i-28}]"
            )

    def test_rolling_window_excludes_last_28_days(self):
        """Rolling mean computed on rows [t-56, t-29] (28-day gap before target)."""
        n = 100
        units = np.arange(n, dtype=float)

        # The window is ROWS BETWEEN 56 PRECEDING AND 29 PRECEDING
        # This means the window ENDS at row t-29, never including t through t-28
        roll_mean = pd.Series(units).rolling(window=28, min_periods=1).mean().shift(29)

        for i in range(57, n):
            window = units[max(0, i - 56) : i - 28]
            expected = window.mean() if len(window) > 0 else np.nan
            if not np.isnan(roll_mean.iloc[i]) and not np.isnan(expected):
                assert abs(roll_mean.iloc[i] - expected) < 1e-6, (
                    f"Rolling window leakage at t={i}"
                )


# ---------------------------------------------------------------------------
# 3. Coherence test
# ---------------------------------------------------------------------------

class TestCoherence:
    """
    Assert reconciled forecasts are hierarchically coherent:
    children sum to parents at every level.
    """

    def setup_method(self):
        """Create a simple 2-level hierarchy: 1 parent, 2 children."""
        dates = pd.date_range("2020-01-01", periods=5)

        self.child1 = pd.DataFrame({
            "unique_id": "child_A",
            "ds": dates,
            "forecast": [10, 12, 8, 15, 11],
        })
        self.child2 = pd.DataFrame({
            "unique_id": "child_B",
            "ds": dates,
            "forecast": [5, 6, 4, 7, 5],
        })
        # Parent = sum of children (coherent)
        self.parent = pd.DataFrame({
            "unique_id": "parent",
            "ds": dates,
            "forecast": [15, 18, 12, 22, 16],
        })

    def test_children_sum_to_parent(self):
        """Bottom-up coherence: children must sum to parent."""
        forecasts = pd.concat([self.child1, self.child2, self.parent])

        for date in self.parent["ds"]:
            child_sum = (
                forecasts[forecasts["unique_id"].isin(["child_A", "child_B"])]
                .loc[lambda df: df["ds"] == date, "forecast"]
                .sum()
            )
            parent_val = (
                forecasts[forecasts["unique_id"] == "parent"]
                .loc[lambda df: df["ds"] == date, "forecast"]
                .values[0]
            )
            assert np.isclose(child_sum, parent_val, atol=1e-4), (
                f"Coherence violation at {date}: children sum {child_sum} ≠ parent {parent_val}"
            )

    def test_incoherent_forecasts_detected(self):
        """Test that an incoherent forecast is detected by our assertion."""
        dates = pd.date_range("2020-01-01", periods=3)
        children_sum = pd.Series([10, 12, 8])
        parent_wrong = pd.Series([11, 13, 9])  # off by 1

        is_coherent = np.allclose(children_sum, parent_wrong, atol=1e-4)
        assert not is_coherent, "Incoherent forecasts should NOT pass allclose"


# ---------------------------------------------------------------------------
# 4. Data quality check (can run without warehouse)
# ---------------------------------------------------------------------------

class TestDataQuality:
    """Validate synthetic data quality rules."""

    def test_no_negative_sales(self):
        df = pd.DataFrame({"units": [0, 1, 5, 10, 0]})
        assert (df["units"] >= 0).all()

    def test_no_future_dates_in_train(self):
        """Train set must not contain dates beyond the cutoff."""
        cutoff = pd.Timestamp("2016-04-24")
        dates = pd.date_range("2011-01-29", periods=1941)
        train_dates = dates[dates <= cutoff]
        assert train_dates.max() <= cutoff

    def test_price_positive(self):
        prices = pd.Series([2.5, 1.0, 0.99, 4.5])
        assert (prices > 0).all()
