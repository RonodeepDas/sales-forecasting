import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import weekly_sales
from src.evaluate import score, seasonal_naive
from src.models import build_features, next_feature_row


class ForecastingTests(unittest.TestCase):
    def test_weekly_series_is_continuous_and_fills_gaps(self):
        source = pd.DataFrame(
            {"Order Date": ["01/01/2020", "01/01/2020", "01/20/2020"], "Sales": [10, 5, 7], "Category": ["A", "A", "A"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orders.csv"
            source.to_csv(path, index=False)
            weekly = weekly_sales("A", path)

        self.assertEqual(len(weekly), 4)
        self.assertEqual(weekly["y"].tolist(), [15, 0, 0, 7])
        self.assertTrue(weekly["ds"].diff().dropna().eq(pd.Timedelta(weeks=1)).all())

    def test_features_use_only_prior_observations(self):
        dates = pd.date_range("2020-01-05", periods=60, freq="W-SUN")
        series = pd.Series(range(60), index=dates, dtype=float)
        features = build_features(series)

        self.assertEqual(features.iloc[-1]["lag_1"], 58)
        self.assertEqual(features.iloc[-1]["rolling_mean_4"], 56.5)
        self.assertTrue(features.iloc[:52].isna().any(axis=None))

    def test_next_week_features_include_latest_observation(self):
        dates = pd.date_range("2020-01-05", periods=60, freq="W-SUN")
        series = pd.Series(range(60), index=dates, dtype=float)
        features = next_feature_row(series)

        self.assertEqual(features.iloc[0]["lag_1"], 59)
        self.assertEqual(features.iloc[0]["lag_52"], 8)
        self.assertEqual(features.iloc[0]["rolling_mean_4"], 57.5)

    def test_seasonal_naive_and_metrics_align_on_dates(self):
        dates = pd.date_range("2021-01-03", periods=60, freq="W-SUN")
        history = pd.Series(range(60), index=dates, dtype=float)
        future = pd.date_range(dates[-1] + pd.Timedelta(weeks=1), periods=2, freq="W-SUN")
        naive = seasonal_naive(history, future)

        self.assertEqual(naive.iloc[0], history.iloc[8])
        self.assertEqual(score(pd.Series([2, 4]), pd.Series([1, 5]))["mae"], 1.0)


if __name__ == "__main__":
    unittest.main()