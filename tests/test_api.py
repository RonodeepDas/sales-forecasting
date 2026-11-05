import unittest

from fastapi.testclient import TestClient

from backend.main import app


class ForecastApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_categories_and_metrics(self):
        categories = self.client.get("/api/categories")
        metrics = self.client.get("/api/metrics", params={"category": "All"})

        self.assertEqual(categories.status_code, 200)
        self.assertEqual(categories.json()["categories"], ["All", "Furniture", "Office Supplies", "Technology"])
        self.assertEqual(metrics.json()["best_model"], "XGBoost")

    def test_forecast_horizon_and_model_intervals(self):
        prophet = self.client.get("/api/forecast", params={"category": "All", "model": "Prophet", "horizon": 4})
        xgboost = self.client.get("/api/forecast", params={"category": "All", "model": "XGBoost", "horizon": 4})

        self.assertEqual(prophet.status_code, 200)
        self.assertEqual(len(prophet.json()["forecast"]), 4)
        self.assertIsNotNone(prophet.json()["forecast"][0]["yhat_lower"])
        self.assertEqual(xgboost.status_code, 200)
        self.assertIsNone(xgboost.json()["forecast"][0]["yhat_lower"])

    def test_invalid_inputs_are_rejected(self):
        invalid_category = self.client.get("/api/metrics", params={"category": "Unknown"})
        invalid_horizon = self.client.get("/api/forecast", params={"horizon": 30})

        self.assertEqual(invalid_category.status_code, 404)
        self.assertEqual(invalid_horizon.status_code, 422)


if __name__ == "__main__":
    unittest.main()