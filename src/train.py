import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from src.data import ROOT, available_categories, weekly_sales
from src.evaluate import score, seasonal_naive
from src.models import prophet_forecast, xgboost_forecast


TEST_WEEKS = 12
MAX_HORIZON = 26
PROPHET_CONFIGS = [
    {"changepoint_prior_scale": 0.03},
    {"changepoint_prior_scale": 0.15},
]
XGBOOST_CONFIGS = [
    {"n_estimators": 160, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 160, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9},
]


def _plot_run(
    history: pd.Series,
    actual: pd.Series,
    forecast: pd.DataFrame,
    category: str,
    model_name: str,
) -> Path:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(history.index, history.values, color="#176b5b", label="Train")
    axis.plot(actual.index, actual.values, color="#222a25", label="Actual")
    axis.plot(forecast["ds"], forecast["yhat"], color="#e46f4c", label="Forecast")
    if {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
        axis.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="#e46f4c",
            alpha=0.18,
            label="80% interval",
        )
    axis.set(title=f"{category} | {model_name} holdout forecast", ylabel="Weekly sales ($)")
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    output = Path(tempfile.gettempdir()) / f"forecast-{category.lower()}-{model_name.lower()}.png"
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def _log_run(
    category: str,
    model_name: str,
    config: dict[str, float | int],
    train: pd.Series,
    actual: pd.Series,
    forecast: pd.DataFrame,
    metrics: dict[str, float],
) -> str:
    with mlflow.start_run(run_name=f"{category}-{model_name}") as run:
        mlflow.log_param("category", category)
        mlflow.log_param("model_type", model_name)
        mlflow.log_params({key: str(value) for key, value in config.items()})
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("test_size", len(actual))
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(_plot_run(train, actual, forecast, category, model_name)))
        return run.info.run_id


def _save_exploration_figures() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    overall = weekly_sales("All").set_index("ds")["y"]

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(overall.index, overall.values, color="#176b5b", linewidth=1.8)
    axis.set(title="Superstore weekly sales", ylabel="Sales (USD)")
    figure.tight_layout()
    figure.savefig(reports_dir / "weekly-sales-trend.png", dpi=160)
    plt.close(figure)

    decomposition = seasonal_decompose(overall, model="additive", period=52, extrapolate_trend="freq")
    figure = decomposition.plot()
    figure.set_size_inches(12, 8)
    figure.tight_layout()
    figure.savefig(reports_dir / "seasonal-decomposition.png", dpi=160)
    plt.close(figure)


def run_training() -> tuple[dict, pd.DataFrame]:
    _save_exploration_figures()
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("sales-forecasting")
    metrics_by_category: dict[str, dict] = {}
    future_rows: list[pd.DataFrame] = []

    for category in available_categories():
        weekly = weekly_sales(category)
        series = weekly.set_index("ds")["y"].astype(float)
        if len(series) <= TEST_WEEKS + 52:
            raise ValueError(f"{category} has too few weeks for the seasonal lag and holdout")
        train = series.iloc[:-TEST_WEEKS]
        actual = series.iloc[-TEST_WEEKS:]
        category_results: dict[str, dict] = {}

        baseline = score(actual, seasonal_naive(train, actual.index))
        category_results["Naive"] = baseline

        candidate_runs: dict[str, list[dict]] = {"Prophet": [], "XGBoost": []}
        for config in PROPHET_CONFIGS:
            predicted = prophet_forecast(train.rename_axis("ds").reset_index(name="y"), actual.index, float(config["changepoint_prior_scale"]))
            metrics = score(actual, predicted.set_index("ds")["yhat"])
            run_id = _log_run(category, "Prophet", config, train, actual, predicted, metrics)
            candidate_runs["Prophet"].append({"config": config, "metrics": metrics, "run_id": run_id})

        for config in XGBOOST_CONFIGS:
            predicted = xgboost_forecast(train, len(actual), config)
            metrics = score(actual, predicted.set_index("ds")["yhat"])
            run_id = _log_run(category, "XGBoost", config, train, actual, predicted, metrics)
            candidate_runs["XGBoost"].append({"config": config, "metrics": metrics, "run_id": run_id})

        for model_name, candidates in candidate_runs.items():
            winner = min(candidates, key=lambda candidate: candidate["metrics"]["mae"])
            category_results[model_name] = {**winner["metrics"], "config": winner["config"], "run_id": winner["run_id"]}
            if model_name == "Prophet":
                full_forecast = prophet_forecast(
                    weekly,
                    pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=MAX_HORIZON, freq="W-SUN"),
                    float(winner["config"]["changepoint_prior_scale"]),
                )
            else:
                full_forecast = xgboost_forecast(series, MAX_HORIZON, winner["config"])
                full_forecast["yhat_lower"] = float("nan")
                full_forecast["yhat_upper"] = float("nan")
            full_forecast.insert(0, "Model", model_name)
            full_forecast.insert(0, "Category", category)
            future_rows.append(full_forecast)

        category_results["best_model"] = min(
            ("Prophet", "XGBoost"), key=lambda name: category_results[name]["mae"]
        )
        metrics_by_category[category] = category_results
        print(f"{category}: Prophet MAE={category_results['Prophet']['mae']:.2f}, "
              f"XGBoost MAE={category_results['XGBoost']['mae']:.2f}, "
              f"naive MAE={baseline['mae']:.2f}")

    forecasts = pd.concat(future_rows, ignore_index=True)
    results_dir = ROOT / "results"
    reports_dir = ROOT / "reports"
    results_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    results = {
        "source": "Kaggle Superstore Dataset (Tableau sample; educational use, see README)",
        "test_weeks": TEST_WEEKS,
        "max_horizon_weeks": MAX_HORIZON,
        "selection_note": "Best configurations are selected by holdout MAE; treat the comparison as exploratory.",
        "categories": metrics_by_category,
    }
    (results_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    forecasts.to_csv(results_dir / "forecasts.csv", index=False)
    return results, forecasts


if __name__ == "__main__":
    run_training()