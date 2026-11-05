import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.data import ROOT, available_categories, weekly_sales


app = FastAPI(title="Northstar Sales Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_origin_regex=r"https://[a-zA-Z0-9-]+\.github\.io",
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_metrics() -> dict:
    path = ROOT / "results" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_forecasts() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "forecasts.csv", parse_dates=["ds"])


def require_category(category: str) -> dict:
    categories = load_metrics().get("categories", {})
    if category not in categories:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")
    return categories[category]


def as_number(value: object) -> float | None:
    return None if pd.isna(value) else float(value)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/categories")
def categories() -> dict[str, list[str]]:
    return {"categories": available_categories()}


@app.get("/api/metrics")
def metrics(category: str = "All") -> dict:
    scores = require_category(category)
    return {
        "category": category,
        "best_model": scores["best_model"],
        "models": {
            name: {"mae": values["mae"], "rmse": values["rmse"]}
            for name, values in scores.items()
            if name in {"Prophet", "XGBoost", "Naive"}
        },
        "test_weeks": load_metrics()["test_weeks"],
    }


@app.get("/api/forecast")
def forecast(
    category: str = "All",
    model: str = "best",
    horizon: int = Query(default=12, ge=4, le=26),
) -> dict:
    scores = require_category(category)
    model_key = model.strip().lower()
    if model_key == "best":
        resolved_model = scores["best_model"]
    elif model_key == "prophet":
        resolved_model = "Prophet"
    elif model_key == "xgboost":
        resolved_model = "XGBoost"
    else:
        raise HTTPException(status_code=422, detail="Model must be best, Prophet, or XGBoost")

    try:
        history = weekly_sales(category)
        forecasts = load_forecasts()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Forecast artifacts are not available") from error

    selected = forecasts.loc[
        (forecasts["Category"] == category) & (forecasts["Model"] == resolved_model)
    ].sort_values("ds").head(horizon)
    if selected.empty:
        raise HTTPException(status_code=503, detail="No forecast rows found for this selection")

    return {
        "category": category,
        "model": resolved_model,
        "horizon": horizon,
        "history": [
            {"ds": row.ds.strftime("%Y-%m-%d"), "y": float(row.y)}
            for row in history.itertuples(index=False)
        ],
        "forecast": [
            {
                "ds": row.ds.strftime("%Y-%m-%d"),
                "yhat": float(row.yhat),
                "yhat_lower": as_number(row.yhat_lower),
                "yhat_upper": as_number(row.yhat_upper),
            }
            for row in selected.itertuples(index=False)
        ],
    }