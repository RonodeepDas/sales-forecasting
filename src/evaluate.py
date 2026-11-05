import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("Cannot evaluate forecasts without overlapping observations")
    return {
        "mae": float(mean_absolute_error(aligned["actual"], aligned["predicted"])),
        "rmse": float(np.sqrt(mean_squared_error(aligned["actual"], aligned["predicted"]))),
    }


def seasonal_naive(history: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    weekly = history.copy()
    weekly.index = pd.DatetimeIndex(weekly.index)
    values = weekly.reindex(dates - pd.Timedelta(weeks=52))
    fallback = float(weekly.iloc[-1])
    return pd.Series(values.fillna(fallback).to_numpy(), index=dates, name="yhat")