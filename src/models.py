from collections.abc import Mapping

import numpy as np
import pandas as pd


def build_features(series: pd.Series) -> pd.DataFrame:
    values = series.astype(float)
    features = pd.DataFrame(index=values.index)
    features["lag_1"] = values.shift(1)
    features["lag_2"] = values.shift(2)
    features["lag_4"] = values.shift(4)
    features["lag_52"] = values.shift(52)
    features["rolling_mean_4"] = values.shift(1).rolling(4).mean()
    features["rolling_mean_13"] = values.shift(1).rolling(13).mean()
    features["rolling_mean_52"] = values.shift(1).rolling(52).mean()
    features["week_of_year"] = values.index.isocalendar().week.astype(int).to_numpy()
    features["month"] = values.index.month
    return features.replace([np.inf, -np.inf], np.nan)


def next_feature_row(history: pd.Series) -> pd.DataFrame:
    next_date = history.index[-1] + pd.Timedelta(weeks=1)
    extended = history.reindex(history.index.append(pd.DatetimeIndex([next_date])))
    return build_features(extended).iloc[[-1]]


def prophet_forecast(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    changepoint_prior_scale: float,
) -> pd.DataFrame:
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(history[["ds", "y"]])
    predicted = model.predict(pd.DataFrame({"ds": future_dates}))
    return predicted[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)


def xgboost_forecast(
    history: pd.Series,
    horizon: int,
    config: Mapping[str, float | int],
) -> pd.DataFrame:
    from xgboost import XGBRegressor

    features = build_features(history)
    training = features.join(history.rename("target")).dropna()
    if training.empty:
        raise ValueError("XGBoost needs at least 53 weekly observations")

    model = XGBRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        subsample=float(config["subsample"]),
        colsample_bytree=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(training.drop(columns="target"), training["target"])

    extended = history.astype(float).copy()
    predictions: list[float] = []
    for _ in range(horizon):
        next_date = extended.index[-1] + pd.Timedelta(weeks=1)
        next_features = next_feature_row(extended)
        estimate = float(model.predict(next_features)[0])
        predictions.append(estimate)
        extended.loc[next_date] = estimate

    dates = pd.date_range(history.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN")
    return pd.DataFrame({"ds": dates, "yhat": predictions})