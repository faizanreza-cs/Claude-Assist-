"""Brand-level weekly TRx/NRx forecasting with honest backtesting.

Three model families, deliberately simple and explainable for a brand team:

* ``seasonal_naive`` – last year's same week scaled by recent YoY growth (baseline)
* ``ets``            – Holt-Winters exponential smoothing (additive trend + 52-week seasonality)
* ``gbm``            – gradient boosting on lag / calendar features, forecast recursively

``backtest`` holds out the last *h* weeks and reports WAPE / MAPE / bias for every model,
which is how a forecast would be signed off in a real S&OP or brand-planning cycle.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pharmapulse.db import query

MODELS = ("seasonal_naive", "ets", "gbm")
SEASON = 52


def weekly_series(brand: str, metric: str = "trx", region: str | None = None) -> pd.Series:
    """National (or regional) weekly series for one brand, zero-filled."""
    if metric not in {"trx", "nrx"}:
        raise ValueError("metric must be 'trx' or 'nrx'")
    sql = f"""
        SELECT r.week, SUM(r.{metric}) AS value
        FROM weekly_rx r JOIN hcps h USING (hcp_id)
        WHERE r.brand = ? {"AND h.region = ?" if region else ""}
        GROUP BY 1 ORDER BY 1
    """
    params: list[Any] = [brand] + ([region] if region else [])
    df = query(sql, params)
    all_weeks = query("SELECT DISTINCT week FROM weekly_rx ORDER BY 1")["week"]
    s = df.set_index("week")["value"].reindex(all_weeks, fill_value=0).astype(float)
    s.index = pd.DatetimeIndex(s.index, freq="W-MON")
    s.name = f"{brand}_{metric}"
    return s


# --------------------------------------------------------------------------- models


def _seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    # needs a full year of history *after* launch, otherwise YoY growth is meaningless
    if (y > 0).sum() < 2 * SEASON:
        return np.repeat(y.iloc[-4:].mean(), h)
    recent = y.iloc[-13:].sum()
    prior = y.iloc[-13 - SEASON : -SEASON].sum()
    growth = recent / prior if prior > 0 else 1.0
    base = y.iloc[-SEASON:].to_numpy()
    return np.array([base[i % SEASON] * growth for i in range(h)])


def _ets(y: pd.Series, h: int) -> np.ndarray:
    nonzero = y[y > 0]
    if len(nonzero) < 2 * SEASON + 4:
        # too short for a seasonal model (e.g. a recent launch): damped trend only
        model = ExponentialSmoothing(
            nonzero.to_numpy(), trend="add", damped_trend=True, seasonal=None
        )
    else:
        model = ExponentialSmoothing(
            y.to_numpy(), trend="add", damped_trend=True, seasonal="add", seasonal_periods=SEASON
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(optimized=True)
    return np.asarray(fit.forecast(h))


def _features(values: list[float], dates: pd.DatetimeIndex, i: int) -> list[float]:
    v = values
    return [
        v[i - 1],
        v[i - 2],
        v[i - 4],
        float(np.mean(v[i - 4 : i])),
        float(np.mean(v[i - 13 : i])),
        v[i - SEASON] if i >= SEASON else v[i - 1],
        float(dates[i].isocalendar().week),
        float(i),
    ]


def _gbm(y: pd.Series, h: int) -> np.ndarray:
    vals = y.to_list()
    idx = pd.date_range(y.index[0], periods=len(y) + h, freq="W-MON")
    start = 13
    X = [_features(vals, idx, i) for i in range(start, len(vals))]
    target = vals[start:]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(np.array(X), np.array(target))
    history = list(vals)
    preds = []
    for i in range(len(vals), len(vals) + h):
        p = float(model.predict(np.array([_features(history, idx, i)]))[0])
        p = max(p, 0.0)
        preds.append(p)
        history.append(p)
    return np.array(preds)


_FITTERS = {"seasonal_naive": _seasonal_naive, "ets": _ets, "gbm": _gbm}


# --------------------------------------------------------------------------- metrics


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom else float("nan")


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])))


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(actual.sum())
    return (float(pred.sum()) - denom) / denom if denom else float("nan")


# --------------------------------------------------------------------------- public API


@dataclass
class Forecast:
    brand: str
    metric: str
    model: str
    history: pd.Series
    forecast: pd.DataFrame  # columns: week, yhat, lower, upper

    def to_dict(self) -> dict[str, Any]:
        f = self.forecast
        return {
            "brand": self.brand,
            "metric": self.metric,
            "model": self.model,
            "horizon_weeks": len(f),
            "total_forecast": round(float(f["yhat"].sum()), 1),
            "last_13w_actual": round(float(self.history.iloc[-13:].sum()), 1),
            "weeks": [
                {
                    "week": w.date().isoformat(),
                    "yhat": round(float(y), 1),
                    "lower": round(float(lo), 1),
                    "upper": round(float(hi), 1),
                }
                for w, y, lo, hi in zip(f["week"], f["yhat"], f["lower"], f["upper"], strict=True)
            ],
        }


def backtest(
    brand: str, metric: str = "trx", horizon: int = 13, models: tuple[str, ...] = MODELS
) -> pd.DataFrame:
    """Hold out the last ``horizon`` weeks and score each model."""
    y = weekly_series(brand, metric)
    train, test = y.iloc[:-horizon], y.iloc[-horizon:].to_numpy()
    rows = []
    for m in models:
        pred = _FITTERS[m](train, horizon)
        rows.append(
            {
                "brand": brand,
                "metric": metric,
                "model": m,
                "wape": round(wape(test, pred), 4),
                "mape": round(mape(test, pred), 4),
                "bias": round(bias(test, pred), 4),
            }
        )
    return pd.DataFrame(rows)


def best_model(brand: str, metric: str = "trx", horizon: int = 13) -> str:
    bt = backtest(brand, metric, horizon)
    return str(bt.sort_values("wape").iloc[0]["model"])


def forecast(brand: str, horizon: int = 13, metric: str = "trx", model: str = "auto") -> Forecast:
    """Forecast ``horizon`` weeks ahead with an empirical prediction interval.

    The interval width comes from the chosen model's backtest residuals, widened with
    sqrt(step), which is easy to explain and well calibrated enough for planning.
    """
    if horizon < 1 or horizon > 52:
        raise ValueError("horizon must be between 1 and 52 weeks")
    if model == "auto":
        model = best_model(brand, metric, horizon=min(horizon, 13))
    if model not in _FITTERS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS} or 'auto'")
    y = weekly_series(brand, metric)
    hold = 13
    resid = y.iloc[-hold:].to_numpy() - _FITTERS[model](y.iloc[:-hold], hold)
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    yhat = np.maximum(_FITTERS[model](y, horizon), 0)
    steps = np.sqrt(np.arange(1, horizon + 1))
    weeks = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    fc = pd.DataFrame(
        {
            "week": weeks,
            "yhat": yhat,
            "lower": np.maximum(yhat - 1.28 * sigma * steps, 0),  # 80% interval
            "upper": yhat + 1.28 * sigma * steps,
        }
    )
    return Forecast(brand=brand, metric=metric, model=model, history=y, forecast=fc)
