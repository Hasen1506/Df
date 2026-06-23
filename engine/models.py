"""
models.py — the forecasting portfolio.

Every model exposes the SAME signature so the backtester can treat them
interchangeably:

    forecast(y_train, h, period=12, exog_train=None, exog_future=None) -> np.ndarray

`y_train` is a float pd.Series indexed by month-start dates; the return is a
length-`h` array of point forecasts. Models that don't use exogenous drivers
simply ignore them.

Why hand-roll some of these instead of importing everything? Because the point
of the app is to make the mechanics legible. Croston / SBA / TSB are a dozen
lines each and you should be able to read exactly what they do. ETS and SARIMA
lean on statsmodels (battle-tested estimation). The ML model uses LightGBM if
installed, else falls back to scikit-learn's HistGradientBoostingRegressor so
the app always runs.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor          # preferred
    _HAS_LGBM = True
except Exception:                               # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_LGBM = False


# --------------------------------------------------------------------------- #
# Baselines  (these define the bar every "real" model must clear -> see FVA)
# --------------------------------------------------------------------------- #
def naive(y, h, period=12, **_):
    return np.repeat(y.iloc[-1], h)


def seasonal_naive(y, h, period=12, **_):
    """Last year's same month. This is the canonical FVA benchmark."""
    vals = y.values
    if len(vals) < period:
        return naive(y, h)
    last_season = vals[-period:]
    return np.array([last_season[i % period] for i in range(h)])


def moving_average(y, h, period=12, k=3, **_):
    return np.repeat(y.iloc[-k:].mean(), h)


# --------------------------------------------------------------------------- #
# Exponential smoothing / Holt-Winters
# --------------------------------------------------------------------------- #
def ets(y, h, period=12, **_):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = y.astype(float)
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
        try: y = y.asfreq("MS")
        except Exception: pass
    seasonal = "add" if len(y) >= 2 * period else None
    trend = "add"
    try:
        model = ExponentialSmoothing(
            y, trend=trend, seasonal=seasonal,
            seasonal_periods=period if seasonal else None,
            initialization_method="estimated").fit()
        fc = model.forecast(h)
        return np.clip(np.asarray(fc, dtype=float), 0, None)
    except Exception:
        return seasonal_naive(y, h, period)


# --------------------------------------------------------------------------- #
# (Auto-)SARIMA  — compact AIC grid search over a small candidate set
# --------------------------------------------------------------------------- #
_SARIMA_GRID = [
    ((1, 1, 1), (0, 1, 1)),
    ((0, 1, 1), (0, 1, 1)),
    ((1, 1, 0), (1, 1, 0)),
    ((2, 1, 0), (0, 1, 1)),
    ((1, 0, 0), (1, 1, 0)),
]


def auto_sarima(y, h, period=12, **_):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = y.astype(float)
    if isinstance(y.index, pd.DatetimeIndex) and y.index.freq is None:
        try: y = y.asfreq("MS")
        except Exception: pass
    if len(y) < period + 6:
        return ets(y, h, period)
    best_aic, best_fc = np.inf, None
    for order, sorder in _SARIMA_GRID:
        try:
            res = SARIMAX(y, order=order,
                          seasonal_order=(*sorder, period),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_fc = np.asarray(res.forecast(h), dtype=float)
        except Exception:
            continue
    if best_fc is None:
        return ets(y, h, period)
    return np.clip(best_fc, 0, None)


# --------------------------------------------------------------------------- #
# Intermittent-demand family: Croston / SBA / TSB
# --------------------------------------------------------------------------- #
def _croston_core(y, alpha=0.1):
    """Return smoothed (size_hat, interval_hat) at the end of the series."""
    y = np.asarray(y, dtype=float)
    nz_idx = np.flatnonzero(y > 0)
    if len(nz_idx) == 0:
        return 0.0, np.inf
    sizes = y[nz_idx]
    intervals = np.diff(np.concatenate(([nz_idx[0] + 1], nz_idx + 1)))
    z = sizes[0]
    p = intervals[0] if len(intervals) else 1.0
    for i in range(1, len(sizes)):
        z = alpha * sizes[i] + (1 - alpha) * z
        p = alpha * intervals[i] + (1 - alpha) * p
    return z, max(p, 1e-6)


def croston(y, h, period=12, alpha=0.1, **_):
    z, p = _croston_core(y, alpha)
    return np.repeat(z / p, h)


def sba(y, h, period=12, alpha=0.1, **_):
    """Syntetos-Boylan Approximation: bias-corrected Croston."""
    z, p = _croston_core(y, alpha)
    return np.repeat((1 - alpha / 2) * z / p, h)


def tsb(y, h, period=12, alpha=0.1, beta=0.05, **_):
    """Teunter-Syntetos-Babai: updates demand *probability* every period;
    handles obsolescence better than Croston."""
    y = np.asarray(y, dtype=float)
    p_hat = (y > 0).mean()                       # init demand probability
    z_hat = y[y > 0].mean() if (y > 0).any() else 0.0
    for v in y:
        occ = 1.0 if v > 0 else 0.0
        p_hat = beta * occ + (1 - beta) * p_hat
        if v > 0:
            z_hat = alpha * v + (1 - alpha) * z_hat
    return np.repeat(p_hat * z_hat, h)


# --------------------------------------------------------------------------- #
# Machine-learning model (gradient boosting on lag + calendar features)
# --------------------------------------------------------------------------- #
def _make_features(y: pd.Series, exog: pd.DataFrame | None = None):
    df = pd.DataFrame({"y": y.values}, index=y.index)
    for lag in (1, 2, 3, 6, 12):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["month"] = df.index.month
    df["trend"] = np.arange(len(df))
    if exog is not None:
        for c in exog.columns:
            df[c] = exog[c].values
    return df


def _new_regressor():
    if _HAS_LGBM:
        return LGBMRegressor(n_estimators=300, learning_rate=0.05,
                             num_leaves=15, min_child_samples=5,
                             subsample=0.9, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=5)


def ml_boost(y, h, period=12, exog_train=None, exog_future=None, **_):
    """
    Recursive multi-step gradient boosting. Trains on lag+calendar features,
    then rolls forward one month at a time, feeding its own predictions back as
    lags. This is the honest way to multi-step with lag features — and the
    backtester is the only valid judge of whether the extra machinery beats ETS.
    """
    y = y.astype(float)
    feat = _make_features(y, exog_train).dropna()
    if len(feat) < 12:                           # too little history for ML
        return ets(y, h, period)
    X_cols = [c for c in feat.columns if c != "y"]
    model = _new_regressor()
    model.fit(feat[X_cols], feat["y"])

    history = list(y.values)
    idx = list(y.index)
    preds = []
    for step in range(h):
        next_date = (pd.Timestamp(idx[-1]) + pd.offsets.MonthBegin(1))
        row = {}
        s = pd.Series(history, index=pd.DatetimeIndex(idx))
        for lag in (1, 2, 3, 6, 12):
            row[f"lag_{lag}"] = s.iloc[-lag] if len(s) >= lag else s.iloc[0]
        row["roll_mean_3"] = s.iloc[-3:].mean()
        row["roll_mean_6"] = s.iloc[-6:].mean()
        row["month"] = next_date.month
        row["trend"] = len(history)
        if exog_future is not None:
            for c in exog_future.columns:
                row[c] = exog_future.iloc[step][c]
        yhat = float(model.predict(pd.DataFrame([row])[X_cols])[0])
        yhat = max(yhat, 0.0)
        preds.append(yhat)
        history.append(yhat)
        idx.append(next_date)
    return np.array(preds)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
MODELS = {
    "Naive": naive,
    "SeasonalNaive": seasonal_naive,
    "MovingAvg(3)": moving_average,
    "ETS/Holt-Winters": ets,
    "AutoSARIMA": auto_sarima,
    "Croston": croston,
    "SBA": sba,
    "TSB": tsb,
    "ML(GBM)": ml_boost,
}

# A reasonable subset to backtest per pattern (full set is allowed too)
PATTERN_MODELS = {
    "Smooth": ["SeasonalNaive", "ETS/Holt-Winters", "AutoSARIMA", "ML(GBM)"],
    "Erratic": ["SeasonalNaive", "ETS/Holt-Winters", "ML(GBM)"],
    "Intermittent": ["SeasonalNaive", "Croston", "SBA", "TSB"],
    "Lumpy": ["SeasonalNaive", "Croston", "TSB"],
}

ML_BACKEND = "LightGBM" if _HAS_LGBM else "sklearn HistGradientBoosting"
