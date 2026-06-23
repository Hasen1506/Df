"""
cleansing.py — turn raw sales history into a clean DEMAND signal.

Two distinct jobs that planners constantly conflate:

1. Unconstraining (demand de-censoring). When you were out of stock, recorded
   `sales` < true `demand`. Forecasting the censored series systematically
   under-forecasts your best movers. We recover latent demand on stockout
   months from the un-censored seasonal profile. This concept separates juniors
   from seniors.

2. Outlier cleansing — done SEASONALITY-AWARE. A one-off bulk order or a data
   error should be damped so it does not pollute the level estimate; a genuine
   seasonal peak or step-change must NOT be touched. Naive filters clip festive
   peaks because they look "high" versus neighbouring months. We therefore
   deseasonalise (monthly median index) and detrend (rolling median) FIRST,
   then run a robust Hampel test on the residual. Every correction is logged —
   silently editing history is how planners lose trust. Intermittent/lumpy
   series are skipped entirely (their spikes ARE the signal).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Unconstraining stockout-censored demand
# --------------------------------------------------------------------------- #
def unconstrain(sales: pd.Series, stockout_flag: pd.Series,
                period: int = 12) -> pd.Series:
    """Estimate latent demand on stockout months from the clean-month seasonal
    profile. Latent demand on a stockout month = max(observed sales, expected),
    so we never push a number *below* what we already sold."""
    s = sales.astype(float).copy()
    flag = stockout_flag.astype(int).reindex(s.index).fillna(0).astype(int)
    if flag.sum() == 0:
        return s

    clean = flag.values == 0
    if clean.sum() < period // 2:
        clean_mean = s[clean].mean() if clean.any() else s.mean()
        out = s.copy()
        out[flag.values == 1] = np.maximum(out[flag.values == 1], clean_mean)
        return out

    dfc = pd.DataFrame({"m": s.index.month[clean], "v": s.values[clean]})
    grand = dfc["v"].mean()
    seas_idx = (dfc.groupby("m")["v"].mean() / grand).reindex(range(1, 13)).fillna(1.0)
    out = s.copy()
    for i in range(len(s)):
        if flag.iloc[i] == 1:
            out.iloc[i] = max(s.iloc[i], grand * seas_idx.loc[s.index[i].month])
    return out


# --------------------------------------------------------------------------- #
# 2. Seasonality-aware outlier detection + correction (single series)
# --------------------------------------------------------------------------- #
def _deseasonalize(y: pd.Series):
    """Additive deseasonalisation by robust per-month median."""
    month_med = y.groupby(y.index.month).transform("median")
    return y - month_med + y.mean(), month_med


def detect_outliers(y: pd.Series, k: float = 4.0, window: int = 7) -> pd.Series:
    """
    Robust Hampel test on the deseasonalised + detrended residual, so seasonal
    peaks/troughs are NOT mistaken for anomalies. Returns a boolean mask.
    """
    deseas, _ = _deseasonalize(y.astype(float))
    med = deseas.rolling(window, center=True, min_periods=3).median()
    resid = deseas - med
    scale = 1.4826 * resid.abs().median()
    if not np.isfinite(scale) or scale == 0:
        scale = resid.std(ddof=1)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return resid.abs() > k * scale


def correct_outliers(y: pd.Series, k: float = 4.0, window: int = 7):
    """Replace flagged points with their seasonally-reconstructed local
    expectation. Returns (cleaned_series, log_dataframe)."""
    y = y.astype(float)
    mask = detect_outliers(y, k=k, window=window)
    deseas, month_med = _deseasonalize(y)
    local = deseas.rolling(window, center=True, min_periods=3).median()
    expected = (local - y.mean()) + month_med          # re-seasonalise
    cleaned = y.copy()
    cleaned[mask] = expected[mask].fillna(y.median())
    log = pd.DataFrame({"date": y.index[mask],
                        "original": y[mask].values,
                        "corrected": cleaned[mask].round(1).values})
    return cleaned, log


# --------------------------------------------------------------------------- #
# Frame-level helpers used by the pipeline
# --------------------------------------------------------------------------- #
def unconstrain_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["demand_unconstrained"] = np.nan
    for (_, _), g in df.groupby(["product_id", "location"]):
        y = pd.Series(g["sales"].values, index=pd.DatetimeIndex(g["date"]))
        flag = pd.Series(g["stockout_flag"].values, index=y.index)
        df.loc[g.index, "demand_unconstrained"] = unconstrain(y, flag).values
    return df


def correct_outliers_frame(df: pd.DataFrame, patterns: dict,
                           outlier_k: float = 4.0):
    """Apply outlier correction only to non-intermittent series. `patterns`
    maps (product_id, location) -> pattern string. Returns (df, change_log)."""
    df = df.copy()
    df["demand_clean"] = df["demand_unconstrained"]
    logs = []
    skip = {"Intermittent", "Lumpy"}
    for (pid, loc), g in df.groupby(["product_id", "location"]):
        if patterns.get((pid, loc)) in skip:
            continue
        y = pd.Series(g["demand_unconstrained"].values,
                      index=pd.DatetimeIndex(g["date"]))
        cleaned, log = correct_outliers(y, k=outlier_k)
        df.loc[g.index, "demand_clean"] = cleaned.values
        if not log.empty:
            log["product_id"], log["location"] = pid, loc
            logs.append(log)
    change_log = (pd.concat(logs, ignore_index=True) if logs else
                  pd.DataFrame(columns=["date", "original", "corrected",
                                        "product_id", "location"]))
    return df, change_log
