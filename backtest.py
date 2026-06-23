"""
backtest.py — prove the forecast is good BEFORE you trust it.

Core ideas this module enforces:

* Rolling-origin (walk-forward) evaluation. NEVER random k-fold on time series
  — that leaks the future into the past. We refit at successive cut-offs and
  score the next `horizon` months out-of-sample.

* Metrics that don't lie on low/zero volume: WMAPE (weighted MAPE) and MAE/RMSE
  alongside plain MAPE, plus signed BIAS and a TRACKING SIGNAL. Bias is the
  dangerous, fixable error — it is usually organisational (sandbagging /
  inflation), not statistical.

* Forecast Value Added (FVA): every candidate is scored against the seasonal-
  naive baseline. If a fancy model can't beat "same month last year", it is
  destroying value and should be dropped. This is the single most credible
  thing you can put in front of a steering committee.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .models import MODELS, PATTERN_MODELS


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mae(a, f):  return float(np.mean(np.abs(a - f)))
def rmse(a, f): return float(np.sqrt(np.mean((a - f) ** 2)))


def mape(a, f):
    mask = a != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def wmape(a, f):
    denom = np.sum(np.abs(a))
    return float(np.sum(np.abs(a - f)) / denom * 100) if denom else np.nan


def bias_pct(a, f):
    denom = np.sum(np.abs(a))
    return float(np.sum(f - a) / denom * 100) if denom else np.nan


def tracking_signal(a, f):
    """Cumulative error / mean absolute deviation. |TS| > ~4 => out of control
    (a persistent, directional miss)."""
    err = f - a
    mad = np.mean(np.abs(err))
    return float(np.sum(err) / mad) if mad else 0.0


# --------------------------------------------------------------------------- #
# Rolling-origin backtest for a single series
# --------------------------------------------------------------------------- #
def backtest_series(y: pd.Series, model_names: list[str],
                    horizon: int = 6, n_folds: int = 4, period: int = 12,
                    exog: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Walk-forward CV. For each fold we train on y[:cutoff] and score the next
    `horizon` actuals. Folds end at the series tail and step back by `horizon`.
    Returns one row per (model) with averaged metrics across folds + FVA vs
    SeasonalNaive.
    """
    y = y.astype(float)
    n = len(y)
    min_train = max(period + 6, n - n_folds * horizon)
    cutoffs = [c for c in range(min_train, n - horizon + 1, horizon)]
    if not cutoffs:
        cutoffs = [n - horizon]

    # collect predictions per model across all folds (concatenated)
    acc = {m: {"a": [], "f": []} for m in model_names + ["SeasonalNaive"]}
    for cut in cutoffs:
        y_tr, y_te = y.iloc[:cut], y.iloc[cut:cut + horizon].values
        ex_tr = exog.iloc[:cut] if exog is not None else None
        ex_fu = exog.iloc[cut:cut + horizon] if exog is not None else None
        for m in set(model_names + ["SeasonalNaive"]):
            try:
                f = MODELS[m](y_tr, len(y_te), period=period,
                              exog_train=ex_tr, exog_future=ex_fu)
            except Exception:
                f = np.repeat(y_tr.iloc[-1], len(y_te))
            acc[m]["a"].append(y_te)
            acc[m]["f"].append(np.asarray(f, dtype=float)[:len(y_te)])

    # seasonal-naive reference WMAPE for FVA
    sa = np.concatenate(acc["SeasonalNaive"]["a"])
    sf = np.concatenate(acc["SeasonalNaive"]["f"])
    base_wmape = wmape(sa, sf)

    rows = []
    for m in model_names:
        a = np.concatenate(acc[m]["a"])
        f = np.concatenate(acc[m]["f"])
        w = wmape(a, f)
        rows.append(dict(
            model=m,
            WMAPE=round(w, 2), MAPE=round(mape(a, f), 2),
            MAE=round(mae(a, f), 2), RMSE=round(rmse(a, f), 2),
            BIAS_pct=round(bias_pct(a, f), 2),
            TrackSignal=round(tracking_signal(a, f), 2),
            FVA_vs_SNaive=round(base_wmape - w, 2),   # +ve = adds value
        ))
    out = pd.DataFrame(rows).sort_values("WMAPE").reset_index(drop=True)
    return out


_BASELINES = {"Naive", "SeasonalNaive", "MovingAvg(3)"}


def select_model(bt: pd.DataFrame, pattern: str) -> str:
    """
    Pick the winner.

    For intermittent/lumpy series, plain MAE rewards forecasting ~zero (which is
    useless for stocking spares) and WMAPE is unstable, so we (a) drop the
    baselines from contention — Seasonal-Naive is a benchmark, not a deployable
    spare-parts model — and (b) rank the Croston family by RMSE, which properly
    penalises being persistently wrong about the occasional demand. For
    smooth/erratic series we rank by WMAPE. Falls back to Seasonal-Naive.
    """
    if bt.empty:
        return "SeasonalNaive"
    if pattern in ("Intermittent", "Lumpy"):
        cand = bt[~bt["model"].isin(_BASELINES)]
        if cand.empty:
            cand = bt
        return cand.sort_values("RMSE", na_position="last").iloc[0]["model"]
    return bt.sort_values("WMAPE", na_position="last").iloc[0]["model"]


def models_for(pattern: str, allow_all: bool = False) -> list[str]:
    if allow_all:
        return [m for m in MODELS if m != "Naive"]
    return PATTERN_MODELS.get(pattern, ["SeasonalNaive", "ETS/Holt-Winters"])
