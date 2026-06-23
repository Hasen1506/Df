"""
pipeline.py — the end-to-end orchestration.

run_pipeline() executes the full demand-planning flow on a tidy demand frame:

    raw sales
      -> unconstrain stockouts          (cleansing.build_clean_demand)
      -> outlier-correct                (cleansing.build_clean_demand)
      -> segment ADI/CV2 + ABC/XYZ      (segmentation.segment)
      -> per-series rolling backtest     (backtest.backtest_series)
      -> pick best model (FVA-aware)     (backtest.select_model)
      -> refit on full history, forecast (models.MODELS)
      -> reconcile bottom-up             (reconcile.bottom_up)

It returns a single results object (dict) the API / UI / scripts all consume,
so there is exactly one source of truth for "what the engine did".
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import data as datamod
from . import cleansing, segmentation, reconcile
from .models import MODELS, ML_BACKEND
from .backtest import backtest_series, select_model, models_for


def _exog_frame(g: pd.DataFrame) -> pd.DataFrame | None:
    cols = [c for c in ("on_promo", "price") if c in g.columns
            and g[c].notna().any()]
    if not cols:
        return None
    ex = g[cols].copy().reset_index(drop=True)
    if "price" in ex:                            # scale price for the model
        ex["price"] = ex["price"] / max(ex["price"].mean(), 1.0)
    return ex


def run_pipeline(df: pd.DataFrame, horizon: int = 6, n_folds: int = 4,
                 period: int = 12, allow_all_models: bool = False,
                 outlier_k: float = 4.0) -> dict:
    df = datamod.validate(df)

    # 1. unconstrain stockout-censored demand (all series)
    df = cleansing.unconstrain_frame(df)

    # 2. segment on the unconstrained signal -> we need the pattern BEFORE we
    #    decide whether outlier-cleansing is even appropriate
    seg = segmentation.segment(df, value_col="demand_unconstrained")
    patterns = {(r.product_id, r.location): r.pattern for r in seg.itertuples()}

    # 3. outlier-correct only smooth/erratic series (skip intermittent/lumpy)
    df, change_log = cleansing.correct_outliers_frame(df, patterns,
                                                      outlier_k=outlier_k)
    # refresh segmentation values on the final clean signal (value/ABC may shift)
    seg = segmentation.segment(df, value_col="demand_clean")
    seg_lookup = patterns

    backtests, selections, leaf_fc, fitted_lines = {}, [], [], {}

    for (pid, loc), g in df.groupby(["product_id", "location"]):
        g = g.sort_values("date")
        y = pd.Series(g["demand_clean"].values,
                      index=pd.DatetimeIndex(g["date"])).astype(float)
        pattern = seg_lookup[(pid, loc)]
        names = models_for(pattern, allow_all=allow_all_models)
        exog = None if pattern in ("Intermittent", "Lumpy") else _exog_frame(g)

        bt = backtest_series(y, names, horizon=horizon, n_folds=n_folds,
                             period=period, exog=exog)
        best = select_model(bt, pattern)
        backtests[f"{pid} | {loc}"] = bt

        # refit best on full history, produce the live forecast
        ex_fu = None
        if exog is not None:                     # naive flat carry of drivers
            ex_fu = pd.concat([exog.iloc[[-1]]] * horizon, ignore_index=True)
        try:
            fc = MODELS[best](y, horizon, period=period,
                              exog_train=exog, exog_future=ex_fu)
        except Exception:
            fc = MODELS["SeasonalNaive"](y, horizon, period=period)
        fc = np.clip(np.asarray(fc, dtype=float), 0, None)

        future_dates = pd.date_range(
            y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        for step, (d, v) in enumerate(zip(future_dates, fc), start=1):
            leaf_fc.append(dict(product_id=pid, location=loc, date=d,
                                step=step, forecast=round(float(v), 1)))

        row = bt[bt["model"] == best]
        def _g(col):
            return float(row[col].iloc[0]) if not row.empty else np.nan
        selections.append(dict(
            product_id=pid, location=loc,
            product_name=g["product_name"].iloc[0],
            pattern=pattern, best_model=best,
            WMAPE=_g("WMAPE"), MAE=_g("MAE"), RMSE=_g("RMSE"),
            BIAS_pct=_g("BIAS_pct"), FVA_vs_SNaive=_g("FVA_vs_SNaive")))

        fitted_lines[f"{pid} | {loc}"] = dict(
            history_dates=[d.strftime("%Y-%m") for d in y.index],
            history=[round(float(v), 1) for v in y.values],
            raw_sales=[round(float(v), 1) for v in g["sales"].values],
            forecast_dates=[d.strftime("%Y-%m") for d in future_dates],
            forecast=[round(float(v), 1) for v in fc],
            best_model=best)

    leaf_df = pd.DataFrame(leaf_fc)
    hierarchy = reconcile.bottom_up(
        leaf_df.rename(columns={"forecast": "forecast"}))

    selections_df = pd.DataFrame(selections)
    return dict(
        clean_df=df,
        change_log=change_log,
        segmentation=seg,
        backtests=backtests,
        selections=selections_df,
        leaf_forecast=leaf_df,
        hierarchy=hierarchy,
        series_lines=fitted_lines,
        meta=dict(horizon=horizon, n_folds=n_folds, period=period,
                  ml_backend=ML_BACKEND,
                  n_series=df.groupby(['product_id', 'location']).ngroups,
                  stockout_months=int(df["stockout_flag"].sum()),
                  outliers_corrected=int(len(change_log))),
    )
