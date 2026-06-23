"""
reconcile.py — make the numbers add up across the hierarchy.

You forecast at SKU x location, but the business plans at product and total
level too. Independently-made forecasts at different levels will NOT sum
correctly ("incoherent"). Reconciliation fixes that. We implement the two
methods planners actually use day to day:

  * Bottom-up: forecast the leaves, sum upward. Best when leaves are
    individually forecastable.
  * Top-down: forecast the aggregate (usually more stable), split down by
    historical proportions. Best when leaves are noisy/intermittent.

(MinT / optimal reconciliation exists and is stronger, but needs the full
covariance machinery; it's noted in the README as the advanced next step.)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def bottom_up(leaf_forecasts: pd.DataFrame) -> dict:
    """
    leaf_forecasts: columns [product_id, location, step, forecast].
    Returns coherent forecasts at three levels: leaf (unchanged), product
    (sum over locations), and total (sum over products).
    """
    leaf = leaf_forecasts.copy()
    product = (leaf.groupby(["product_id", "step"])["forecast"]
               .sum().reset_index())
    total = (leaf.groupby("step")["forecast"].sum().reset_index())
    total["level"] = "TOTAL"
    return {"leaf": leaf, "product": product, "total": total}


def top_down(total_forecast: pd.DataFrame, history: pd.DataFrame,
             value_col: str = "demand_clean") -> pd.DataFrame:
    """
    Split a total-level forecast down to leaves using each leaf's historical
    share of volume. total_forecast: columns [step, forecast]. history is the
    tidy demand frame. Returns leaf-level coherent forecasts.
    """
    shares = (history.groupby(["product_id", "location"])[value_col].sum())
    shares = (shares / shares.sum()).rename("share").reset_index()
    out = shares.assign(key=1).merge(
        total_forecast.assign(key=1), on="key").drop(columns="key")
    out["forecast"] = out["forecast"] * out["share"]
    return out[["product_id", "location", "step", "forecast"]]


def coherence_gap(independent_total: np.ndarray,
                  bottom_up_total: np.ndarray) -> float:
    """How far an independently-made top-level forecast sits from the sum of
    the leaves (mean absolute % gap). A live KPI of planning hygiene."""
    denom = np.where(independent_total == 0, np.nan, independent_total)
    return float(np.nanmean(np.abs(
        (bottom_up_total - independent_total) / denom)) * 100)
