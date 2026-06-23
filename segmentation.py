"""
segmentation.py — decide WHICH SKUs deserve WHICH treatment.

You can't hand-tune 50,000 SKUs. Segmentation tells you, per series:
  * the demand pattern (Syntetos-Boylan ADI / CV^2 quadrant) -> which model
    class can even work, and which SKUs are effectively un-forecastable and
    should be handled with inventory buffering instead of chasing a forecast.
  * ABC (value/Pareto) -> where forecast error costs the most money.
  * XYZ (variability) -> how predictable each one is.
The ABC x XYZ grid is the everyday triage map of a demand planner.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# Syntetos-Boylan cut-offs (the standard reference values)
ADI_CUT = 1.32
CV2_CUT = 0.49


def _adi_cv2(values: np.ndarray):
    nz = values[values > 0]
    n = len(values)
    adi = n / len(nz) if len(nz) else np.inf      # avg interval between demands
    cv2 = (np.std(nz, ddof=1) / np.mean(nz)) ** 2 if len(nz) > 1 else 0.0
    return adi, cv2


def classify_pattern(adi: float, cv2: float) -> str:
    if adi < ADI_CUT and cv2 < CV2_CUT:
        return "Smooth"
    if adi >= ADI_CUT and cv2 < CV2_CUT:
        return "Intermittent"
    if adi < ADI_CUT and cv2 >= CV2_CUT:
        return "Erratic"
    return "Lumpy"


def segment(df: pd.DataFrame, value_col: str = "demand_clean") -> pd.DataFrame:
    """
    One row per (product_id, location) with pattern, ADI, CV^2, total value,
    ABC class, CV, and XYZ class. ABC is computed on revenue share if price is
    available, else on demand volume.
    """
    has_price = "price" in df.columns and df["price"].notna().any()
    recs = []
    for (pid, loc), g in df.groupby(["product_id", "location"]):
        v = g[value_col].fillna(0).values
        adi, cv2 = _adi_cv2(v)
        cv = (np.std(v, ddof=1) / np.mean(v)) if np.mean(v) > 0 else np.inf
        value = (g[value_col] * g["price"]).sum() if has_price \
            else g[value_col].sum()
        recs.append(dict(product_id=pid, location=loc,
                         product_name=g["product_name"].iloc[0],
                         pattern=classify_pattern(adi, cv2),
                         ADI=round(adi, 2), CV2=round(cv2, 2),
                         CV=round(float(cv), 2),
                         annual_value=round(float(value), 0)))
    seg = pd.DataFrame(recs).sort_values("annual_value", ascending=False)

    # ABC by cumulative value share
    seg["value_share"] = seg["annual_value"] / seg["annual_value"].sum()
    seg["cum_share"] = seg["value_share"].cumsum()
    seg["ABC"] = np.where(seg["cum_share"] <= 0.80, "A",
                          np.where(seg["cum_share"] <= 0.95, "B", "C"))
    # XYZ by coefficient of variation
    seg["XYZ"] = np.where(seg["CV"] <= 0.5, "X",
                          np.where(seg["CV"] <= 1.0, "Y", "Z"))
    seg["segment"] = seg["ABC"] + seg["XYZ"]
    return seg.reset_index(drop=True)


# Recommended model class per pattern — a sane default the engine can fall back
# on, and a teaching aid for why backtest sometimes "disagrees".
RECOMMENDED = {
    "Smooth": "ETS / SARIMA (trend+seasonality estimable)",
    "Erratic": "ETS + wide safety stock (level ok, noise high)",
    "Intermittent": "Croston / SBA / TSB (zero-inflated)",
    "Lumpy": "TSB or buffer-and-forget (largely un-forecastable)",
}
