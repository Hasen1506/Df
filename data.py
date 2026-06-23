"""
data.py — data generation, ingestion and validation.

The engine works on a tidy ("long") monthly demand table with one row per
(date, product_id, location). Real-world history is almost never clean true
demand: it is censored shipments, polluted by promos, outliers and stockouts.
The synthetic generator below deliberately bakes in those pathologies so every
stage of the pipeline has something real to bite on.

Required columns for your own CSV:
    date        -> month start (YYYY-MM-01)
    product_id  -> SKU code
    location    -> shipping/selling location
    sales       -> observed quantity (what you actually shipped/invoiced)
Optional but used if present:
    price, on_promo (0/1), stockout_flag (0/1), product_name
"""

from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED_COLS = ["date", "product_id", "location", "sales"]


def _seasonal_profile(amplitude: float, peak_month: int) -> np.ndarray:
    """A 12-point multiplicative seasonal index centred on 1.0."""
    months = np.arange(1, 13)
    prof = 1.0 + amplitude * np.cos(2 * np.pi * (months - peak_month) / 12)
    return prof / prof.mean()


def make_synthetic(seed: int = 7) -> pd.DataFrame:
    """
    Build a 5-year (Jan 2020 - Dec 2024), monthly, multi-SKU / multi-location
    dataset that contains, on purpose:
      * trend + seasonality  (P001 washer)
      * intermittent demand  (P002 compressor spare  -> Croston territory)
      * stockout censoring   (P003 microwave MG-750   -> unconstraining demo)
      * high-variability/erratic (P004 premium fridge)
      * promotions, price, and a few one-off outliers
    The column `demand_true` exists ONLY because this is synthetic; it lets us
    score unconstraining honestly. Real feeds never have it.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    t = np.arange(n)
    rows = []

    def push(pid, name, loc, demand_true, sales, price, promo, stockout):
        for i, d in enumerate(dates):
            rows.append(
                dict(date=d, product_id=pid, product_name=name, location=loc,
                     demand_true=round(float(demand_true[i])),
                     sales=round(float(sales[i])),
                     price=round(float(price[i]), 2),
                     on_promo=int(promo[i]), stockout_flag=int(stockout[i]))
            )

    # ---- P001 Front-Load Washer: 2 locations, trend + seasonality + promos ----
    for loc, base, growth, amp, peak in [("Chennai", 480, 2.6, 0.28, 4),
                                         ("Mumbai", 640, 3.1, 0.22, 5)]:
        seas = _seasonal_profile(amp, peak)[((dates.month - 1) % 12)]
        promo = (rng.random(n) < 0.18).astype(int)
        price = 32000 - 1500 * promo + rng.normal(0, 250, n)
        demand = (base + growth * t) * seas * (1 + 0.22 * promo) \
                 + rng.normal(0, 22, n)
        # two one-off demand spikes (bulk B2B orders -> outliers to be cleansed)
        for k in rng.choice(np.arange(6, n - 6), size=2, replace=False):
            demand[k] *= rng.uniform(1.8, 2.4)
        demand = np.clip(demand, 0, None)
        push("P001", "Front-Load Washer 7kg", loc, demand, demand,
             price, promo, np.zeros(n))

    # ---- P002 Compressor Spare: intermittent / lumpy ----
    base_p = 0.40 + 0.20 * (t / n)               # demand slowly more frequent
    occur = (rng.random(n) < base_p)             # demand only some months
    sizes = rng.gamma(shape=2.2, scale=10.0, size=n)
    demand = np.where(occur, sizes, 0.0)
    price = 4200 + rng.normal(0, 80, n)
    push("P002", "Compressor Spare Unit", "Chennai", demand, demand,
         price, np.zeros(n), np.zeros(n))

    # ---- P003 Microwave MG-750: seasonality + STOCKOUT CENSORING ----
    seas = _seasonal_profile(0.30, 11)[((dates.month - 1) % 12)]   # festive Q4 peak
    promo = (rng.random(n) < 0.15).astype(int)
    price = 9800 - 600 * promo + rng.normal(0, 120, n)
    demand_true = (300 + 1.4 * t) * seas * (1 + 0.18 * promo) + rng.normal(0, 18, n)
    demand_true = np.clip(demand_true, 0, None)
    # supply capacity caps some festive months -> we only observe censored sales
    capacity = np.full(n, 1e9)
    cap_months = [10, 11, 22, 23, 34, 46, 47]    # recurring Nov/Dec shortfalls
    for m in cap_months:
        capacity[m] = demand_true[m] * rng.uniform(0.55, 0.78)
    sales = np.minimum(demand_true, capacity)
    stockout = (sales < demand_true - 1e-6).astype(int)
    push("P003", "Microwave MG-750", "Chennai", demand_true, sales,
         price, promo, stockout)

    # ---- P004 Premium Refrigerator: erratic / high CV ----
    seas = _seasonal_profile(0.15, 6)[((dates.month - 1) % 12)]
    promo = (rng.random(n) < 0.2).astype(int)
    price = 58000 - 4000 * promo + rng.normal(0, 600, n)
    demand = (140 + 0.4 * t) * seas * (1 + 0.3 * promo) \
             + rng.normal(0, 38, n)              # large noise -> high CV
    demand = np.clip(demand, 0, None)
    push("P004", "Premium Refrigerator", "Chennai", demand, demand,
         price, promo, np.zeros(n))

    df = pd.DataFrame(rows).sort_values(["product_id", "location", "date"])
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce/validate a user-supplied frame into the engine's contract."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Need at least {REQUIRED_COLS}.")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0.0)
    for opt, default in [("on_promo", 0), ("stockout_flag", 0),
                         ("price", np.nan), ("product_name", None),
                         ("demand_true", np.nan)]:
        if opt not in df.columns:
            df[opt] = default
    if df["product_name"].isna().all():
        df["product_name"] = df["product_id"]
    return df.sort_values(["product_id", "location", "date"]).reset_index(drop=True)


def series_key(df: pd.DataFrame) -> pd.Series:
    return df["product_id"].astype(str) + " | " + df["location"].astype(str)
