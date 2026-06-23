"""
demo.py — run the whole engine end-to-end on synthetic data and print a report.
Also saves two PNG charts so you can see the forecasts, not just the numbers.

    python scripts/demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import data, run_pipeline

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)

print("Generating synthetic 5-year monthly multi-SKU dataset...")
df = data.make_synthetic()
df.to_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                       "sample_demand.csv"), index=False)
print(f"  rows={len(df)}  series={df.groupby(['product_id','location']).ngroups}"
      f"  span={df['date'].min():%Y-%m}..{df['date'].max():%Y-%m}\n")

print("Running pipeline (clean -> segment -> backtest -> select -> forecast -> reconcile)...")
res = run_pipeline(df, horizon=6, n_folds=4)
m = res["meta"]
print(f"  ML backend: {m['ml_backend']} | stockout months unconstrained: "
      f"{m['stockout_months']} | outliers corrected: {m['outliers_corrected']}\n")

print("=" * 70, "\nSEGMENTATION (ADI/CV^2 pattern + ABC/XYZ)\n", "=" * 70, sep="")
print(res["segmentation"][["product_id", "location", "pattern", "ADI", "CV2",
                           "ABC", "XYZ", "segment", "annual_value"]]
      .to_string(index=False))

print("\n", "=" * 70, "\nMODEL SELECTION + ACCURACY (per series, FVA vs Seasonal-Naive)\n",
      "=" * 70, sep="")
print(res["selections"].round(2).to_string(index=False))

print("\n", "=" * 70, "\nBACKTEST DETAIL — P001 | Mumbai (all candidates)\n",
      "=" * 70, sep="")
print(res["backtests"]["P001 | Mumbai"].to_string(index=False))

print("\n", "=" * 70, "\nBACKTEST DETAIL — P002 | Chennai (intermittent spare)\n",
      "=" * 70, sep="")
print(res["backtests"]["P002 | Chennai"].to_string(index=False))

print("\n", "=" * 70, "\nRECONCILED TOTAL FORECAST (bottom-up sum of all leaves)\n",
      "=" * 70, sep="")
print(res["hierarchy"]["total"].round(1).to_string(index=False))

# ---- charts -------------------------------------------------------------- #
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")

# Chart 1: P003 unconstraining (raw sales vs unconstrained demand vs forecast)
g = df[(df.product_id == "P003")].sort_values("date")
line = res["series_lines"]["P003 | Chennai"]
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(pd.to_datetime(line["history_dates"]), line["raw_sales"], "o-",
        color="#9aa0a6", lw=1.3, ms=3, label="Raw sales (censored by stockouts)")
ax.plot(pd.to_datetime(line["history_dates"]), line["history"], "-",
        color="#4f8cff", lw=2, label="Clean demand (unconstrained + de-spiked)")
ax.plot(pd.to_datetime(line["forecast_dates"]), line["forecast"], "s--",
        color="#36d399", lw=2, ms=4, label=f"Forecast ({line['best_model']})")
so = g[g.stockout_flag == 1]
ax.scatter(pd.to_datetime(so["date"].dt.strftime("%Y-%m")),
           so["sales"], color="#f87272", zorder=5, s=42,
           label="Stockout month")
ax.set_title("P003 Microwave MG-750 — unconstraining recovers lost festive demand")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "chart_unconstraining.png"), dpi=130)

# Chart 2: P001 Mumbai history + forecast
line = res["series_lines"]["P001 | Mumbai"]
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(pd.to_datetime(line["history_dates"]), line["history"], "-",
        color="#4f8cff", lw=2, label="Clean demand")
ax.plot(pd.to_datetime(line["forecast_dates"]), line["forecast"], "s--",
        color="#36d399", lw=2, ms=4, label=f"Forecast ({line['best_model']})")
ax.set_title("P001 Front-Load Washer (Mumbai) — trend + seasonality, 6-month forecast")
ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(out_dir, "chart_forecast.png"), dpi=130)

print("\nSaved charts: data/chart_unconstraining.png, data/chart_forecast.png")
print("DONE.")
