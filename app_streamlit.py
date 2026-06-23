"""
app_streamlit.py — the zero-config front door.

If you don't want to run a separate API + browser, this gives you the whole
engine in one window:

    pip install -r requirements.txt
    streamlit run app_streamlit.py

It imports the EXACT same engine package as the FastAPI backend, so there is
only one source of forecasting truth.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from engine import data as datamod, run_pipeline
from engine.models import ML_BACKEND

st.set_page_config(page_title="Demand Planner's Cockpit", layout="wide",
                   page_icon="📦")

st.markdown("""
<style>
  .stApp{background:#0d1015;color:#e7eaf0}
  h1,h2,h3{font-family:Georgia,serif;letter-spacing:-.02em}
  [data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace;color:#45c8f5}
</style>""", unsafe_allow_html=True)

st.title("Demand Planner's Cockpit")
st.caption("clean → segment → backtest → forecast → reconcile · "
           f"ML backend: {ML_BACKEND}")

# ---- sidebar controls ---------------------------------------------------- #
with st.sidebar:
    st.header("Run settings")
    horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
    n_folds = st.slider("Backtest folds", 2, 6, 4)
    outlier_k = st.slider("Outlier sensitivity (k)", 2.5, 6.0, 4.0, 0.5,
                          help="Lower = more aggressive outlier removal")
    allow_all = st.checkbox("Backtest the full model portfolio", value=False,
                            help="Otherwise only pattern-appropriate models run")
    up = st.file_uploader("Upload demand CSV", type=["csv"],
                          help="Columns: date, product_id, location, sales "
                               "(+ optional price, on_promo, stockout_flag)")
    run = st.button("Run engine", type="primary", use_container_width=True)

if "res" not in st.session_state or run:
    df = pd.read_csv(up) if up is not None else datamod.make_synthetic()
    with st.spinner("Cleansing, backtesting and forecasting every series…"):
        st.session_state.res = run_pipeline(
            df, horizon=horizon, n_folds=n_folds,
            allow_all_models=allow_all, outlier_k=outlier_k)
        st.session_state.source = "uploaded" if up is not None else "synthetic"

res = st.session_state.res
m = res["meta"]
sel = res["selections"]
avg_fva = sel["FVA_vs_SNaive"].dropna().mean()

# ---- KPI row ------------------------------------------------------------- #
c = st.columns(5)
c[0].metric("SKU-locations", m["n_series"])
c[1].metric("Stockout months recovered", m["stockout_months"])
c[2].metric("Outliers cleaned", m["outliers_corrected"])
c[3].metric("Avg FVA vs naive", f"+{avg_fva:.1f} pts")
c[4].metric("Source", st.session_state.source)

# ---- per-series forecast chart ------------------------------------------- #
st.subheader("01 · Forecast & signal recovery")
keys = list(res["series_lines"].keys())
key = st.selectbox("Series", keys)
s = res["series_lines"][key]
hist_idx = pd.to_datetime(s["history_dates"])
fc_idx = pd.to_datetime(s["forecast_dates"])
plot_df = pd.DataFrame(index=hist_idx.union(fc_idx))
plot_df.loc[hist_idx, "Raw sales (constrained)"] = s["raw_sales"]
plot_df.loc[hist_idx, "Clean demand"] = s["history"]
plot_df.loc[fc_idx, "Forecast"] = s["forecast"]
st.caption(f"Model selected by backtest: **{s['best_model']}**")
st.line_chart(plot_df, color=["#e0a13c", "#45c8f5", "#4ad6a0"])

# ---- segmentation -------------------------------------------------------- #
st.subheader("02 · Segmentation (ADI/CV² + ABC×XYZ)")
seg = res["segmentation"][["product_id", "location", "product_name", "pattern",
                           "ADI", "CV2", "ABC", "XYZ", "segment", "annual_value"]]
st.dataframe(seg, use_container_width=True, hide_index=True)

# ---- model leaderboard --------------------------------------------------- #
st.subheader("03 · Model selection & value added")
st.dataframe(sel.round(2), use_container_width=True, hide_index=True)

# ---- backtest detail ----------------------------------------------------- #
st.subheader("04 · Backtest detail")
bkey = st.selectbox("Series ", keys, key="bt")
st.dataframe(res["backtests"][bkey], use_container_width=True, hide_index=True)

# ---- reconciliation ------------------------------------------------------ #
st.subheader("05 · Reconciled network total (bottom-up)")
tot = res["hierarchy"]["total"].set_index("step")["forecast"]
st.bar_chart(tot, color="#45c8f5")

with st.expander("What just happened (the pipeline in words)"):
    st.markdown("""
1. **Unconstrain** — stockout months were lifted from censored *sales* to an
   estimate of true *demand* using each SKU's clean-month seasonal profile.
2. **Segment** — every series classified by **ADI/CV²** (smooth / erratic /
   intermittent / lumpy) and by **ABC×XYZ**. Pattern decides the model class.
3. **Cleanse** — one-off outliers removed *seasonality-aware* (festive peaks are
   kept); intermittent series are left untouched because spikes are signal.
4. **Backtest** — rolling-origin (walk-forward) CV per series; metrics include
   WMAPE, RMSE, directional **bias** and a **tracking signal**.
5. **Select** — the model with the best out-of-sample error wins, and we report
   **Forecast Value Added** vs a seasonal-naive baseline. If nothing beats
   naive, that is itself the finding.
6. **Reconcile** — leaf forecasts summed bottom-up into a coherent total.
""")
