# Demand Planner's Cockpit

A small but **real** demand-forecasting application — not a toy. It does the
unglamorous 80% that actually separates a demand planner from someone who can
call `.fit()`: it cleans censored history into true demand, segments SKUs by
demand pattern, backtests a portfolio of models *the right way* (rolling-origin,
no leakage), keeps only the models that beat a naive baseline (**Forecast Value
Added**), and reconciles the hierarchy so the numbers add up.

Everything runs on the included synthetic data out of the box, or on your own
CSV.

```
raw sales
  └─▶ unconstrain stockouts        recover demand lost when you were out of stock
       └─▶ segment ADI/CV² + ABC/XYZ   decide which SKU gets which treatment
            └─▶ cleanse outliers       seasonality-aware; intermittent left alone
                 └─▶ rolling backtest  walk-forward CV, WMAPE/RMSE/bias/tracking
                      └─▶ select + FVA  pick the winner, prove it beats naive
                           └─▶ forecast refit on full history
                                └─▶ reconcile  bottom-up coherent total
```

---

## Run it

Three ways, same engine underneath. Python 3.10+.

```bash
pip install -r requirements.txt
```

**A. One-file app (easiest)**

```bash
streamlit run app_streamlit.py
```

**B. API + dashboard (the cockpit UI)**

```bash
uvicorn api:app --reload --port 8000
# open http://localhost:8000
```

**C. Headless demo (prints a full report + saves charts)**

```bash
python scripts/demo.py
```

If LightGBM isn't installed, the ML model automatically falls back to
scikit-learn's `HistGradientBoostingRegressor`, so the app always runs.

---

## Use your own data

Drop a CSV with these columns (extra columns are ignored):

| column          | required | meaning                                   |
|-----------------|----------|-------------------------------------------|
| `date`          | ✅       | month start, e.g. `2024-03-01`            |
| `product_id`    | ✅       | SKU code                                  |
| `location`      | ✅       | selling / shipping location               |
| `sales`         | ✅       | what you actually shipped/invoiced        |
| `price`         | optional | used by ML / regression models            |
| `on_promo`      | optional | 0/1 promotion flag                        |
| `stockout_flag` | optional | 0/1 — month was supply-constrained        |

Upload it in either UI, or `pd.read_csv(...)` and call
`engine.run_pipeline(df)`. Monthly buckets are assumed (`period=12`); change the
`period` argument for weekly/other.

---

## What each part teaches

The engine is deliberately split so every stage is readable on its own.

### `engine/data.py` — history is never clean
Generates 5 years of monthly, multi-SKU/multi-location demand that *intentionally*
contains trend, seasonality, promotions, one-off outliers, intermittent spare-part
demand, and stockout censoring — so every downstream stage has something real to
do. Also validates/normalises any CSV you bring.

### `engine/cleansing.py` — sales ≠ demand
- **Unconstraining**: when `stockout_flag = 1`, recorded sales understate demand.
  We estimate the latent demand from the SKU's *un-censored* seasonal profile and
  lift those months (never below what you already sold). This is the single most
  senior move in the whole pipeline — forecasting censored sales quietly
  under-forecasts your best movers forever.
- **Seasonality-aware outlier cleansing**: we deseasonalise (monthly median) and
  detrend (rolling median) *before* a robust Hampel test, so a genuine festive
  peak is **kept** while a one-off bulk order is damped. Every correction is
  logged. Intermittent/lumpy series are skipped — their spikes are the signal.

### `engine/segmentation.py` — you can't hand-tune 50,000 SKUs
- **ADI / CV²** (Syntetos–Boylan): classifies each series as **Smooth /
  Erratic / Intermittent / Lumpy**. This decides which model class can even work.
- **ABC × XYZ**: value (Pareto) × variability — the everyday triage map for where
  forecast error costs the most money.

### `engine/models.py` — the portfolio
| model | when it shines |
|-------|----------------|
| Naive / **Seasonal-Naive** | the bar every real model must clear (FVA benchmark) |
| Moving average | stable, low-signal series |
| **ETS / Holt-Winters** | clear trend + seasonality |
| **Auto-SARIMA** | autocorrelated series (compact AIC grid search) |
| **Croston / SBA / TSB** | intermittent demand (zero-inflated); TSB handles obsolescence |
| **ML (LightGBM/GBM)** | many drivers, lots of history; recursive multi-step with lag + calendar features |

All models share one signature, so the backtester treats them interchangeably.
Croston/SBA/TSB are hand-rolled (~a dozen lines each) so you can read exactly
what they do.

### `engine/backtest.py` — prove it before you trust it
- **Rolling-origin (walk-forward)** evaluation — never random k-fold on a time
  series; that leaks the future into the past.
- Metrics that don't lie on low/zero volume: **WMAPE**, MAE, RMSE, signed
  **BIAS%**, and a **tracking signal** (|TS| > ~4 ⇒ a persistent, directional
  miss — usually an organisational problem, not a statistical one).
- **Forecast Value Added (FVA)**: every candidate is scored against seasonal-naive.
  If a fancy model can't beat "same month last year", it's destroying value. This
  is the most credible single number to put in front of a steering committee.

### `engine/reconcile.py` — make the numbers add up
Independently-made forecasts at SKU / product / total level won't sum correctly.
**Bottom-up** (sum the leaves) and **top-down** (forecast the aggregate, split by
share) are implemented, plus a coherence-gap KPI. *(MinT / optimal reconciliation
is the advanced next step — see below.)*

### `engine/pipeline.py` — one source of truth
Orchestrates all of the above into a single results object that the API, the
Streamlit app and the demo script all consume identically.

---

## The seven pillars, mapped to code

This app is the practical half of the demand-planning skill set:

1. **Forecasting methods** → `models.py`
2. **Forecastability & accuracy science** (WMAPE, bias, FVA, ADI/CV²) →
   `backtest.py`, `segmentation.py`
3. **Data & history management** (sales≠demand, unconstraining, outliers) →
   `cleansing.py`
4. **Tools** → this is a miniature of what IBP / Kinaxis / o9 automate
5. **S&OP / IBP process** → the reconcile + FVA loop is the analytical core of it
6. **Domain specialisation** → the intermittent-demand path is built for
   spare-parts / maritime, where most planners are weak
7. **Situational judgment** → the engine *shows* its working (logs, FVA,
   backtests) so a human can overrule it

---

## Where to take it next

- **MinT reconciliation** — optimal (trace-minimising) hierarchical reconciliation
  instead of bottom-up/top-down. Needs the residual covariance matrix.
- **`statsforecast` (Nixtla)** — drop-in, C-fast AutoARIMA/ETS/Croston; uncomment
  in `requirements.txt`.
- **Probabilistic forecasts** — prediction intervals + service-level-driven safety
  stock, not just point forecasts.
- **Demand sensing** — blend in short-term signals (orders, weather, web traffic)
  for the 0–4 week horizon.
- **Promo/price elasticity** — the exogenous hooks are already wired into the ML
  model; add a causal layer.

---

## Project layout

```
demand-forecasting-app/
├── engine/              the forecasting core (read this)
│   ├── data.py          generate / validate data
│   ├── cleansing.py     unconstrain + outlier-correct
│   ├── segmentation.py  ADI/CV² + ABC/XYZ
│   ├── models.py        baselines, ETS, SARIMA, Croston family, ML
│   ├── backtest.py      rolling-origin CV, metrics, FVA, selection
│   ├── reconcile.py     hierarchical reconciliation
│   └── pipeline.py      end-to-end orchestration
├── api.py               FastAPI backend
├── static/index.html    dark dashboard (the cockpit)
├── app_streamlit.py     one-file launcher
├── scripts/demo.py      headless end-to-end demo + charts
├── data/                sample CSV + generated charts
└── requirements.txt
```

## Put it on GitHub

```bash
cd demand-forecasting-app
git init && git add . && git commit -m "Demand forecasting engine"
git branch -M main
git remote add origin https://github.com/<you>/demand-forecasting-app.git
git push -u origin main
```

A `.gitignore` for `__pycache__/`, `*.pyc` and `data/*.png` is included.
