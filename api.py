"""
api.py — FastAPI backend.

Endpoints
---------
GET  /                      -> the dashboard (static/index.html)
GET  /api/health            -> liveness + which ML backend is active
POST /api/run               -> run the full pipeline. Body (optional):
                               {horizon, n_folds, allow_all_models, outlier_k}
                               If a CSV file is uploaded (multipart 'file'),
                               it is used; otherwise the built-in synthetic
                               dataset is generated.

The pipeline returns pandas objects; everything is converted to plain JSON here
so the engine never needs to know about the transport layer.

Run locally:
    uvicorn api:app --reload --port 8000
then open http://localhost:8000
"""
from __future__ import annotations
import io
import warnings
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

from engine import data as datamod, run_pipeline
from engine.models import ML_BACKEND

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Demand Forecasting Engine", version="1.0.0")


# --------------------------------------------------------------------------- #
# JSON serialisation of the results object
# --------------------------------------------------------------------------- #
def _clean(obj):
    """Recursively make numpy/pandas types JSON-safe."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 3)
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, 3)
    return obj


def serialize(res: dict) -> dict:
    seg = res["segmentation"]
    sel = res["selections"]
    out = {
        "meta": res["meta"],
        "segmentation": _clean(seg.to_dict(orient="records")),
        "selections": _clean(sel.to_dict(orient="records")),
        "change_log": _clean(
            res["change_log"].assign(
                date=res["change_log"]["date"].astype(str)
            ).to_dict(orient="records")) if not res["change_log"].empty else [],
        "series": {k: v for k, v in res["series_lines"].items()},
        "backtests": {k: _clean(v.to_dict(orient="records"))
                      for k, v in res["backtests"].items()},
        "hierarchy_total": _clean(
            res["hierarchy"]["total"].to_dict(orient="records")),
        "hierarchy_product": _clean(
            res["hierarchy"]["product"]
            .assign(step=lambda d: d["step"].astype(int))
            .to_dict(orient="records")),
    }
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok", "ml_backend": ML_BACKEND}


@app.post("/api/run")
async def run(file: UploadFile | None = File(default=None),
              horizon: int = Form(default=6),
              n_folds: int = Form(default=4),
              allow_all_models: bool = Form(default=False),
              outlier_k: float = Form(default=4.0)):
    if file is not None:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        source = f"uploaded:{file.filename}"
    else:
        df = datamod.make_synthetic()
        source = "synthetic"
    try:
        res = run_pipeline(df, horizon=horizon, n_folds=n_folds,
                           allow_all_models=allow_all_models,
                           outlier_k=outlier_k)
    except Exception as e:                       # surface a usable error
        return JSONResponse(status_code=400,
                            content={"error": str(e),
                                     "hint": "CSV needs columns: date, "
                                             "product_id, location, sales"})
    payload = serialize(res)
    payload["meta"]["source"] = source
    return payload


# static dashboard (mounted last so /api/* wins)
if os.path.isdir(os.path.join(HERE, "static")):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(HERE, "static", "index.html"))
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")),
              name="static")
