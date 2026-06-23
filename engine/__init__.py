"""Demand-forecasting engine — clean, modular, tool-agnostic.

Pipeline stages (each its own module so they can be read, tested and swapped):
    data          ingestion / validation / synthetic generator
    cleansing     unconstraining + outlier correction
    segmentation  ADI/CV^2 pattern + ABC/XYZ
    models        baselines, ETS, SARIMA, Croston family, ML
    backtest      rolling-origin CV, metrics, bias, FVA, selection
    reconcile     hierarchical bottom-up / top-down
    pipeline      end-to-end orchestration
"""
from . import data, cleansing, segmentation, models, backtest, reconcile, pipeline
from .pipeline import run_pipeline

__all__ = ["data", "cleansing", "segmentation", "models", "backtest",
           "reconcile", "pipeline", "run_pipeline"]
__version__ = "1.0.0"
