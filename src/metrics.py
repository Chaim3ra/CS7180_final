"""Pure metric functions for solar generation forecast evaluation."""
from __future__ import annotations

import numpy as np
from typing import Optional


def _flat(arr: np.ndarray) -> np.ndarray:
    return arr.flatten().astype(np.float64)


def compute_mae(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.abs(_flat(preds) - _flat(actuals))))


def compute_rmse(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((_flat(preds) - _flat(actuals)) ** 2)))


def compute_mape(preds: np.ndarray, actuals: np.ndarray) -> float:
    """MAPE (%), skipping timesteps where actual == 0."""
    p, a = _flat(preds), _flat(actuals)
    mask = a > 1e-6
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((p[mask] - a[mask]) / a[mask])) * 100)


def compute_r2(preds: np.ndarray, actuals: np.ndarray) -> float:
    p, a = _flat(preds), _flat(actuals)
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot < 1e-10:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_skill_score(
    preds: np.ndarray, actuals: np.ndarray, persistence: np.ndarray
) -> float:
    """Skill score vs persistence baseline (%). Higher = better than yesterday."""
    mae_model = compute_mae(preds, actuals)
    mae_pers  = compute_mae(persistence, actuals)
    if mae_pers < 1e-10:
        return float("nan")
    return float((1.0 - mae_model / mae_pers) * 100)


def compute_peak_mae(
    preds: np.ndarray, actuals: np.ndarray, hours: np.ndarray
) -> float:
    """MAE restricted to 8 am–4 pm timesteps (solar production hours)."""
    p, a, h = _flat(preds), _flat(actuals), np.asarray(hours).flatten()
    mask = (h >= 8) & (h < 16)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(p[mask] - a[mask])))


def compute_all(
    preds: np.ndarray,
    actuals: np.ndarray,
    persistence: Optional[np.ndarray] = None,
    hours: Optional[np.ndarray] = None,
) -> dict:
    """Compute all 7 metrics. Returns a flat dict."""
    return {
        "mae":         compute_mae(preds, actuals),
        "rmse":        compute_rmse(preds, actuals),
        "mape":        compute_mape(preds, actuals),
        "r2":          compute_r2(preds, actuals),
        "skill_score": (
            compute_skill_score(preds, actuals, persistence)
            if persistence is not None else float("nan")
        ),
        "peak_mae": (
            compute_peak_mae(preds, actuals, hours)
            if hours is not None else float("nan")
        ),
    }
