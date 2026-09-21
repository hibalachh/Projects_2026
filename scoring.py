"""Scoring flows with the three models and turning scores into decisions.

Convention used everywhere: **higher score = more anomalous**, and a flow is flagged
as an anomaly when ``score > threshold`` (the same rule as in the notebook).

* Autoencoder      -> mean squared reconstruction error
* Isolation Forest -> ``-decision_function``
* One-Class SVM    -> ``-decision_function``
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .artifacts import Artifacts

MODE_SAVED = "Saved threshold"
MODE_BALANCED = "Balanced (Youden's J)"
MODE_PERCENT = "Top % of scores"


def score_col(key: str) -> str:
    return f"score_{key}"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def autoencoder_errors(model, Xs: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    out = np.empty(len(Xs), dtype=np.float64)
    for start in range(0, len(Xs), batch_size):
        chunk = Xs[start:start + batch_size]
        recon = model.predict(chunk, batch_size=batch_size, verbose=0)
        out[start:start + batch_size] = np.mean(np.square(chunk - recon), axis=1)
    return out


def svm_scores(model, Xs: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    """Chunked so the kernel matrix (rows x support vectors) never gets huge."""
    out = np.empty(len(Xs), dtype=np.float64)
    for start in range(0, len(Xs), chunk):
        out[start:start + chunk] = -model.decision_function(Xs[start:start + chunk])
    return out


def score_dataframe(art: Artifacts, X: pd.DataFrame) -> pd.DataFrame:
    """Scale ``X`` and return one score column per available model (index preserved)."""
    if art.scaler is None:
        raise RuntimeError("The scaler is not available.")
    Xs = np.asarray(art.scaler.transform(X[art.features]), dtype=np.float64)
    out = pd.DataFrame(index=X.index)
    for lm in art.available_models():
        key = lm.spec.key
        if key == "autoencoder":
            out[score_col(key)] = autoencoder_errors(lm.model, Xs)
        elif key == "isolation_forest":
            out[score_col(key)] = -lm.model.decision_function(Xs)
        else:
            out[score_col(key)] = svm_scores(lm.model, Xs)
    return out


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
@dataclass
class ThresholdChoice:
    value: float
    mode: str
    note: str


def resolve_thresholds(
    art: Artifacts,
    scores: pd.DataFrame,
    y: np.ndarray | None,
    mode: str,
    percent: float = 20.0,
    calibration: tuple[pd.DataFrame, np.ndarray] | None = None,
) -> dict[str, ThresholdChoice]:
    """Return the effective threshold for every available model under ``mode``.

    ``calibration`` = (scores, labels) of a *separate* labelled set used to fit the
    Balanced threshold. Without it, Balanced is fitted on the scores being displayed
    (fine for exploration, but optimistic as an evaluation).
    """
    result: dict[str, ThresholdChoice] = {}
    for lm in art.available_models():
        key = lm.spec.key
        col = score_col(key)
        s = scores[col].to_numpy()

        if mode == MODE_BALANCED:
            if calibration is not None:
                cal_scores, cal_y = calibration[0][col].to_numpy(), calibration[1]
                where = "the threshold-selection split (not used for the test metrics)"
            elif y is not None:
                cal_scores, cal_y = s, y
                where = "this sample (in-sample)"
            else:
                cal_scores = cal_y = None
            if cal_scores is not None and len(np.unique(cal_y)) == 2:
                value, j = metrics.youden_threshold(cal_y, cal_scores)
                result[key] = ThresholdChoice(
                    value, mode, f"Maximises true-positive rate minus false-positive rate on {where} (J = {j:.2f}).")
                continue
            # No labels available -> fall through to the saved threshold.
            result[key] = ThresholdChoice(
                lm.threshold, MODE_SAVED, "Balanced needs labelled data; using the saved threshold.")
        elif mode == MODE_PERCENT:
            value = float(np.percentile(s, 100.0 - percent))
            result[key] = ThresholdChoice(value, mode, f"Flags the {percent:g}% highest-scoring flows.")
        else:
            src = {"saved": "Loaded from the model's .pkl file.",
                   "notebook_fallback": "Saved file was invalid; using the value printed in the notebook."}
            result[key] = ThresholdChoice(lm.threshold, MODE_SAVED, src.get(lm.threshold_source, ""))
    return result


def predict(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(scores) > threshold).astype(int)
