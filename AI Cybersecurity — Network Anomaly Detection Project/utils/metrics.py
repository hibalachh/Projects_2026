"""Evaluation metrics, curve helpers and the data-driven analysis text."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

METRIC_COLS = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC", "Specificity"]


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def evaluate(y_true, score, threshold: float) -> dict:
    """All headline metrics for one model at one threshold."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    pred = (s > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    both = len(np.unique(y)) == 2
    return {
        "Accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "Precision": precision,
        "Recall": recall,
        "F1-score": _safe_div(2 * precision * recall, precision + recall),
        "ROC-AUC": float(roc_auc_score(y, s)) if both else math.nan,
        "PR-AUC": float(average_precision_score(y, s)) if both else math.nan,
        "Specificity": _safe_div(tn, tn + fp),
        "cm": np.array([[tn, fp], [fn, tp]], dtype=int),
        "threshold": float(threshold),
        "flag_rate": float(pred.mean()),
    }


def baseline_flag_everything(y_true) -> dict:
    """Metrics of a 'detector' that labels every flow as an attack.

    A useful yardstick when the evaluation set is dominated by one class.
    """
    y = np.asarray(y_true).astype(int)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    p = _safe_div(n1, n1 + n0)
    return {
        "Accuracy": p, "Precision": p, "Recall": 1.0 if n1 else 0.0,
        "F1-score": _safe_div(2 * p, 1 + p), "ROC-AUC": 0.5, "PR-AUC": p, "Specificity": 0.0,
        "cm": np.array([[0, n0], [0, n1]], dtype=int), "threshold": math.nan, "flag_rate": 1.0,
    }


def youden_threshold(y_true, score) -> tuple[float, float]:
    """Threshold maximising TPR - FPR (Youden's J), for the rule ``score > threshold``."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    i = int(np.argmax(j))
    t = thr[i]
    if not np.isfinite(t):
        t = float(np.median(s))
    # roc_curve flags score >= t; nextafter makes the strict rule `score > t'` equivalent.
    return float(np.nextafter(t, -np.inf)), float(j[i])


def _thin(arrays, max_points: int):
    n = len(arrays[0])
    if n <= max_points:
        return arrays
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return tuple(a[idx] for a in arrays)


def roc_points(y_true, score, max_points: int = 1500):
    fpr, tpr, _ = roc_curve(y_true, score)
    return _thin((fpr, tpr), max_points)


def pr_points(y_true, score, max_points: int = 1500):
    precision, recall, _ = precision_recall_curve(y_true, score)
    return _thin((recall, precision), max_points)


def operating_point(cm: np.ndarray) -> dict:
    tn, fp, fn, tp = cm.ravel()
    return {"fpr": _safe_div(fp, fp + tn), "tpr": _safe_div(tp, tp + fn), "precision": _safe_div(tp, tp + fp)}


# --------------------------------------------------------------------------- #
# Analysis text
# --------------------------------------------------------------------------- #
def build_analysis(rows: dict[str, dict], baseline: dict, threshold_mode: str) -> list[str]:
    """Plain-language reading of the metric table, generated from the numbers themselves
    so it stays correct whichever thresholds or evaluation set are selected."""
    names = list(rows)
    if not names:
        return []

    def best(col: str) -> str:
        return max(names, key=lambda n: -1 if math.isnan(rows[n][col]) else rows[n][col])

    roc_b, pr_b, f1_b = best("ROC-AUC"), best("PR-AUC"), best("F1-score")
    p = baseline["Accuracy"]
    out: list[str] = []

    out.append(
        f"**Ranking quality, independent of any threshold.** {roc_b} separates attacks from normal "
        f"flows best (ROC-AUC {rows[roc_b]['ROC-AUC']:.3f}), and {pr_b} leads on PR-AUC "
        f"({rows[pr_b]['PR-AUC']:.3f}). These two numbers judge the score itself, before any cut-off is applied."
    )

    sentences = []
    for n in names:
        r = rows[n]
        sentences.append(
            f"{n} catches {r['Recall']:.0%} of attacks and raises a false alarm on "
            f"{1 - r['Specificity']:.0%} of normal flows (precision {r['Precision']:.0%})"
        )
    out.append(
        f"**Operating point, which depends on the threshold ({threshold_mode.lower()}).** "
        f"{f1_b} has the highest F1-score ({rows[f1_b]['F1-score']:.3f}). "
        + "; ".join(sentences) + "."
    )

    out.append(
        f"**Read every number against a trivial baseline.** {p:.0%} of this evaluation set is attack "
        f"traffic, so a detector that flags *every* flow already reaches {p:.0%} accuracy and an F1-score "
        f"of {baseline['F1-score']:.3f} (the first row of the table). A model whose F1 sits close to that "
        f"figure is mostly reflecting class balance, not skill; specificity and ROC-AUC show it more clearly."
    )

    degenerate = [n for n in names if rows[n]["Specificity"] < 0.01]
    if degenerate:
        who = ", ".join(degenerate)
        worst_spec = min(rows[n]["Specificity"] for n in degenerate)
        aucs = ", ".join("{:.3f}".format(rows[n]["ROC-AUC"]) for n in degenerate)
        out.append(
            f"**{who}: the threshold, not the score, is the problem.** At this cut-off the model flags "
            f"essentially every flow (specificity {worst_spec:.1%}). "
            f"Its ROC-AUC ({aucs}) shows the underlying "
            f"score still carries real signal. Choosing a threshold that maximises F1 on an attack-heavy "
            f"validation set rewards flagging everything; a balanced criterion such as Youden's J avoids this. "
            f"Switch the threshold mode in the sidebar to compare."
        )

    out.append(
        "**Practical trade-offs.** The Isolation Forest is cheap to train on hundreds of thousands of flows. "
        "The One-Class SVM's cost grows quickly with training size, which is why it was fitted on a "
        "30,000-flow subsample, and its scoring cost grows with its number of support vectors. The "
        "Autoencoder can capture non-linear relations between features and gives a per-flow error that is "
        "easy to explain, at the cost of a heavier dependency (TensorFlow) and more tuning."
    )
    return out
