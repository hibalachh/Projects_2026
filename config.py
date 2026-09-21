"""Central configuration: paths, model registry, palette and reference values.

Everything that a person might want to edit (author name, GitHub link, dataset
location) or that several modules need to agree on (feature list, model names,
colours) lives here so it is defined exactly once.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Personal details shown on the Home page  ->  EDIT THESE
# --------------------------------------------------------------------------- #
AUTHOR = "Your Name"
GITHUB_URL = "https://github.com/your-username"
PORTFOLIO_URL = ""  # optional, e.g. "https://your-site.dev"
REPO_URL = ""       # optional, link to this project's repository

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# The dataset the dashboard opens with. Override with an environment variable,
# e.g.  ANOMALY_DATASET_PATH=/data/cicids2017_binary_balanced.csv
DEFAULT_DATASET = Path(
    os.environ.get("ANOMALY_DATASET_PATH", DATA_DIR / "cicids2017_binary_balanced.csv")
)

# --------------------------------------------------------------------------- #
# Dataset / notebook conventions (must match Project.ipynb)
# --------------------------------------------------------------------------- #
LABEL_COL = "Attack_Binary"          # 0 = normal, 1 = attack
REDUNDANT_COL = "Subflow Fwd Bytes"  # dropped in the notebook (duplicate of another feature)
SEED = 42                            # random_state used for every split in the notebook

# The 51 features the scaler and all three models were trained on, in order.
# At run time the order stored inside scaler.pkl takes precedence; this list is
# only a fallback if the scaler carries no feature names.
FEATURE_COLUMNS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "PSH Flag Count", "ACK Flag Count", "Average Packet Size",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Max", "Active Min", "Idle Mean",
    "Idle Max", "Idle Min",
]

# Columns shown by default in result tables (kept short so tables stay readable).
KEY_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "Packet Length Mean", "Average Packet Size",
]

# --------------------------------------------------------------------------- #
# Palette. Green/red are reserved for normal/anomaly; each model owns one hue.
# --------------------------------------------------------------------------- #
COLORS = {
    "ink": "#17232E",
    "muted": "#5B6B79",
    "paper": "#F5F7F9",
    "panel": "#FFFFFF",
    "line": "#DCE3E9",
    "grid": "#E6EBEF",
    "brand": "#1D4E89",
    "normal": "#2E9E6B",
    "anomaly": "#D6403A",
    "warn": "#B7791F",
}
FONT_STACK = "'Schibsted Grotesk', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    short: str
    color: str
    model_file: str
    threshold_file: str
    score_label: str          # axis label for the anomaly score
    score_rule: str           # one-line definition of the score
    blurb: str                # one-line description for cards


MODEL_SPECS: dict[str, ModelSpec] = {
    "autoencoder": ModelSpec(
        key="autoencoder", label="Autoencoder", short="AE", color="#1D4E89",
        model_file="autoencoder.keras", threshold_file="autoencoder_threshold.pkl",
        score_label="Reconstruction error (MSE)",
        score_rule="Mean squared error between a flow and its reconstruction.",
        blurb="A neural network squeezes each flow through an 8-unit bottleneck and rebuilds it. "
              "Flows it rebuilds badly don't look like the normal traffic it learned.",
    ),
    "isolation_forest": ModelSpec(
        key="isolation_forest", label="Isolation Forest", short="IF", color="#D9822B",
        model_file="isolation_forest.pkl", threshold_file="isolation_forest_threshold.pkl",
        score_label="Anomaly score (−decision function)",
        score_rule="Negative decision function: higher means easier to isolate, so more anomalous.",
        blurb="200 random trees try to isolate each flow. Unusual flows are cut off from the "
              "rest in fewer splits.",
    ),
    "one_class_svm": ModelSpec(
        key="one_class_svm", label="One-Class SVM", short="SVM", color="#7B4FA6",
        model_file="one_class_svm.pkl", threshold_file="one_class_svm_threshold.pkl",
        score_label="Anomaly score (−decision function)",
        score_rule="Negative decision function: higher means further outside the learned normal region.",
        blurb="An RBF-kernel boundary drawn around normal flows. Anything far outside it is "
              "flagged.",
    ),
}
MODEL_ORDER = list(MODEL_SPECS)
SCALER_FILE = "scaler.pkl"

# --------------------------------------------------------------------------- #
# Threshold sanity checks
# --------------------------------------------------------------------------- #
# Scores of an Isolation Forest are bounded: decision_function lies in [-0.5, 0.5].
# A saved threshold outside that range cannot be right for that model.
THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    "autoencoder": (0.0, math.inf),        # MSE is non-negative
    "isolation_forest": (-0.5, 0.5),
    "one_class_svm": (-math.inf, math.inf),
}
# Values printed by Project.ipynb, used only if the saved value fails the check above.
# Background: notebook cell 84 saved the same variable (`best_t`) for both the SVM and the
# Isolation Forest, so isolation_forest_threshold.pkl originally held the SVM's -16.417.
NOTEBOOK_THRESHOLDS: dict[str, float] = {
    "isolation_forest": -0.143621,
}

# --------------------------------------------------------------------------- #
# Results recorded in Project.ipynb (held-out test set, saved thresholds).
# Used as a reference so the app can show whether it reproduces the notebook.
# --------------------------------------------------------------------------- #
REFERENCE_SPLIT = {
    "train_normal": 297_310,
    "thresh_normal": 63_709,
    "thresh_attack": 127_582,
    "test_normal": 63_710,
    "test_attack": 297_694,
}
REFERENCE_RESULTS: dict[str, dict[str, float]] = {
    "autoencoder": {"Accuracy": 0.8237, "Precision": 0.8237, "Recall": 1.0000,
                    "F1-score": 0.9033, "ROC-AUC": 0.8985, "PR-AUC": 0.9757},
    "isolation_forest": {"Accuracy": 0.8803, "Precision": 0.9116, "Recall": 0.9464,
                         "F1-score": 0.9287, "ROC-AUC": 0.8390, "PR-AUC": 0.9550},
    "one_class_svm": {"Accuracy": 0.8495, "Precision": 0.8956, "Recall": 0.9251,
                      "F1-score": 0.9101, "ROC-AUC": 0.8288, "PR-AUC": 0.9600},
}
