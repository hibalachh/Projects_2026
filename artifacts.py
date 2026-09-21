"""Loading of the serialized scaler, models and thresholds.

This module is deliberately free of Streamlit imports so it can be unit-tested and
reused from scripts. Caching (``st.cache_resource``) is applied in ``services.py``.

Design goals
------------
* Each model has its *own* threshold file. Nothing is loaded from a generic
  ``threshold.pkl`` (the bug in the original ``app.py``).
* A missing or unreadable file disables that one model; the rest of the app keeps
  working and the problem is reported in plain language.
* Saved thresholds are sanity-checked against the score range of the model they
  belong to, so a mix-up between files cannot silently produce a nonsense dashboard.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib

from .config import (
    FEATURE_COLUMNS,
    MODEL_ORDER,
    MODEL_SPECS,
    MODELS_DIR,
    NOTEBOOK_THRESHOLDS,
    SCALER_FILE,
    THRESHOLD_BOUNDS,
    ModelSpec,
)


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: object | None = None
    saved_threshold: float | None = None   # exactly what the .pkl contained
    threshold: float | None = None         # value the app uses in "Saved" mode
    threshold_source: str = "missing"      # "saved" | "notebook_fallback" | "missing"
    error: str | None = None               # why the model is unavailable
    notes: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.model is not None and self.threshold is not None and self.error is None


@dataclass
class Artifacts:
    scaler: object | None
    scaler_error: str | None
    features: list[str]
    models: dict[str, LoadedModel]
    version_notes: list[str] = field(default_factory=list)

    def available_models(self) -> list[LoadedModel]:
        return [self.models[k] for k in MODEL_ORDER if self.models[k].available]

    @property
    def ready(self) -> bool:
        return self.scaler is not None and bool(self.available_models())

    def problems(self) -> list[str]:
        """Human-readable list of everything that failed to load."""
        out = []
        if self.scaler_error:
            out.append(f"Scaler: {self.scaler_error}")
        for lm in self.models.values():
            if lm.error:
                out.append(f"{lm.spec.label}: {lm.error}")
        return out


# --------------------------------------------------------------------------- #
def expected_files() -> list[Path]:
    files = [MODELS_DIR / SCALER_FILE]
    for spec in MODEL_SPECS.values():
        files += [MODELS_DIR / spec.model_file, MODELS_DIR / spec.threshold_file]
    return files


def artifact_signature() -> tuple:
    """Cheap fingerprint (name, mtime, size) used to invalidate the Streamlit cache
    when a file in ``models/`` is replaced."""
    sig = []
    for p in expected_files():
        if p.exists():
            st_ = p.stat()
            sig.append((p.name, st_.st_mtime_ns, st_.st_size))
        else:
            sig.append((p.name, None, None))
    return tuple(sig)


def _load_pickle(path: Path):
    if not path.exists():
        return None, f"file not found: models/{path.name}"
    try:
        return joblib.load(path), None
    except Exception as exc:  # corrupt file, incompatible library version, ...
        return None, f"could not read models/{path.name} ({type(exc).__name__}: {exc})"


def _load_keras(path: Path):
    if not path.exists():
        return None, f"file not found: models/{path.name}"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow start-up logs
    try:
        try:
            import keras  # Keras 3, installed together with TensorFlow
            model = keras.saving.load_model(path, compile=False)
        except ImportError:
            from tensorflow import keras as tf_keras
            model = tf_keras.models.load_model(path, compile=False)
        return model, None
    except ImportError:
        return None, "TensorFlow is not installed (pip install tensorflow)"
    except Exception as exc:
        return None, f"could not read models/{path.name} ({type(exc).__name__}: {exc})"


def _coerce_threshold(raw) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def resolve_saved_threshold(key: str, raw: float) -> tuple[float, str, list[str]]:
    """Validate a saved threshold against the score range of its model.

    Returns ``(value_to_use, source, notes)``.
    """
    lo, hi = THRESHOLD_BOUNDS.get(key, (-math.inf, math.inf))
    if lo <= raw <= hi:
        return raw, "saved", []
    fallback = NOTEBOOK_THRESHOLDS.get(key)
    label = MODEL_SPECS[key].label
    if fallback is not None:
        return fallback, "notebook_fallback", [
            f"The saved {label} threshold ({raw:.6g}) lies outside the possible score range "
            f"[{lo:g}, {hi:g}], so it cannot belong to this model. The value printed in "
            f"Project.ipynb ({fallback:g}) is used instead."
        ]
    return raw, "saved", [
        f"The saved {label} threshold ({raw:.6g}) lies outside the expected range "
        f"[{lo:g}, {hi:g}]. Check the file."
    ]


# --------------------------------------------------------------------------- #
def load_artifacts() -> Artifacts:
    """Load the scaler and every model. Never raises; failures are recorded."""
    version_notes: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        scaler, scaler_error = _load_pickle(MODELS_DIR / SCALER_FILE)
        features = list(getattr(scaler, "feature_names_in_", FEATURE_COLUMNS)) if scaler else list(FEATURE_COLUMNS)

        models: dict[str, LoadedModel] = {}
        for key in MODEL_ORDER:
            spec = MODEL_SPECS[key]
            lm = LoadedModel(spec=spec)

            if key == "autoencoder":
                lm.model, err = _load_keras(MODELS_DIR / spec.model_file)
            else:
                lm.model, err = _load_pickle(MODELS_DIR / spec.model_file)
            if err:
                lm.error = err

            raw, terr = _load_pickle(MODELS_DIR / spec.threshold_file)
            raw = _coerce_threshold(raw) if terr is None else None
            if terr is not None:
                lm.error = lm.error or terr
            elif raw is None:
                lm.error = lm.error or f"models/{spec.threshold_file} does not contain a number"
            else:
                lm.saved_threshold = raw
                lm.threshold, lm.threshold_source, notes = resolve_saved_threshold(key, raw)
                lm.notes += notes

            # Input width must match the scaler, otherwise scoring would crash later.
            if lm.model is not None and scaler is not None:
                width = _input_width(lm.model)
                if width is not None and width != len(features):
                    lm.error = f"expects {width} features but the scaler produces {len(features)}"
            models[key] = lm

    # Collapse the (very repetitive) scikit-learn version warnings into one sentence.
    seen = set()
    for w in caught:
        orig = getattr(w.message, "original_sklearn_version", None)
        curr = getattr(w.message, "current_sklearn_version", None)
        if orig and curr and (orig, curr) not in seen:
            seen.add((orig, curr))
            version_notes.append(
                f"The pickled models were created with scikit-learn {orig} but this environment "
                f"runs {curr}. Install scikit-learn=={orig} to remove the mismatch."
            )

    return Artifacts(scaler=scaler, scaler_error=scaler_error, features=features,
                     models=models, version_notes=version_notes)


def _input_width(model) -> int | None:
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    shape = getattr(model, "input_shape", None)
    if shape:
        return int(shape[-1])
    return None
