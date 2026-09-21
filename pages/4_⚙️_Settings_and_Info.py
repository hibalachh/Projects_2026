"""Page 4: model artifacts, thresholds, parameters, environment and background concepts."""
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `utils` import when run directly

import numpy as np
import pandas as pd
import streamlit as st

from utils import ui
from utils.artifacts import expected_files
from utils.config import DEFAULT_DATASET, MODEL_ORDER, MODEL_SPECS, MODELS_DIR, SCALER_FILE, THRESHOLD_BOUNDS
from utils.services import get_artifacts

ui.setup_page("Settings and info", "⚙️")
ui.page_header("Settings and info",
               "What is loaded, which thresholds are in use, and the ideas behind the three models.")

art = get_artifacts()
ui.sidebar_status(art)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


tab_art, tab_thr, tab_par, tab_concepts, tab_app = st.tabs(
    ["Model files", "Thresholds", "Parameters", "How the models work", "App and environment"])

# ------------------------------------------------------------------ model files
with tab_art:
    st.markdown(f"Contents of `{MODELS_DIR.name}/`. Each file is loaded once when the app starts and reloaded "
                "automatically if it changes.")

    roles = {SCALER_FILE: ("Feature scaler", None)}
    for k, spec in MODEL_SPECS.items():
        roles[spec.model_file] = (f"{spec.label} model", k)
        roles[spec.threshold_file] = (f"{spec.label} threshold", k)

    rows = []
    present = {p.name: p for p in MODELS_DIR.glob("*")} if MODELS_DIR.exists() else {}
    for name in sorted(set(present) | {p.name for p in expected_files()}):
        role, key = roles.get(name, ("Not used by the app", None))
        path = present.get(name)
        if path is None:
            status, size, modified = "❌ missing", "", ""
        else:
            stt = path.stat()
            size, modified = human_size(stt.st_size), datetime.fromtimestamp(stt.st_mtime).strftime("%Y-%m-%d %H:%M")
            if name == SCALER_FILE:
                status = "✅ loaded" if art.scaler is not None else "❌ failed"
            elif key is None:
                status = "—"
            elif name == MODEL_SPECS[key].model_file:
                status = "✅ loaded" if art.models[key].model is not None else "❌ failed"
            else:
                status = "✅ loaded" if art.models[key].saved_threshold is not None else "❌ failed"
        rows.append({"File": name, "Role": role, "Size": size, "Modified": modified, "Status": status})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    for problem in art.problems():
        st.error(problem)
    for note in art.version_notes:
        st.warning(note)
    for lm in art.models.values():
        for note in lm.notes:
            st.warning(note)
    if art.ready and not (art.problems() or art.version_notes or any(lm.notes for lm in art.models.values())):
        st.success("Every expected file loaded without warnings.")

    if art.scaler is not None:
        with st.expander(f"Expected input: {len(art.features)} features, in this order"):
            st.dataframe(pd.DataFrame({
                "Feature": art.features,
                "Training mean": getattr(art.scaler, "mean_", np.nan),
                "Training std": getattr(art.scaler, "scale_", np.nan),
            }), width="stretch", height=380, hide_index=True)

# ------------------------------------------------------------------ thresholds
with tab_thr:
    st.markdown(
        "A flow is flagged when its score is **above** the threshold. Each model has its own threshold file, "
        "chosen in the notebook on a separate split and stored next to the model.")
    trows = []
    for key in MODEL_ORDER:
        lm, spec = art.models[key], MODEL_SPECS[key]
        lo, hi = THRESHOLD_BOUNDS[key]
        trows.append({
            "Model": spec.label,
            "Stored in .pkl": lm.saved_threshold,
            "Used in “Saved” mode": lm.threshold,
            "Source": {"saved": "the .pkl file", "notebook_fallback": "notebook value (file invalid)",
                       "missing": "not available"}[lm.threshold_source],
            "Valid range": "any" if np.isinf(lo) and np.isinf(hi) else f"{lo:g} to {'∞' if np.isinf(hi) else f'{hi:g}'}",
            "Score": spec.score_rule,
        })
    st.dataframe(pd.DataFrame(trows).set_index("Model"), width="stretch",
                 column_config={"Stored in .pkl": st.column_config.NumberColumn(format="%.6g"),
                                "Used in “Saved” mode": st.column_config.NumberColumn(format="%.6g")})
    st.caption("The app checks each saved value against the range its model can actually produce, so a threshold "
               "that was saved for a different model is caught instead of silently used.")

    st.markdown("#### Threshold modes in the app")
    st.markdown(
        "- **Saved threshold**: the value from the notebook.\n"
        "- **Balanced (Youden's J)**: the score that maximises the true-positive rate minus the false-positive "
        "rate. It treats catching attacks and sparing normal flows as equally important and is not affected by "
        "how many flows of each class there are.\n"
        "- **Top % of scores**: flags a fixed share of the highest-scoring flows, which needs no labels.")
    st.markdown(
        "**Why the choice matters.** The notebook picked thresholds by maximising F1 on a split that is "
        "about two-thirds attack traffic. On such a split, flagging every flow already gives an F1-score near "
        "0.80, so the search can settle on a threshold that is meaningless. Balanced thresholds avoid that.")

# ------------------------------------------------------------------ parameters
with tab_par:
    if art.scaler is not None:
        st.markdown("#### Scaler")
        seen = getattr(art.scaler, "n_samples_seen_", None)
        seen_txt = f"{int(np.max(seen)):,}" if seen is not None else "an unknown number of"
        st.write(f"`{type(art.scaler).__name__}` fitted on **{seen_txt}** normal training flows, "
                 f"{art.scaler.n_features_in_} features (zero mean and unit variance).")
    for key in MODEL_ORDER:
        lm, spec = art.models[key], MODEL_SPECS[key]
        st.markdown(f"#### {spec.label}")
        if lm.model is None:
            st.warning(lm.error or "Not loaded.")
            continue
        if key == "autoencoder":
            layer_rows = []
            for layer in lm.model.layers:
                act = getattr(getattr(layer, "activation", None), "__name__", "")
                layer_rows.append({"Layer": layer.name, "Type": type(layer).__name__,
                                   "Units": getattr(layer, "units", ""), "Activation": act,
                                   "Parameters": layer.count_params()})
            st.dataframe(pd.DataFrame(layer_rows), width="stretch", hide_index=True)
            st.caption(f"{lm.model.count_params():,} parameters in total. Loss: mean squared error; "
                       "optimizer: Adam; training stopped early on validation loss.")
        else:
            params = lm.model.get_params()
            extra = {}
            if key == "isolation_forest":
                extra = {"trees_in_forest": len(lm.model.estimators_), "max_samples_": lm.model.max_samples_,
                         "offset_": lm.model.offset_}
            else:
                extra = {"support_vectors": int(lm.model.support_vectors_.shape[0]),
                         "gamma_ (resolved from 'scale')": float(getattr(lm.model, "_gamma", np.nan))}
            st.json({"hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                         for k, v in params.items()},
                     "fitted": {k: (v if isinstance(v, (int, str)) else float(v)) for k, v in extra.items()}},
                    expanded=False)

# ------------------------------------------------------------------ concepts
with tab_concepts:
    sv = art.models["one_class_svm"]
    n_sv = f"{sv.model.support_vectors_.shape[0]:,}" if sv.model is not None else "about a thousand"
    c1, c2, c3, c4 = st.tabs(["Autoencoder", "Isolation Forest", "One-Class SVM", "Learning from normal traffic"])
    with c1:
        st.markdown(f"""
**The idea.** A neural network is trained to copy its input to its output, but it has to squeeze the data
through a narrow middle layer first. Trained only on normal traffic, it becomes good at compressing and
rebuilding *normal* flows. An attack looks different, so it is rebuilt badly.

**In this project.** 51 features go in and pass through layers of 32, 16 and 8 units (the bottleneck) and back
up through 16 and 32 to 51 outputs. Hidden layers use ReLU, the output is linear, and the loss is mean squared
error.

**The score.** The average squared difference between the 51 scaled input values and their reconstruction.
Larger means the flow is harder to explain as normal traffic.

**Strengths.** Captures non-linear relationships between features and gives a per-flow score that is easy to
explain. **Limits.** Heavy-tailed features produce very large errors for extreme flows, it needs a threshold,
and a model that generalises too well can reconstruct some attacks too.""")
    with c2:
        st.markdown("""
**The idea.** Build many random trees that split the data on random features at random values. A point that is
very different from the rest gets separated from everything else after only a few splits, while a typical point
needs many. The average path length across trees becomes the score.

**In this project.** 200 trees fitted on the scaled normal training flows, with scikit-learn's automatic
contamination setting.

**The score.** The negative of `decision_function`, which always lies between -0.5 and 0.5. scikit-learn's own
cut-off would be 0; higher means easier to isolate, so more anomalous.

**Strengths.** Fast, scales to large datasets and does not compute distances. **Limits.** Its splits are
aligned to single features, so anomalies that only show up as an unusual *combination* of features are harder
to catch.""")
    with c3:
        st.markdown(f"""
**The idea.** Map the data into a high-dimensional space with a kernel and find the smallest boundary that
encloses most of the normal data. Anything far outside the boundary is an anomaly.

**In this project.** An RBF kernel with `nu = 0.05`, which caps the fraction of training points allowed outside
the boundary at about 5%. Training cost grows quickly with the number of flows, so it was fitted on a random
subsample of 30,000 normal flows and ended up with {n_sv} support vectors.

**The score.** The negative of `decision_function`: the signed distance from the boundary, positive outside.
It is not bounded, so its scale differs greatly from the other two models.

**Strengths.** Flexible, smooth boundary. **Limits.** Slow to train on large data, and sensitive to the kernel
width and to `nu`.""")
    with c4:
        st.markdown("""
**Why train on normal traffic only?** New attack types appear constantly. A model that has only learned what
*normal* looks like can flag things it has never seen, where a classifier trained on known attacks would miss
them.

**The trade-off.** Nothing tells these models what an attack looks like, so anything unusual gets flagged,
including harmless rarities. Expect more false alarms than from a supervised classifier.

**How the data was split.**
- Normal flows: 70% to train and fit the scaler, 15% to choose thresholds, 15% to test.
- Attack flows: 30% to choose thresholds, 70% to test.
- The scaler is fitted on training flows only so no information leaks from the test set.""")

# ------------------------------------------------------------------ app + environment
with tab_app:
    st.markdown("#### Data")
    exists = DEFAULT_DATASET.exists()
    st.write(f"Default dataset: `{DEFAULT_DATASET}` " + ("✅ found" if exists else "❌ not found"))
    if exists:
        st.caption(f"{human_size(DEFAULT_DATASET.stat().st_size)}. Change the location with the "
                   "`ANOMALY_DATASET_PATH` environment variable.")
    else:
        st.caption("Place the CSV there, or upload a file from any data page. Uploads larger than 200 MB need "
                   "`maxUploadSize` in `.streamlit/config.toml` (already set to 1024 MB in this project).")

    st.markdown("#### Environment")
    pkgs = ["streamlit", "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "plotly", "joblib"]
    env_rows = [{"Component": "Python", "Version": platform.python_version()}]
    for p in pkgs:
        try:
            env_rows.append({"Component": p, "Version": metadata.version(p)})
        except metadata.PackageNotFoundError:
            env_rows.append({"Component": p, "Version": "not installed"})
    st.dataframe(pd.DataFrame(env_rows), width="stretch", hide_index=True)

    st.markdown("#### Cache")
    st.caption("Models, datasets and scores are cached for speed. Clear the cache after replacing files.")
    if st.button("Clear caches and reload"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for k in ("upload_ds", "upload_fp", "upload_error", "upload_dismissed"):
            st.session_state.pop(k, None)
        st.rerun()
