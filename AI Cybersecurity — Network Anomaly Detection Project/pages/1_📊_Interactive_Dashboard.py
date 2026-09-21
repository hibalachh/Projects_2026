"""Page 1: choose a model and see how it labels each network flow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `utils` import when run directly

import numpy as np
import pandas as pd
import streamlit as st

from utils import metrics, plots, ui
from utils.config import DEFAULT_DATASET, KEY_FEATURES, LABEL_COL, MODEL_SPECS
from utils.scoring import MODE_BALANCED, MODE_PERCENT, MODE_SAVED, predict, resolve_thresholds, score_col
from utils.services import dataset_selector, get_artifacts, score_frame

MAX_TABLE_ROWS = 50_000

ui.setup_page("Interactive dashboard", "📊")
ui.page_header("Interactive dashboard", "Choose a model and see how it labels each network flow.")

art = get_artifacts()
ui.sidebar_status(art)
if not ui.require_models(art):
    st.stop()

ds = dataset_selector(art)
if ds is None:
    ui.no_data_state(DEFAULT_DATASET)
    st.stop()

# ------------------------------------------------------------------ sidebar controls
with st.sidebar:
    st.markdown("#### Analysis")
    base = [5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
    size_opts = sorted({o for o in base if o < len(ds.df)} | {min(len(ds.df), 200_000)})
    if len(size_opts) > 1:
        n_flows = st.select_slider(
            "Flows to analyse", options=size_opts, value=20_000 if 20_000 in size_opts else size_opts[-1],
            format_func=lambda v: f"{v:,}",
            help="A random sample keeps scoring fast. The same sample is used every time.")
    else:
        n_flows = size_opts[0]

    modes = [MODE_SAVED] + ([MODE_BALANCED] if ds.has_both_classes else []) + [MODE_PERCENT]
    if st.session_state.get("dash_mode") not in modes:
        st.session_state["dash_mode"] = MODE_SAVED
    mode = st.radio(
        "Threshold", modes, key="dash_mode",
        help="**Saved**: the value stored with each model. **Balanced**: maximises true-positive rate minus "
             "false-positive rate on this sample (needs labels). **Top %**: flag a fixed share of flows.")
    percent = 20.0
    if mode == MODE_PERCENT:
        percent = st.slider("Flag the top … % of flows", 0.5, 60.0, 20.0, 0.5)

# ------------------------------------------------------------------ scoring
sample = ds.sample(n_flows)
try:
    scores_df = score_frame(sample[ds.features])
except Exception as exc:  # scoring problems should never produce a raw traceback for the viewer
    st.error(f"Scoring failed: {exc}")
    st.stop()

y = sample[LABEL_COL].to_numpy() if ds.has_labels else None
thresholds = resolve_thresholds(art, scores_df, y, mode, percent)
available = [lm.spec.key for lm in art.available_models()]

if st.session_state.get("dash_model") not in available:
    st.session_state["dash_model"] = available[0]
chosen = st.segmented_control(
    "Model", available, format_func=lambda k: MODEL_SPECS[k].label, key="dash_model", selection_mode="single")
key = chosen or available[0]
spec, lm = MODEL_SPECS[key], art.models[key]

score = scores_df[score_col(key)].to_numpy()
choice = thresholds[key]
thr = choice.value
pred = predict(score, thr)
total, n_anom = len(pred), int(pred.sum())
n_norm = total - n_anom
rate = n_anom / total

# Number of models that flag each flow (each judged at its own threshold under the current mode).
consensus = sum(
    (scores_df[score_col(k)].to_numpy() > thresholds[k].value).astype(int) for k in available
)

# ------------------------------------------------------------------ KPI cards
ui.kpi_row([
    dict(label="Total flows", value=f"{total:,}",
         sub=f"sampled from {len(ds.df):,}" if total < len(ds.df) else "the whole dataset"),
    dict(label="Anomalies detected", value=f"{n_anom:,}", sub=f"score above {thr:.4g}", tone="anomaly"),
    dict(label="Detection rate", value=f"{rate:.1%}", sub="share of flows flagged"),
    dict(label="Normal traffic", value=f"{n_norm:,}", sub=f"{n_norm / total:.1%} of flows", tone="normal"),
])
st.caption(f"**{spec.label}** · threshold {thr:.6g} · {choice.note}")

if y is not None:
    ev = metrics.evaluate(y, score, thr)
    tn, fp, fn, tp = ev["cm"].ravel()
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    ui.kpi_row([
        dict(label="Actual attacks in sample", value=f"{int(y.sum()):,}", sub=f"{y.mean():.1%} of flows"),
        dict(label="Attacks caught (recall)", value=f"{ev['Recall']:.1%}", sub=f"{tp:,} of {tp + fn:,}"),
        dict(label="Flags that are attacks (precision)", value=f"{ev['Precision']:.1%}", sub=f"{tp:,} of {tp + fp:,}"),
        dict(label="Normal flows wrongly flagged", value=f"{1 - ev['Specificity']:.1%}", sub=f"{fp:,} of {fp + tn:,}"),
    ], compact=True)
    st.caption("This file includes flows the models were trained on, so these figures are optimistic. "
               "The **Model Comparison** page evaluates on the held-out test split.")

# ------------------------------------------------------------------ threshold warnings
if mode == MODE_SAVED and lm.notes:
    for note in lm.notes:
        st.info(note)
if rate >= 0.99 or rate <= 0.005:
    what = "nearly every flow" if rate >= 0.99 else "almost no flows"
    st.warning(
        f"At this threshold the {spec.label} flags {what} ({rate:.1%}), so the charts can't separate "
        f"normal from abnormal traffic. The model's scores may still be informative; the cut-off is what needs changing."
    )
    if ds.has_both_classes and mode != MODE_BALANCED:
        st.button("Switch to a balanced threshold",
                  on_click=lambda: st.session_state.update(dash_mode=MODE_BALANCED))
    elif mode != MODE_PERCENT:
        st.caption("Try the “Top % of scores” threshold in the sidebar.")

st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ charts, row 1
col_a, col_b = st.columns([7, 5], gap="large")
with col_a:
    st.subheader("Reconstruction error trend" if key == "autoencoder" else "Anomaly score trend")
    ctl1, ctl2 = st.columns([3, 2])
    if total > 100:
        show_n = ctl1.slider("Flows shown", 100, min(1500, total), min(500, total), 50, key="trend_n",
                             help="A random subset of the sample, drawn in file order.")
    else:
        show_n = total
    can_log = bool((score > 0).all() and thr > 0)
    log_y = ctl2.toggle("Log scale", value=(key == "autoencoder" and can_log), disabled=not can_log,
                        key=f"log_{key}", help="Only available when every score is positive.")
    pick = np.sort(np.random.default_rng(1).choice(total, size=show_n, replace=False))
    ui.show_fig(plots.score_trend_fig(sample.index.to_numpy()[pick], score[pick], thr, spec, log_y))
with col_b:
    st.subheader("Traffic distribution")
    ui.show_fig(plots.donut_fig(n_norm, n_anom))

# ------------------------------------------------------------------ results frame
res = pd.DataFrame({
    "Prediction": np.where(pred == 1, "🔴 Anomaly", "🟢 Normal"),
    "Score": score,
    "Models flagging": consensus,
}, index=sample.index)
res.index.name = "Flow ID"
if y is not None:
    res["Actual"] = np.where(y == 1, "Attack", "Normal")

# ------------------------------------------------------------------ charts, row 2
col_c, col_d = st.columns(2, gap="large")
with col_c:
    st.subheader("Top anomalies")
    st.caption(f"The 10 flows with the highest {spec.label} score.")
    peek = [c for c in KEY_FEATURES if c in sample.columns][:5]
    top = pd.concat([res.drop(columns="Prediction"), sample[peek]], axis=1).nlargest(10, "Score")
    st.dataframe(top, width="stretch", height=390,
                 column_config={"Score": st.column_config.NumberColumn(format="%.5g")})
with col_d:
    st.subheader("Score distribution")
    st.caption("Where normal and attack flows fall on the score axis." if y is not None
               else "Where the flows fall on the score axis.")
    ui.show_fig(plots.score_hist_fig(score, y, thr, spec))

# ------------------------------------------------------------------ results table
st.subheader("Flow results")
f1, f2 = st.columns([3, 2])
view = f1.radio("Show", ["All flows", "Anomalies only", "Normal only"], horizontal=True, key="tbl_view")
actual = "All"
if y is not None:
    actual = f2.radio("Actual class", ["All", "Attack", "Normal"], horizontal=True, key="tbl_actual")
default_cols = [c for c in KEY_FEATURES if c in sample.columns]
shown_cols = st.multiselect("Feature columns", ds.features, default=default_cols, key="tbl_cols")

table = pd.concat([res, sample[shown_cols]], axis=1)
if view == "Anomalies only":
    table = table[pred == 1]
elif view == "Normal only":
    table = table[pred == 0]
if actual != "All":
    table = table[table["Actual"] == actual]
table = table.sort_values("Score", ascending=False)

st.caption(f"{len(table):,} of {total:,} flows match. Click a column header to re-sort"
           + (f"; the first {MAX_TABLE_ROWS:,} are shown." if len(table) > MAX_TABLE_ROWS else "."))
st.dataframe(
    table.head(MAX_TABLE_ROWS), width="stretch", height=460,
    column_config={
        "Score": st.column_config.NumberColumn(format="%.5g", help=spec.score_rule),
        "Models flagging": st.column_config.NumberColumn(
            help=f"How many of the {len(available)} models flag this flow at their own threshold."),
    },
)
st.download_button("Download these rows (CSV)", table.to_csv().encode("utf-8"),
                   file_name=f"flow_results_{key}.csv", mime="text/csv")
