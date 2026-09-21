"""Page 2: evaluate and compare the three models against the ground-truth labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `utils` import when run directly

import numpy as np
import pandas as pd
import streamlit as st

from utils import metrics, plots, ui
from utils.config import COLORS, DEFAULT_DATASET, LABEL_COL, MODEL_SPECS, REFERENCE_RESULTS, REFERENCE_SPLIT
from utils.data import DatasetError, stratified_positions
from utils.scoring import MODE_BALANCED, MODE_SAVED, resolve_thresholds, score_col
from utils.services import dataset_selector, get_artifacts, get_split, score_frame

EVAL_HOLDOUT = "Held-out test split"
EVAL_ACTIVE = "Every row of the active dataset"
BASELINE = "Baseline: flag every flow"

ui.setup_page("Model comparison", "📈")
ui.page_header(
    "Model comparison and evaluation",
    "How well does each model separate attacks from normal traffic, and what does each threshold cost?",
)

art = get_artifacts()
ui.sidebar_status(art)
if not ui.require_models(art):
    st.stop()

ds = dataset_selector(art)
if ds is None:
    ui.no_data_state(DEFAULT_DATASET)
    st.stop()
if not ds.has_both_classes:
    st.warning(
        "This page compares the models against ground truth, so the dataset needs an **Attack_Binary** "
        "column (0 = normal, 1 = attack) with both classes present. Upload a labelled file in the sidebar."
    )
    st.stop()

# ------------------------------------------------------------------ sidebar controls
with st.sidebar:
    st.markdown("#### Evaluation")
    eval_mode = st.radio(
        "Evaluate on", [EVAL_HOLDOUT, EVAL_ACTIVE], key="cmp_eval",
        help="**Held-out test split** recreates the notebook's test set: flows no model saw during training "
             "or threshold selection. **Every row** includes the flows the models were trained on, so scores "
             "look better than they should.")
    size = st.select_slider(
        "Flows to score", options=[10_000, 25_000, 50_000, 100_000, 0], value=25_000, key="cmp_size",
        format_func=lambda v: "Full set" if v == 0 else f"{v:,}",
        help="A stratified sample keeps the class ratio. The full set is exact but slower: in testing, "
             "scoring all three models took roughly 10 seconds per 100,000 flows on a laptop CPU, so the "
             "361,404-flow test set takes about 40 seconds the first time (results are then cached).")
    thr_mode = st.radio(
        "Thresholds", [MODE_SAVED, MODE_BALANCED], key="cmp_thr",
        help="**Saved**: the value stored with each model. **Balanced**: maximises true-positive rate minus "
             "false-positive rate. On the held-out split it is fitted on the separate threshold-selection "
             "split, so the test metrics stay honest.")

n_pick = None if size == 0 else int(size)
y_all = ds.df[LABEL_COL].to_numpy()

# ------------------------------------------------------------------ choose rows and score
calibration = None
split = None
try:
    if eval_mode == EVAL_HOLDOUT:
        split = get_split(ds)
        pos = stratified_positions(split.test_idx, y_all, n_pick)
    else:
        pos = stratified_positions(np.arange(len(ds.df)), y_all, n_pick)

    y_eval = y_all[pos]
    scores = score_frame(ds.df.iloc[pos][ds.features])

    if thr_mode == MODE_BALANCED and split is not None:
        cal_pos = stratified_positions(split.thresh_idx, y_all, n_pick)
        calibration = (score_frame(ds.df.iloc[cal_pos][ds.features]), y_all[cal_pos])
except DatasetError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Scoring failed: {exc}")
    st.stop()

thresholds = resolve_thresholds(art, scores, y_eval, thr_mode, calibration=calibration)
available = [lm.spec.key for lm in art.available_models()]

# ------------------------------------------------------------------ evaluation context
n_norm, n_att = int((y_eval == 0).sum()), int((y_eval == 1).sum())
if split is not None:
    c = split.counts
    if split.matches_notebook:
        st.success(
            f"Recreated the notebook's split exactly: {c['train_normal']:,} normal flows for training, "
            f"{c['thresh_normal'] + c['thresh_attack']:,} for threshold selection and "
            f"{c['test_normal'] + c['test_attack']:,} held out for testing.", icon="✅")
    else:
        st.warning(
            "This file gives a different split size than the notebook "
            f"(test set {c['test_normal'] + c['test_attack']:,} flows vs "
            f"{REFERENCE_SPLIT['test_normal'] + REFERENCE_SPLIT['test_attack']:,}). The same recipe is applied, "
            "but to different flows, so figures will not match the notebook.")
else:
    st.warning("Includes flows the models were trained on. Treat these numbers as optimistic; "
               "switch to the held-out test split for an unbiased comparison.")
st.caption(f"Scored **{len(y_eval):,}** flows: {n_norm:,} normal and {n_att:,} attack "
           f"({n_att / len(y_eval):.1%} attack).")

# ------------------------------------------------------------------ metrics
evals = {k: metrics.evaluate(y_eval, scores[score_col(k)].to_numpy(), thresholds[k].value) for k in available}
baseline = metrics.baseline_flag_everything(y_eval)
labels = [MODEL_SPECS[k].label for k in available]

table = pd.DataFrame(
    [{c: baseline[c] for c in metrics.METRIC_COLS}] + [{c: evals[k][c] for c in metrics.METRIC_COLS} for k in available],
    index=[BASELINE] + labels,
)

st.subheader("Metrics summary")
styled = (
    table.style.format("{:.3f}")
    .highlight_max(subset=pd.IndexSlice[labels, metrics.METRIC_COLS], axis=0,
                   props=f"font-weight:700;color:{COLORS['brand']};background-color:#E8F0FA")
    .apply(lambda r: ["color:#7B8894;font-style:italic"] * len(r) if r.name == BASELINE else [""] * len(r), axis=1)
)
st.dataframe(styled, width="stretch")
st.caption("Best value per column is highlighted. **Specificity** is the share of normal flows correctly left "
           "alone. The baseline row is a detector that flags everything, a yardstick for how much of an "
           "F1-score is just class balance.")

thr_table = pd.DataFrame({
    "Threshold": [thresholds[k].value for k in available],
    "Flows flagged": [evals[k]["flag_rate"] for k in available],
    "How it was chosen": [thresholds[k].note for k in available],
}, index=labels)
with st.expander("Thresholds used"):
    st.dataframe(thr_table, width="stretch",
                 column_config={"Threshold": st.column_config.NumberColumn(format="%.6g"),
                                "Flows flagged": st.column_config.NumberColumn(format="percent")})

with st.expander("Compare with the notebook's recorded results"):
    ref = pd.DataFrame(REFERENCE_RESULTS).T
    ref.index = [MODEL_SPECS[k].label for k in ref.index]
    ref = ref.loc[[l for l in labels if l in ref.index]]
    comparable = (split is not None and split.matches_notebook and n_pick is None and thr_mode == MODE_SAVED)
    if comparable:
        diff = table.loc[ref.index, ref.columns] - ref
        worst = float(diff.abs().to_numpy().max())
        st.markdown("Difference between this app and the notebook (app minus notebook), full test set, saved thresholds:")
        st.dataframe(diff.style.format("{:+.4f}"), width="stretch")
        (st.success if worst < 0.002 else st.warning)(
            f"Largest difference: {worst:.4f}. "
            + ("The app reproduces the notebook's results." if worst < 0.002 else
               "Larger than rounding; check that the same dataset and library versions are in use."))
    else:
        st.markdown("Values printed in Project.ipynb. A like-for-like comparison needs the **held-out test "
                    "split**, the **full set** and **saved thresholds**; small differences are expected "
                    "when scoring a sample.")
        st.dataframe(ref.style.format("{:.4f}"), width="stretch")

# ------------------------------------------------------------------ confusion matrices
st.subheader("Confusion matrices")
st.caption("Cells are shaded by the share of each actual class, so the smaller normal class stays visible; "
           "counts are printed in bold.")
cols = st.columns(len(available), gap="medium")
for col, k in zip(cols, available):
    spec = MODEL_SPECS[k]
    with col:
        st.markdown(f'<span class="swatch" style="background:{spec.color}"></span><b>{spec.label}</b>',
                    unsafe_allow_html=True)
        ui.show_fig(plots.confusion_fig(evals[k]["cm"], spec.color), key=f"cm_{k}")

# ------------------------------------------------------------------ curves
st.subheader("ROC and precision-recall curves")
roc_curves, pr_curves, roc_ops, pr_ops = {}, {}, {}, {}
for k in available:
    s = scores[score_col(k)].to_numpy()
    fpr, tpr = metrics.roc_points(y_eval, s)
    rec, prec = metrics.pr_points(y_eval, s)
    roc_curves[k] = (fpr, tpr, evals[k]["ROC-AUC"])
    pr_curves[k] = (rec, prec, evals[k]["PR-AUC"])
    op = metrics.operating_point(evals[k]["cm"])
    roc_ops[k], pr_ops[k] = (op["fpr"], op["tpr"]), (op["tpr"], op["precision"])

left, right = st.columns(2, gap="large")
with left:
    st.markdown("**ROC curves**")
    ui.show_fig(plots.roc_fig(roc_curves, roc_ops), key="roc")
with right:
    st.markdown("**Precision-recall curves**")
    ui.show_fig(plots.pr_fig(pr_curves, pr_ops, prevalence=baseline["Accuracy"]), key="pr")
st.caption("Diamonds mark each model at the threshold currently selected. A model can rank flows well "
           "(a high curve) and still be poorly calibrated (a marker far from the curve's best corner).")

# ------------------------------------------------------------------ analysis
st.subheader("Analysis")
paragraphs = metrics.build_analysis({MODEL_SPECS[k].label: evals[k] for k in available}, baseline, thr_mode)
with st.container(border=True):
    for para in paragraphs:
        st.markdown(para)
st.caption("This text is generated from the numbers above, so it changes when you change the evaluation "
           "set or thresholds.")
