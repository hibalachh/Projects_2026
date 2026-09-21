"""Entry point: project overview. Run with  `streamlit run Home.py`."""
import pandas as pd
import streamlit as st

from utils import ui
from utils.config import (
    AUTHOR, COLORS, GITHUB_URL, MODEL_ORDER, MODEL_SPECS, PORTFOLIO_URL, REFERENCE_RESULTS,
    REFERENCE_SPLIT, REPO_URL,
)
from utils.services import get_artifacts

ui.setup_page("Home", "🛡️")
art = get_artifacts()
ui.sidebar_status(art)

# ------------------------------------------------------------------ hero
left, right = st.columns([5, 6], gap="large", vertical_alignment="center")
with left:
    st.markdown('<h1 class="pg-title" style="font-size:2.7rem">Network anomaly detection</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="pg-sub" style="font-size:1.12rem">Three unsupervised models learn what normal network '
        "traffic looks like, then flag flows that don't fit. Explore their decisions, compare them on data "
        "they never saw, and inspect the raw traffic behind them.</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.page_link("pages/1_📊_Interactive_Dashboard.py", label="Open the dashboard", icon="📊", width="stretch")
    c2.page_link("pages/2_📈_Model_Comparison.py", label="Compare the models", icon="📈", width="stretch")
with right:
    st.markdown(f'<div class="hero-svg">{ui.score_strip_svg()}</div>', unsafe_allow_html=True)

st.markdown("&nbsp;")

# ------------------------------------------------------------------ project facts
st.subheader("The project")
f1, f2, f3 = st.columns(3, gap="medium")
f1.markdown(
    '<div class="panel"><h4>Objective</h4><p>Detect malicious network flows without ever showing the '
    "models an attack during training. They learn only from normal traffic, so they can also react to "
    "attack types they have never seen.</p></div>", unsafe_allow_html=True)
f2.markdown(
    '<div class="panel"><h4>Dataset</h4><p><b>CICIDS2017</b>, a benchmark of labelled network flows from '
    "the Canadian Institute for Cybersecurity. This project uses a balanced binary version: "
    "850,005 flows after removing 965 duplicates, about half normal and half attack, described by "
    "51 flow features such as duration, packet sizes and inter-arrival times.</p></div>",
    unsafe_allow_html=True)
f3.markdown(
    '<div class="panel"><h4>Tech</h4><p>TensorFlow / Keras, scikit-learn, pandas, Plotly and Streamlit. '
    "Models are serialized once and loaded by the app, and each has its own decision threshold.</p></div>",
    unsafe_allow_html=True)

st.markdown("&nbsp;")
st.subheader("How it works")
n_train = REFERENCE_SPLIT["train_normal"]
st.markdown(
    f"""
<ol class="pipe">
  <li><b>Clean</b><span>Remove duplicates and the redundant Subflow Fwd Bytes column. Outliers are kept because extreme flows can be attacks.</span></li>
  <li><b>Scale</b><span>StandardScaler fitted on normal training flows only, so nothing leaks from the test data.</span></li>
  <li><b>Train on normal traffic</b><span>{n_train:,} normal flows (70% of all normal traffic). The SVM uses a 30,000-flow subsample.</span></li>
  <li><b>Score</b><span>Each flow gets an anomaly score: reconstruction error, or a negated decision function.</span></li>
  <li><b>Apply a threshold</b><span>Above it, a flow is flagged. Thresholds are chosen on a separate split, then tested on held-out flows.</span></li>
</ol>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ models
st.markdown("&nbsp;")
st.subheader("The three models")
cols = st.columns(3, gap="medium")
for col, key in zip(cols, MODEL_ORDER):
    spec, lm = MODEL_SPECS[key], art.models[key]
    chip = '<span class="chip ok">ready</span>' if lm.available else '<span class="chip bad">unavailable</span>'
    thr = f"{lm.threshold:.6g}" if lm.threshold is not None else "not loaded"
    col.markdown(
        f'<div class="panel"><h4><span class="swatch" style="background:{spec.color}"></span>{spec.label} '
        f"&nbsp;{chip}</h4><p>{spec.blurb}</p>"
        f'<p class="muted"><b>Score:</b> {spec.score_rule}<br><b>Saved threshold:</b> {thr}</p></div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------ recorded results
st.markdown("&nbsp;")
st.subheader("Results recorded in the notebook")
st.caption(
    f"Held-out test set from Project.ipynb: {REFERENCE_SPLIT['test_normal']:,} normal and "
    f"{REFERENCE_SPLIT['test_attack']:,} attack flows, each model at its saved threshold. "
    "The Model Comparison page recomputes these live."
)
ref = pd.DataFrame(REFERENCE_RESULTS).T
ref.index = [MODEL_SPECS[k].label for k in ref.index]
st.dataframe(ref.style.format("{:.3f}").highlight_max(axis=0, props=f"font-weight:700;color:{COLORS['brand']}"),
             width="stretch")

st.info(
    "**What this project surfaced about thresholds.** The Autoencoder has the best ROC-AUC of the three, "
    "yet its saved threshold flags every flow as an attack. That threshold was chosen by maximising F1 on a "
    "validation set that is two-thirds attack traffic, and flagging everything already scores well on that "
    "measure. The dashboard lets you switch to a balanced threshold and see the difference, and the "
    "comparison page adds an “always flag” baseline so this can't hide inside an F1-score.",
    icon="💡",
)

# ------------------------------------------------------------------ footer
st.markdown("---")
links = [f"[GitHub]({GITHUB_URL})"]
if PORTFOLIO_URL:
    links.append(f"[Portfolio]({PORTFOLIO_URL})")
if REPO_URL:
    links.append(f"[Source code]({REPO_URL})")
st.markdown(f"Built by **{AUTHOR}** &nbsp;·&nbsp; " + " &nbsp;·&nbsp; ".join(links))
