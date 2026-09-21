"""Page 3: explore the raw flow data before any model touches it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `utils` import when run directly

import numpy as np
import pandas as pd
import streamlit as st

from utils import plots, ui
from utils.config import DEFAULT_DATASET, KEY_FEATURES, LABEL_COL
from utils.data import Dataset
from utils.services import dataset_selector, get_artifacts

ui.setup_page("Data explorer", "📁")
ui.page_header("Data explorer",
               "Look at the traffic itself: how big it is, how it was cleaned, and how attacks differ from normal flows.")

art = get_artifacts()
ui.sidebar_status(art)
ds = dataset_selector(art)
if ds is None:
    ui.no_data_state(DEFAULT_DATASET)
    st.stop()

feats = ds.features
labelled = ds.has_labels
label_col = LABEL_COL if labelled else None


# ------------------------------------------------------------------ cached computations
@st.cache_data(show_spinner="Summarising features…")
def describe_features(key: str, _ds: Dataset) -> pd.DataFrame:
    d = _ds.df[_ds.features].describe().T
    d.insert(0, "dtype", _ds.df[_ds.features].dtypes.astype(str))
    d["unique"] = _ds.df[_ds.features].nunique()
    return d


@st.cache_data(show_spinner="Computing correlations…")
def correlations(key: str, method: str, n: int, _ds: Dataset):
    """Correlation matrix on a random sample (Spearman is O(n log n) per column, so keep n modest)."""
    cols = _ds.features + ([LABEL_COL] if _ds.has_labels else [])
    frame = _ds.df[cols]
    if len(frame) > n:
        frame = frame.sample(n=n, random_state=42)
    full = frame.corr(method=method)
    with_label = full[LABEL_COL].drop(LABEL_COL) if _ds.has_labels else None
    return full.loc[_ds.features, _ds.features], with_label


tab_sum, tab_table, tab_dist, tab_corr = st.tabs(
    ["Summary", "Data table", "Feature distributions", "Correlations"])

# ------------------------------------------------------------------ summary
with tab_sum:
    attack_share = f"{ds.df[LABEL_COL].mean():.1%}" if labelled else "no labels"
    ui.kpi_row([
        dict(label="Flows", value=f"{len(ds.df):,}", sub=f"from {ds.name}"),
        dict(label="Features", value=f"{len(feats)}", sub=f"{ds.df[feats].select_dtypes('integer').shape[1]} integer, "
             f"{ds.df[feats].select_dtypes('floating').shape[1]} float"),
        dict(label="Memory", value=f"{ds.memory_mb():,.0f} MB", sub="in the app's process"),
        dict(label="Attack share", value=attack_share, sub="Attack_Binary = 1" if labelled else "upload labels for more"),
    ])
    if ds.report.get("label_note"):
        st.info(ds.report["label_note"])

    st.markdown("#### Cleaning applied on load")
    r = ds.report
    st.dataframe(pd.DataFrame({
        "Step": ["Rows in the file", "Missing or infinite values removed", "Duplicate rows removed", "Rows used"],
        "Rows": [r["rows_raw"], -r["dropped_nan_inf"], -r["duplicates_removed"], r["rows_clean"]],
    }).set_index("Step"), width="stretch", column_config={"Rows": st.column_config.NumberColumn(format="%d")})
    st.caption("The same steps as the notebook. Negative values and extreme outliers are kept on purpose, "
               "because unusual flows are often the attacks.")

    if labelled:
        counts = ds.df[LABEL_COL].value_counts().rename({0: "Normal", 1: "Attack"})
        st.markdown("#### Class balance")
        st.dataframe(pd.DataFrame({"Flows": counts, "Share": counts / counts.sum()}),
                     width="stretch", column_config={"Share": st.column_config.NumberColumn(format="percent")})

    st.markdown("#### Feature statistics")
    st.dataframe(describe_features(ds.key, ds), width="stretch", height=420,
                 column_config={"count": st.column_config.NumberColumn(format="%d")})

# ------------------------------------------------------------------ data table
with tab_table:
    c1, c2, c3 = st.columns([2, 2, 3])
    n_rows = c1.slider("Rows to show", 100, 5_000, 1_000, 100, key="dx_rows")
    cls = c2.radio("Class", ["All", "Normal", "Attack"], horizontal=True, key="dx_cls") if labelled else "All"
    default_cols = ([LABEL_COL] if labelled else []) + [c for c in KEY_FEATURES if c in feats]
    cols = c3.multiselect("Columns", ([LABEL_COL] if labelled else []) + feats, default=default_cols, key="dx_cols")

    f1, f2 = st.columns([2, 3])
    fcol = f1.selectbox("Filter by feature (optional)", ["None"] + feats, key="dx_fcol")
    pct_range = (0, 100)
    if fcol != "None":
        pct_range = f2.slider(f"Keep flows between these percentiles of {fcol}", 0, 100, (0, 100), key="dx_pct")

    view = ds.df
    if cls != "All":
        view = view[view[LABEL_COL] == (1 if cls == "Attack" else 0)]
    if fcol != "None" and pct_range != (0, 100):
        lo, hi = np.percentile(view[fcol], pct_range)
        view = view[(view[fcol] >= lo) & (view[fcol] <= hi)]
    st.caption(f"{len(view):,} flows match; showing a random {min(n_rows, len(view)):,}.")
    if len(view) and cols:
        shown = view.sample(n=min(n_rows, len(view)), random_state=42).sort_index()[cols]
        shown.index.name = "Flow ID"
        st.dataframe(shown, width="stretch", height=480)
    elif not cols:
        st.info("Choose at least one column to display.")
    else:
        st.info("No flows match these filters.")

# ------------------------------------------------------------------ distributions
with tab_dist:
    top = st.columns([3, 2, 2])
    default_idx = feats.index("Flow Duration") if "Flow Duration" in feats else 0
    feature = top[0].selectbox("Feature", feats, index=default_idx, key="dx_feat")
    bins = top[1].slider("Bins", 20, 100, 50, 5, key="dx_bins")
    opt1, opt2, opt3 = st.columns(3)
    log_x = opt1.toggle("Log scale (log10 of 1 + x)", value=False, key="dx_log",
                        help="Useful for durations and byte counts, which span many orders of magnitude.")
    trim = opt2.toggle("Hide the extreme 1% at each end", value=True, key="dx_trim",
                       help="A few huge values otherwise squash the histogram into a single bar.")
    pct = opt3.toggle("Show share of each class", value=True, key="dx_pct_norm",
                      help="Normalises each class to 100% so differently sized classes compare fairly.")

    fig, info = plots.feature_distribution_fig(ds.df, feature, label_col, bins, log_x, trim, pct)
    ui.show_fig(fig, key="dx_hist")
    notes = []
    if info["log_dropped"]:
        notes.append(f"{info['log_dropped']:,} negative values can't be log-scaled and are left out.")
    if info["trimmed"]:
        notes.append(f"{info['trimmed']:,} extreme values are hidden.")
    if notes:
        st.caption(" ".join(notes))

    if labelled:
        stats = (ds.df.groupby(LABEL_COL)[feature]
                 .describe(percentiles=[0.5, 0.95]).rename(index={0: "Normal", 1: "Attack"}))
        st.markdown(f"**{feature} by class**")
        st.dataframe(stats, width="stretch")

# ------------------------------------------------------------------ correlations
with tab_corr:
    a, b, c = st.columns([2, 2, 3])
    method = a.radio("Method", ["pearson", "spearman"], horizontal=True, key="dx_method",
                     format_func=str.capitalize,
                     help="Pearson measures linear relations and is sensitive to outliers. Spearman uses ranks "
                          "and is more robust for heavy-tailed network data.")
    sample_n = b.select_slider("Rows used", options=[10_000, 25_000, 50_000, 100_000], value=50_000,
                               format_func=lambda v: f"{v:,}", key="dx_corr_n")
    lower = c.toggle("Show the lower triangle only", value=True, key="dx_lower")

    corr, with_label = correlations(ds.key, method, sample_n, ds)
    ui.show_fig(plots.corr_heatmap_fig(corr, lower_only=lower), key="dx_corr")

    st.markdown("#### Highly correlated feature pairs")
    cutoff = st.slider("Absolute correlation above", 0.5, 0.99, 0.90, 0.01, key="dx_cut")
    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    pairs = tri[tri.abs() > cutoff].sort_values(key=lambda s: -s.abs()).reset_index()
    pairs.columns = ["Feature A", "Feature B", "Correlation"]
    st.caption(f"{len(pairs)} pair{'s' if len(pairs) != 1 else ''} above {cutoff:.2f}. Highly redundant "
               "features carry the same information; the notebook removed only the exact duplicate, "
               "Subflow Fwd Bytes.")
    st.dataframe(pairs, width="stretch", height=min(420, 40 + 35 * max(len(pairs), 1)), hide_index=True,
                 column_config={"Correlation": st.column_config.NumberColumn(format="%.3f")})

    if with_label is not None:
        st.markdown("#### Features most associated with the attack label")
        strongest = with_label.reindex(with_label.abs().sort_values(ascending=False).index).head(15).dropna()
        ui.show_fig(plots.label_corr_fig(strongest), key="dx_labelcorr")
        st.caption("This is a descriptive view of the data. The models never see the label during training.")
