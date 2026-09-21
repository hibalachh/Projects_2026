"""Plotly figure builders. Every function returns a ``go.Figure`` and never touches Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import COLORS, FONT_STACK, MODEL_SPECS, ModelSpec

NORMAL_NAME, ANOMALY_NAME = "Normal", "Anomaly"


def _style(fig: go.Figure, height: int = 360, legend: bool = True, **layout) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=13, color=COLORS["ink"]),
        hoverlabel=dict(font_family=FONT_STACK, bgcolor="white", bordercolor=COLORS["line"]),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        **layout,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], linecolor=COLORS["line"], zeroline=False, ticks="outside",
                     tickcolor=COLORS["line"], automargin=True, title_standoff=12)
    fig.update_yaxes(gridcolor=COLORS["grid"], linecolor=COLORS["line"], zeroline=False,
                     automargin=True, title_standoff=12)
    return fig


def _fmt(x: float) -> str:
    return f"{x:.4g}"


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
def score_trend_fig(flow_ids, scores, threshold: float, spec: ModelSpec, log_y: bool = False,
                    height: int = 380) -> go.Figure:
    """Score per flow with the decision threshold. Colour *and* marker shape encode the decision
    so the chart stays readable without relying on colour alone."""
    scores = np.asarray(scores, dtype=float)
    flow_ids = np.asarray(flow_ids)
    x = np.arange(1, len(scores) + 1)
    flagged = scores > threshold

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines", line=dict(color="#B9C5CF", width=1),
                             hoverinfo="skip", showlegend=False))
    for mask, name, color, symbol in (
        (~flagged, NORMAL_NAME, COLORS["normal"], "circle"),
        (flagged, ANOMALY_NAME, COLORS["anomaly"], "diamond"),
    ):
        fig.add_trace(go.Scatter(
            x=x[mask], y=scores[mask], mode="markers", name=name,
            marker=dict(color=color, size=6 if name == NORMAL_NAME else 7, symbol=symbol,
                        line=dict(width=0)),
            customdata=flow_ids[mask],
            hovertemplate="Flow %{customdata}<br>score %{y:.5g}<extra>" + name + "</extra>",
        ))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y", y0=threshold, y1=threshold,
                  line=dict(color=COLORS["ink"], width=1.5, dash="dash"))
    # Annotations on a log axis are positioned in log10 units, unlike shapes.
    fig.add_annotation(xref="paper", x=0.995, xanchor="right", yref="y", yanchor="bottom", showarrow=False,
                       y=float(np.log10(threshold)) if log_y else threshold,
                       text=f"threshold {_fmt(threshold)}", font=dict(size=12, color=COLORS["ink"]),
                       bgcolor="rgba(255,255,255,0.8)")
    _style(fig, height)
    fig.update_xaxes(title="Flows in the sample (file order)")
    fig.update_yaxes(title=spec.score_label, type="log" if log_y else "linear")
    return fig


def donut_fig(n_normal: int, n_anomaly: int, height: int = 380) -> go.Figure:
    total = max(n_normal + n_anomaly, 1)
    fig = go.Figure(go.Pie(
        labels=[NORMAL_NAME, ANOMALY_NAME], values=[n_normal, n_anomaly], hole=0.66, sort=False,
        marker=dict(colors=[COLORS["normal"], COLORS["anomaly"]], line=dict(color="white", width=2)),
        textinfo="percent", textfont=dict(size=13, color="white"),
        hovertemplate="%{label}: %{value:,} flows (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{n_anomaly / total:.1%}</b><br>flagged", showarrow=False,
                       font=dict(size=20, color=COLORS["ink"]))
    _style(fig, height)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5))
    return fig


def score_hist_fig(scores, y, threshold: float, spec: ModelSpec, height: int = 380) -> go.Figure:
    """Share of flows per score bin, split by actual class when labels exist."""
    scores = np.asarray(scores, dtype=float)
    use_log = spec.key == "autoencoder" and (scores > 0).all() and threshold > 0
    s = np.log10(scores) if use_log else scores
    t = np.log10(threshold) if use_log else threshold

    lo, hi = np.percentile(s, [0.5, 99.5])
    lo, hi = min(lo, t), max(hi, t)
    pad = (hi - lo) * 0.03 or 1.0
    bins = np.linspace(lo - pad, hi + pad, 61)
    centers, widths = (bins[:-1] + bins[1:]) / 2, np.diff(bins)

    groups = [("All flows", np.ones(len(s), bool), COLORS["brand"])] if y is None else [
        ("Normal (actual)", np.asarray(y) == 0, COLORS["normal"]),
        ("Attack (actual)", np.asarray(y) == 1, COLORS["anomaly"]),
    ]
    fig = go.Figure()
    for name, mask, color in groups:
        counts, _ = np.histogram(s[mask], bins=bins)
        share = counts / max(mask.sum(), 1) * 100
        fig.add_trace(go.Bar(x=centers, y=share, width=widths, name=name, marker_color=color,
                             opacity=0.65, customdata=counts,
                             hovertemplate="%{y:.2f}% of class (%{customdata:,} flows)<extra>" + name + "</extra>"))
    fig.add_vline(x=t, line_dash="dash", line_color=COLORS["ink"], line_width=1.5,
                  annotation_text="threshold", annotation_position="top")
    _style(fig, height, barmode="overlay", bargap=0)
    fig.update_xaxes(title=f"log10 of {spec.score_label.lower()}" if use_log else spec.score_label)
    fig.update_yaxes(title="Share of class (%)" if y is not None else "Share of flows (%)")
    return fig


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def confusion_fig(cm: np.ndarray, color: str, height: int = 300) -> go.Figure:
    """Heat-map coloured by the share of each *actual* class, annotated with counts,
    so the small normal class is not visually swamped by the large attack class."""
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / row_sums
    text = [[f"<b>{cm[i, j]:,}</b><br>{pct[i, j]:.1%} of row" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=pct, x=["Predicted normal", "Predicted attack"], y=["Actual normal", "Actual attack"],
        colorscale=[[0, "#FFFFFF"], [1, color]], zmin=0, zmax=1, showscale=False,
        text=text, texttemplate="%{text}", textfont=dict(size=14), hoverinfo="skip", xgap=3, ygap=3,
    ))
    _style(fig, height, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return fig


def roc_fig(curves: dict, operating: dict, height: int = 420) -> go.Figure:
    """``curves``: key -> (fpr, tpr, auc); ``operating``: key -> (fpr, tpr) at the chosen threshold."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9AA8B4", dash="dot", width=1),
                             name="Random guess", hoverinfo="skip"))
    for key, (fpr, tpr, auc) in curves.items():
        spec = MODEL_SPECS[key]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{spec.label} (AUC {auc:.3f})",
                                 line=dict(color=spec.color, width=2.5),
                                 hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra>" + spec.label + "</extra>"))
    for key, (fx, ty) in operating.items():
        spec = MODEL_SPECS[key]
        fig.add_trace(go.Scatter(x=[fx], y=[ty], mode="markers", showlegend=False,
                                 marker=dict(color=spec.color, size=12, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate=f"{spec.label} at its threshold<br>FPR %{{x:.3f}}, TPR %{{y:.3f}}<extra></extra>"))
    _style(fig, height)
    fig.update_xaxes(title="False-positive rate (normal flows flagged)", range=[0, 1])
    fig.update_yaxes(title="True-positive rate (attacks caught)", range=[0, 1.01])
    return fig


def pr_fig(curves: dict, operating: dict, prevalence: float, height: int = 420) -> go.Figure:
    """``curves``: key -> (recall, precision, ap); ``operating``: key -> (recall, precision)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                             line=dict(color="#9AA8B4", dash="dot", width=1),
                             name=f"Always flag (precision {prevalence:.2f})", hoverinfo="skip"))
    for key, (rec, prec, ap) in curves.items():
        spec = MODEL_SPECS[key]
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{spec.label} (AP {ap:.3f})",
                                 line=dict(color=spec.color, width=2.5),
                                 hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra>" + spec.label + "</extra>"))
    for key, (rx, py) in operating.items():
        spec = MODEL_SPECS[key]
        fig.add_trace(go.Scatter(x=[rx], y=[py], mode="markers", showlegend=False,
                                 marker=dict(color=spec.color, size=12, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate=f"{spec.label} at its threshold<br>Recall %{{x:.3f}}, Precision %{{y:.3f}}<extra></extra>"))
    _style(fig, height)
    fig.update_xaxes(title="Recall (attacks caught)", range=[0, 1])
    fig.update_yaxes(title="Precision (flagged flows that are attacks)", range=[0, 1.02])
    return fig


# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #
def feature_distribution_fig(df: pd.DataFrame, feature: str, label_col: str | None, bins: int = 50,
                             log_x: bool = False, trim: bool = True, percent: bool = True,
                             height: int = 400) -> tuple[go.Figure, dict]:
    """Overlaid histograms per class, computed with NumPy (fast even for 850k rows)."""
    x = df[feature].to_numpy(dtype=float)
    labels = df[label_col].to_numpy() if label_col else None
    info = {"log_dropped": 0, "trimmed": 0, "n": len(x)}

    keep = np.ones(len(x), bool)
    if log_x:
        neg = x < 0
        info["log_dropped"] = int(neg.sum())
        keep &= ~neg
    xv = np.log10(1 + x[keep]) if log_x else x[keep]
    lv = labels[keep] if labels is not None else None

    lo, hi = np.min(xv), np.max(xv)
    if trim and len(xv) > 20:
        lo, hi = np.percentile(xv, [1, 99])
        inside = (xv >= lo) & (xv <= hi)
        info["trimmed"] = int((~inside).sum())
        xv = xv[inside]
        lv = lv[inside] if lv is not None else None
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    centers, widths = (edges[:-1] + edges[1:]) / 2, np.diff(edges)

    groups = [("All flows", np.ones(len(xv), bool), COLORS["brand"])] if lv is None else [
        ("Normal", lv == 0, COLORS["normal"]), ("Attack", lv == 1, COLORS["anomaly"])]
    fig = go.Figure()
    for name, mask, color in groups:
        counts, _ = np.histogram(xv[mask], bins=edges)
        y = counts / max(counts.sum(), 1) * 100 if percent else counts
        fig.add_trace(go.Bar(x=centers, y=y, width=widths, name=name, marker_color=color, opacity=0.6,
                             customdata=counts,
                             hovertemplate="%{customdata:,} flows<extra>" + name + "</extra>"))
    _style(fig, height, barmode="overlay", bargap=0)
    fig.update_xaxes(title=("log10(1 + " + feature + ")") if log_x else feature)
    fig.update_yaxes(title="Share of class (%)" if percent else "Flows")
    return fig, info


def corr_heatmap_fig(corr: pd.DataFrame, lower_only: bool = True, height: int = 760) -> go.Figure:
    z = corr.to_numpy(copy=True)
    if lower_only:
        z[np.triu_indices_from(z, k=1)] = np.nan
    fig = go.Figure(go.Heatmap(
        z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu_r", zmid=0,
        colorbar=dict(title="r", thickness=12, len=0.8),
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    _style(fig, height, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    # dtick=1 forces a label for every feature (Plotly otherwise skips every other one).
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10), dtick=1)
    fig.update_xaxes(showgrid=False, tickangle=-60, tickfont=dict(size=10), dtick=1)
    return fig


def label_corr_fig(series: pd.Series, height: int = 420) -> go.Figure:
    s = series.sort_values()
    fig = go.Figure(go.Bar(
        x=s.to_numpy(), y=s.index, orientation="h",
        marker_color=np.where(s.to_numpy() >= 0, COLORS["anomaly"], COLORS["normal"]),
        hovertemplate="%{y}<br>r = %{x:.2f}<extra></extra>",
    ))
    _style(fig, height, legend=False)
    fig.update_xaxes(title="Correlation with Attack_Binary (Pearson r)", range=[-1, 1])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11))
    return fig
