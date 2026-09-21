"""Shared Streamlit UI helpers: page setup, theme CSS, KPI cards, status widgets."""
from __future__ import annotations

import html

import numpy as np
import streamlit as st

from .artifacts import Artifacts
from .config import COLORS, FONT_STACK, MODEL_ORDER, MODEL_SPECS

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');

.stApp {{ font-family: {FONT_STACK}; font-variant-numeric: tabular-nums; color: {COLORS['ink']}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: {FONT_STACK}; letter-spacing: -0.015em; }}
.stApp h1 {{ font-weight: 700; }}
.stApp h2, .stApp h3 {{ font-weight: 650; }}
.block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1320px; }}

/* page header */
.pg-title {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 .15rem 0; line-height: 1.15; }}
.pg-sub {{ color: {COLORS['muted']}; font-size: 1.02rem; margin: 0 0 1.4rem 0; max-width: 62ch; line-height: 1.5; }}

/* KPI cards */
.kpi {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; border-radius: 8px;
        padding: 14px 16px 12px 16px; height: 100%; }}
.kpi .k-label {{ color: {COLORS['muted']}; font-size: .86rem; font-weight: 500; }}
.kpi .k-value {{ font-size: 2rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; }}
.kpi .k-sub {{ color: {COLORS['muted']}; font-size: .82rem; margin-top: 2px; }}
.kpi.t-anomaly .k-value {{ color: {COLORS['anomaly']}; }}
.kpi.t-normal .k-value {{ color: {COLORS['normal']}; }}
.kpi.compact .k-value {{ font-size: 1.35rem; }}
.kpi.compact {{ padding: 10px 14px; }}

/* generic panel + chips */
.panel {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; border-radius: 8px; padding: 16px 18px; height: 100%; }}
.panel h4 {{ margin: 0 0 .35rem 0; font-size: 1.05rem; }}
.panel p {{ margin: .25rem 0; color: {COLORS['ink']}; line-height: 1.5; font-size: .95rem; }}
.panel .muted {{ color: {COLORS['muted']}; font-size: .86rem; }}
.chip {{ display: inline-block; font-size: .78rem; font-weight: 600; padding: 2px 9px; border-radius: 999px;
         border: 1px solid {COLORS['line']}; background: {COLORS['paper']}; color: {COLORS['muted']}; }}
.chip.ok {{ color: #1D6B48; background: #E6F4EC; border-color: #BFE3CE; }}
.chip.bad {{ color: #9B2C27; background: #FBE9E8; border-color: #F1C4C1; }}
.chip.warn {{ color: #7A5310; background: #FBF1DC; border-color: #EBD49C; }}
.swatch {{ display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .45rem; vertical-align: -1px; }}

/* pipeline (a genuine sequence, so it is numbered) */
.pipe {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; padding: 0; margin: 0; list-style: none; counter-reset: s; }}
.pipe li {{ counter-increment: s; background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; border-radius: 8px; padding: 12px 14px; }}
.pipe li::before {{ content: counter(s); display: inline-flex; align-items: center; justify-content: center; width: 1.5rem; height: 1.5rem;
                    border-radius: 50%; background: {COLORS['brand']}; color: white; font-size: .8rem; font-weight: 700; margin-bottom: .4rem; }}
.pipe b {{ display: block; font-size: .95rem; }}
.pipe span {{ color: {COLORS['muted']}; font-size: .85rem; line-height: 1.4; display: block; margin-top: 2px; }}
@media (max-width: 900px) {{ .pipe {{ grid-template-columns: 1fr; }} }}

.hero-svg svg {{ width: 100%; height: auto; display: block; }}
.stApp a {{ color: {COLORS['brand']}; }}
</style>
"""


def setup_page(title: str, icon: str) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} · Network anomaly detection", page_icon=icon,
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<h1 class="pg-title">{html.escape(title)}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="pg-sub">{html.escape(subtitle)}</p>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", tone: str = "", compact: bool = False) -> str:
    cls = "kpi" + (f" t-{tone}" if tone else "") + (" compact" if compact else "")
    sub_html = f'<div class="k-sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="{cls}"><div class="k-label">{html.escape(label)}</div>'
            f'<div class="k-value">{html.escape(value)}</div>{sub_html}</div>')


def kpi_row(items: list[dict], compact: bool = False) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(it["label"], it["value"], it.get("sub", ""), it.get("tone", ""), compact),
                     unsafe_allow_html=True)


def show_fig(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def require_models(art: Artifacts) -> bool:
    """Show a readable error and return False when the app cannot score anything."""
    if art.ready:
        return True
    st.error("The models could not be loaded, so there is nothing to score with yet.")
    for problem in art.problems():
        st.markdown(f"- {problem}")
    st.markdown("Place the exported files in the `models/` folder next to `Home.py` "
                "(see the **⚙️ Settings and info** page for the full list), then reload.")
    return False


def no_data_state(default_path) -> None:
    st.info(
        f"**No dataset loaded yet.** Upload a CSV in the sidebar, or place "
        f"`{default_path.name}` in the `{default_path.parent.name}/` folder to open this page with it by default."
    )
    st.markdown("The CSV needs the 51 CICIDS2017 flow-feature columns the models were trained on. "
                "An `Attack_Binary` column (0 = normal, 1 = attack) is optional; with it you also get "
                "accuracy, precision and recall.")


def sidebar_status(art: Artifacts) -> None:
    """Compact model status shown under the page navigation."""
    with st.sidebar:
        st.markdown("#### Models")
        for key in MODEL_ORDER:
            lm = art.models[key]
            spec = MODEL_SPECS[key]
            chip = '<span class="chip ok">ready</span>' if lm.available else '<span class="chip bad">unavailable</span>'
            st.markdown(f'<span class="swatch" style="background:{spec.color}"></span>{spec.label} &nbsp;{chip}',
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Hero illustration
# --------------------------------------------------------------------------- #
def score_strip_svg(seed: int = 11) -> str:
    """Two strips of dots on one score axis with a threshold line: the idea of the whole project.
    The classes deliberately overlap, because real detectors make mistakes."""
    rng = np.random.default_rng(seed)
    w, x0, x1 = 640, 24, 616
    thr_x = x0 + 0.40 * (x1 - x0)

    def dots(xs, y_center, color):
        out = []
        for x in xs:
            y = y_center + rng.normal(0, 11)
            y = float(np.clip(y, y_center - 24, y_center + 24))
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}" fill-opacity=".72"/>')
        return "".join(out)

    normal_x = x0 + (x1 - x0) * rng.beta(1.7, 8.0, 150)
    attack_x = x0 + (x1 - x0) * rng.beta(3.2, 2.6, 110)
    ink, muted, grid = COLORS["ink"], COLORS["muted"], COLORS["line"]
    return f"""
<svg viewBox="0 0 {w} 268" role="img" aria-label="Illustration: normal flows cluster at low anomaly scores and attacks at high scores, separated by a threshold line, with some overlap.">
  <rect x="0" y="0" width="{w}" height="268" rx="10" fill="{COLORS['panel']}" stroke="{grid}"/>
  <rect x="{thr_x:.1f}" y="38" width="{x1 - thr_x + 8:.1f}" height="178" fill="{COLORS['anomaly']}" fill-opacity=".05"/>
  <text x="{x0}" y="30" font-size="13" font-weight="600" fill="{ink}" font-family="{FONT_STACK}">Normal flows</text>
  {dots(normal_x, 78, COLORS['normal'])}
  <text x="{x0}" y="132" font-size="13" font-weight="600" fill="{ink}" font-family="{FONT_STACK}">Attacks</text>
  {dots(attack_x, 172, COLORS['anomaly'])}
  <line x1="{thr_x:.1f}" y1="38" x2="{thr_x:.1f}" y2="216" stroke="{ink}" stroke-width="1.6" stroke-dasharray="5 4"/>
  <text x="{thr_x + 8:.1f}" y="52" font-size="12" fill="{ink}" font-family="{FONT_STACK}">threshold: flag everything to the right</text>
  <line x1="{x0}" y1="232" x2="{x1}" y2="232" stroke="{muted}" stroke-width="1.2"/>
  <path d="M{x1 - 7} 227 L{x1} 232 L{x1 - 7} 237" fill="none" stroke="{muted}" stroke-width="1.2"/>
  <text x="{x1}" y="254" font-size="12" text-anchor="end" fill="{muted}" font-family="{FONT_STACK}">anomaly score</text>
</svg>"""
