"""
app_utils.py - Streamlit-facing helpers shared by all pages.

core.py stays pure (no streamlit) so it remains unit-testable; this module is
the thin Streamlit layer: cached loaders, theming, plot styling, the variable
symbol system, and reusable widgets.

Typography rules:
    * General UI font: the clean default sans (set in .streamlit/config.toml).
    * Variables are rendered in Times New Roman with correct italics/subscripts:
        V_p:V_s        upright italic-free ratio, p and s as subscripts
        epsilon        italic
        R_th, P_tot    R and P italic, subscripts upright, Delta upright
        k_eq, k_eff    k italic, subscript upright
      In app text -> SYM[...] (Times-New-Roman <span>).
      In figures  -> AX[...]  (whole figure font is already Times New Roman).
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

import core

# --------------------------------------------------------------------------- #
# Light palette (dark text for contrast)
# --------------------------------------------------------------------------- #
C_PRIMARY = "#1f5fb0"   # blue   (lines, markers, headers)
C_ACCENT  = "#c0392b"   # red    (constraint)
C_OK      = "#1e8449"   # green  (feasible)
C_WARN    = "#b9770e"   # amber  (moderate)
C_PURPLE  = "#6f54b0"   # secondary series

TEXT   = "#1a2230"      # primary dark text
MUTED  = "#5a6675"      # captions / secondary
METRIC = "#1f4e79"      # metric values
PANEL  = "#ffffff"      # plot panel background (white)
GRID   = "#e1e6ee"      # subtle grid lines
AXLINE = "#3a4656"      # axis / box border
CARD_BG = "#f5f7fb"     # metric-card background
BORDER  = "#dfe4ec"     # card / element borders

FONT_TNR = "'Times New Roman', Times, serif"

PLOTLY_CONFIG = {
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d",
                               "toggleSpikelines", "hoverClosestCartesian",
                               "hoverCompareCartesian"],
}


# --------------------------------------------------------------------------- #
# Variable symbol system
# --------------------------------------------------------------------------- #
def _tnr(inner: str) -> str:
    return f'<span style="font-family:{FONT_TNR}">{inner}</span>'

# HTML snippets for use inside Streamlit markdown / labels / cards
SYM = {
    "vp_vs": _tnr("V<sub>p</sub>:V<sub>s</sub>"),
    "po":    _tnr("<i>&epsilon;</i>"),
    "r_th":  _tnr("<i>R</i><sub>th</sub>"),
    "p_tot": _tnr("&Delta;<i>P</i><sub>tot</sub>"),
    "k_eq":  _tnr("<i>k</i><sub>eq</sub>"),
    "k_eff": _tnr("<i>k</i><sub>eff</sub>"),
    "dT":    _tnr("&Delta;<i>T</i>"),
}

# Axis-title strings for figures (whole figure font is already Times New Roman)
AX = {
    "vp_vs": "V<sub>p</sub>:V<sub>s</sub>",
    "po":    "<i>\u03b5</i>  (porosity)",
    "r_th":  "<i>R</i><sub>th</sub>  (K/W)",
    "p_tot": "\u0394<i>P</i><sub>tot</sub>  (Pa)",
    "k_eq":  "<i>k</i><sub>eq</sub>  (W m\u207b\u00b9 K\u207b\u00b9)",
}


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading surrogate model\u2026")
def get_assets() -> core.Assets:
    return core.Assets()


@st.cache_data(show_spinner=False)
def cached_grid(n_vp: int, n_po: int):
    return core.evaluate_grid(get_assets(), n_vp, n_po)


@st.cache_data(show_spinner="Tracing the Pareto front\u2026")
def cached_pareto(n: int):
    return core.pareto_front(get_assets(), n=n)


# --------------------------------------------------------------------------- #
# Theming / chrome
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        f"""
        <style>
          .block-container {{ padding-top: 2rem; max-width: 1200px; }}
          /* Remove Streamlit chrome for a professional look */
          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}
          [data-testid="stStatusWidget"] {{ visibility: hidden; }}
          [data-testid="stDecoration"] {{ display: none; }}
          [data-testid="stToolbar"] {{ visibility: hidden; }}
          header[data-testid="stHeader"] {{ background: transparent; }}

          div[data-testid="stMetricValue"] {{ color: {METRIC}; font-weight: 700; }}
          div[data-testid="stMetricLabel"] {{ color: {TEXT}; }}
          .domain-badge {{ padding: 10px 14px; border-radius: 8px;
                           font-weight: 600; margin: 6px 0; }}
          .html-label {{ font-size: 0.92rem; color: {TEXT}; margin: 2px 0 -6px 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"<h2 style='color:{C_PRIMARY};margin-bottom:0;font-weight:700'>{title}</h2>"
        + (f"<p style='color:{MUTED};margin-top:4px;font-size:1.02rem'>{subtitle}</p>"
           if subtitle else ""),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
def html_label(html: str):
    """Render an HTML label (allows Times-New-Roman variables) above a widget
    whose own label is collapsed."""
    st.markdown(f"<div class='html-label'>{html}</div>", unsafe_allow_html=True)


def metric_card(label_html: str, value: str, sub: str = "", value_color: str = None):
    """A metric-style card whose label may contain Times-New-Roman variables.

    The HTML is emitted as a SINGLE line with no leading whitespace: multi-line
    indented HTML makes Streamlit's Markdown parser treat trailing tags as an
    indented code block (the stray '</div>' artifact)."""
    color = value_color or METRIC
    sub_html = (f'<div style="font-size:0.82rem;color:{MUTED};margin-top:2px">{sub}</div>'
                if sub else "")
    html = (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:10px;padding:10px 14px;margin:6px 0;">'
        f'<div style="font-size:0.9rem;color:{TEXT};">{label_html}</div>'
        f'<div style="font-size:1.7rem;font-weight:700;color:{color};'
        f'font-family:{FONT_TNR};">{value}</div>'
        f'{sub_html}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def input_sliders(default_vp=0.5, default_po=0.6, key_prefix=""):
    lo_vp, hi_vp = core.BOUNDS["vp_vs"]
    lo_po, hi_po = core.BOUNDS["po"]
    html_label(SYM["vp_vs"] + "&nbsp;&nbsp;(wick volume ratio)")
    vp = st.slider("Vp:Vs", float(lo_vp), float(hi_vp), float(default_vp),
                   step=0.005, key=f"{key_prefix}vp", label_visibility="collapsed")
    html_label(SYM["po"] + "&nbsp;&nbsp;(porosity)")
    po = st.slider("porosity", float(lo_po), float(hi_po), float(default_po),
                   step=0.005, key=f"{key_prefix}po", label_visibility="collapsed")
    return vp, po


def domain_badge(status: dict):
    color, label, msg = LEVEL_STYLE[status["level"]]
    st.markdown(
        f"<div class='domain-badge' style='background:{color}22;"
        f"border-left:5px solid {color}'>"
        f"<span style='color:{color}'>{label}</span><br>"
        f"<span style='color:{TEXT};font-weight:400'>{msg} "
        f"(\u03c3-percentile {status['pct']:.0f}, "
        f"{'inside' if status['in_hull'] else 'outside'} convex hull)</span></div>",
        unsafe_allow_html=True,
    )


LEVEL_STYLE = {
    "high":     (C_OK,     "High confidence",
                 "Inside the data hull; predictive uncertainty is low."),
    "moderate": (C_WARN,   "Moderate confidence",
                 "Near the edge of the sampled region - treat with care."),
    "low":      (C_ACCENT, "Low confidence / extrapolation",
                 "Outside the trusted region - the surrogate is guessing."),
}


# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
def base_layout(fig: go.Figure, height=480, legend_below=True):
    """Light, boxed, Times-New-Roman 2-D figure styling.

    Note: we deliberately do NOT set layout.title (setting title.font without
    title.text makes the frontend render the literal 'undefined')."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        margin=dict(l=78, r=80, t=30, b=92 if legend_below else 60),
        font=dict(family=FONT_TNR, size=14, color=TEXT),
    )
    if legend_below:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5,
            font=dict(family=FONT_TNR, size=15, color=TEXT),
            bgcolor="rgba(0,0,0,0)"))
    else:
        fig.update_layout(legend=dict(
            font=dict(family=FONT_TNR, size=15, color=TEXT),
            bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1))

    axis_common = dict(
        showline=True, linewidth=1.6, linecolor=AXLINE, mirror=True,   # full BOX
        showgrid=True, gridcolor=GRID, zeroline=False,
        title_font=dict(family=FONT_TNR, size=17, color=TEXT),
        tickfont=dict(family=FONT_TNR, size=13, color=TEXT),
        ticks="outside", tickcolor=AXLINE, ticklen=5,
    )
    fig.update_xaxes(**axis_common)
    fig.update_yaxes(**axis_common)

    for tr in fig.data:
        if tr.type in ("contour", "heatmap"):
            try:
                tr.colorbar.tickfont = dict(family=FONT_TNR, size=13, color=TEXT)
                tr.colorbar.title.font = dict(family=FONT_TNR, size=15, color=TEXT)
                tr.colorbar.outlinecolor = AXLINE
            except Exception:
                pass
    return fig


def style_scene(fig: go.Figure, height=640):
    """Light, boxed, Times-New-Roman 3-D (scene) styling."""
    def ax(title):
        return dict(
            title=dict(text=title, font=dict(family=FONT_TNR, size=15, color=TEXT)),
            backgroundcolor="#fafbfd", showbackground=True,
            gridcolor=GRID, zerolinecolor=GRID,
            tickfont=dict(family=FONT_TNR, size=12, color=TEXT),
            linecolor=AXLINE, showline=True,
        )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family=FONT_TNR, size=14, color=TEXT),
        legend=dict(orientation="h", yanchor="top", y=0.02, xanchor="center", x=0.5,
                    font=dict(family=FONT_TNR, size=15, color=TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        scene=dict(
            xaxis=ax(AX["vp_vs"]),
            yaxis=ax(AX["po"]),
            camera=dict(eye=dict(x=1.55, y=-1.55, z=1.0)),
        ),
    )
    return fig


def add_training_points(fig: go.Figure, assets, marker_color=C_PRIMARY):
    fig.add_trace(go.Scatter(
        x=assets.X_train[:, 0], y=assets.X_train[:, 1], mode="markers",
        marker=dict(size=6, color=marker_color, symbol="circle-open",
                    line=dict(width=1.6)),
        name="FEM samples (n=49)",
        hovertemplate="V<sub>p</sub>:V<sub>s</sub>=%{x:.3f}<br><i>\u03b5</i>=%{y:.3f}<extra></extra>",
    ))
    return fig
