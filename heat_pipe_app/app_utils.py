"""
app_utils.py - Streamlit-facing helpers shared by all pages.

core.py stays pure (no streamlit) so it remains unit-testable; this module is
the thin Streamlit layer: cached loaders, theming, plot styling, the variable
symbol system, and reusable widgets.

THEME: the light theme is FORCED via CSS in inject_css() so it applies even if
.streamlit/config.toml is missing or unread on the host (the failure observed
on Streamlit Cloud when the file isn't at the repository root). Figures use an
opaque WHITE paper so axis labels/legends are readable on any page background.

TYPOGRAPHY: general UI font is the default sans; variables render in Times New
Roman with correct italics/subscripts:
    SYM[...]  -> HTML spans for app text (V_p:V_s, italic eps, R_th, dP_tot, k_eq)
    AX[...]   -> figure axis-title strings (figures are wholly Times New Roman)
    U[...]    -> plain-unicode fallbacks for widget labels that cannot render
                 HTML (e.g. st.radio options)
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

import core

# --------------------------------------------------------------------------- #
# Light palette (dark text for contrast)
# --------------------------------------------------------------------------- #
C_PRIMARY = "#1f5fb0"
C_ACCENT  = "#c0392b"
C_OK      = "#1e8449"
C_WARN    = "#b9770e"
C_PURPLE  = "#6f54b0"

TEXT   = "#1a2230"
MUTED  = "#5a6675"
METRIC = "#1f4e79"
PAPER  = "#ffffff"      # figure canvas - opaque white, never transparent
PANEL  = "#ffffff"      # plot panel background
GRID   = "#e1e6ee"
AXLINE = "#3a4656"
CARD_BG = "#f5f7fb"
BORDER  = "#dfe4ec"
BG      = "#ffffff"     # app background
BG_SIDE = "#eef2f8"     # sidebar background

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

SYM = {
    "vp_vs": _tnr("V<sub>p</sub>:V<sub>s</sub>"),
    "po":    _tnr("<i>&epsilon;</i>"),
    "r_th":  _tnr("<i>R</i><sub>th</sub>"),
    "p_tot": _tnr("&Delta;<i>P</i><sub>tot</sub>"),
    "k_eq":  _tnr("<i>k</i><sub>eq</sub>"),
    "k_eff": _tnr("<i>k</i><sub>eff</sub>"),
    "dT":    _tnr("&Delta;<i>T</i>"),
}

AX = {
    "vp_vs": "V<sub>p</sub>:V<sub>s</sub>",
    "po":    "<i>\u03b5</i>  (porosity)",
    "r_th":  "<i>R</i><sub>th</sub>  (K/W)",
    "p_tot": "\u0394<i>P</i><sub>tot</sub>  (Pa)",
    "k_eq":  "<i>k</i><sub>eq</sub>  (W m\u207b\u00b9 K\u207b\u00b9)",
    "sigma": "\u03c3(<i>R</i><sub>th</sub>)  (K/W)",
}

# Plain-unicode labels for widgets that cannot render HTML (radio options etc.)
U = {
    "r_th":  "R\u209c\u2095",
    "p_tot": "\u0394P\u209c\u2092\u209c",
    "sigma": "\u03c3(R\u209c\u2095) model uncertainty",
    "vp_vs": "V\u209a:V\u209b",
    "po":    "\u03b5",
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


@st.cache_data(show_spinner=False)
def cached_sigma_grid(n_vp: int, n_po: int):
    """Grid of the GPR predictive std of R_th in physical units (K/W)."""
    A = get_assets()
    gx = np.linspace(*core.BOUNDS["vp_vs"], n_vp)
    gy = np.linspace(*core.BOUNDS["po"], n_po)
    VP, PO = np.meshgrid(gx, gy)
    pts = np.column_stack([VP.ravel(), PO.ravel()])
    u = core.predict_with_uncertainty(A, pts, k=1.0)
    SIG = u["r_th_sigma"].reshape(VP.shape)
    return VP, PO, SIG


@st.cache_data(show_spinner="Tracing the Pareto front\u2026")
def cached_pareto(n: int):
    return core.pareto_front(get_assets(), n=n)


# --------------------------------------------------------------------------- #
# Theming / chrome  (FORCED light theme - works without config.toml)
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(
        f"""
        <style>
          /* ---------- forced light theme (independent of config.toml) ----- */
          .stApp, [data-testid="stAppViewContainer"],
          [data-testid="stMain"] {{ background-color: {BG} !important; }}
          [data-testid="stHeader"] {{ background-color: {BG} !important; }}
          [data-testid="stSidebar"] {{ background-color: {BG_SIDE} !important; }}
          [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
          [data-testid="stSidebarNav"] a span {{ color: {TEXT} !important; }}

          /* all body text dark */
          .stApp p, .stApp span, .stApp label, .stApp li, .stApp td, .stApp th,
          .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
          [data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] *
          {{ color: {TEXT}; }}
          [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *
          {{ color: {MUTED} !important; }}

          /* ---------- widgets: number/text inputs (white field, dark text) -- */
          .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
          .stApp [data-baseweb="input"] > div, .stApp [data-baseweb="select"] > div
          {{ background-color: #ffffff !important; border-color: #c9d2de !important; }}
          .stApp [data-baseweb="input"] input,
          .stApp input[type="number"], .stApp input[type="text"]
          {{ background-color: #ffffff !important; color: {TEXT} !important;
             -webkit-text-fill-color: {TEXT} !important; caret-color: {TEXT}; }}
          .stApp [data-baseweb="select"] * {{ color: {TEXT} !important; }}
          [data-testid="stNumberInput"] button
          {{ background-color: #eef2f8 !important; color: {TEXT} !important;
             border: 1px solid #c9d2de !important; }}
          [data-testid="stNumberInput"] button svg {{ fill: {TEXT} !important; }}

          /* ---------- sliders: thumb value + min/max ticks, in TNR --------- */
          [data-testid="stSliderThumbValue"]
          {{ color: {C_ACCENT} !important; font-family: {FONT_TNR} !important; }}
          [data-testid="stTickBar"] *
          {{ color: #44505f !important; font-family: {FONT_TNR} !important; }}

          /* ---------- radio buttons: visible circles, TNR option text ------ */
          [data-baseweb="radio"] > div:first-child
          {{ background-color: #ffffff !important;
             border: 2px solid #8b97a6 !important; }}
          [data-testid="stRadio"] label p
          {{ color: {TEXT} !important; font-family: {FONT_TNR} !important;
             font-size: 1.0rem; }}

          /* ---------- checkboxes: white unchecked, accent checked ---------- */
          [data-baseweb="checkbox"] > span:first-of-type
          {{ background-color: #ffffff !important;
             border: 2px solid #8b97a6 !important; }}
          [data-baseweb="checkbox"][aria-checked="true"] > span:first-of-type,
          [data-baseweb="checkbox"] input:checked + span
          {{ background-color: {C_ACCENT} !important;
             border-color: {C_ACCENT} !important; }}
          [data-testid="stCheckbox"] label p {{ color: {TEXT} !important; }}

          /* code chips */
          code {{ background-color: #eef2f8 !important; color: #b02a50 !important; }}

          /* alerts (info/success/warning/error) -> light card, dark text */
          [data-testid="stAlert"]
          {{ background-color: #eef3fa !important; border: 1px solid {BORDER}; }}
          [data-testid="stAlert"] * {{ color: {TEXT} !important; }}

          /* expanders */
          [data-testid="stExpander"]
          {{ background-color: #ffffff; border: 1px solid {BORDER}; border-radius: 8px; }}
          [data-testid="stExpander"] summary span {{ color: {TEXT} !important; }}

          .block-container {{ padding-top: 2rem; max-width: 1200px; }}

          /* hide Streamlit chrome */
          #MainMenu {{ visibility: hidden; }}
          footer {{ visibility: hidden; }}
          [data-testid="stStatusWidget"] {{ visibility: hidden; }}
          [data-testid="stDecoration"] {{ display: none; }}
          [data-testid="stToolbar"] {{ visibility: hidden; }}

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
    st.markdown(f"<div class='html-label'>{html}</div>", unsafe_allow_html=True)


def metric_card(label_html: str, value: str, sub: str = "", value_color: str = None):
    """A metric-style card; HTML emitted as a SINGLE line (multi-line indented
    HTML makes the Markdown parser print trailing tags as a code block)."""
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


def synced_input(label_html_str, lo, hi, default, step, key, fmt="%.3f"):
    """A slider AND a number box bound to the same value.

    - The slider shows its EXACT value (format=fmt matches the step
      granularity, fixing the 'thumb label does not change' issue).
    - The number box lets the user type a raw value directly; it is clipped to
      the bounds. Whichever control moved last wins, via on_change callbacks.
    """
    skey, nkey = f"{key}_sl", f"{key}_nb"
    if skey not in st.session_state:
        st.session_state[skey] = float(default)
    if nkey not in st.session_state:
        st.session_state[nkey] = float(default)

    def _from_slider():
        st.session_state[nkey] = st.session_state[skey]

    def _from_number():
        v = float(np.clip(st.session_state[nkey], lo, hi))
        st.session_state[nkey] = v
        st.session_state[skey] = v

    if label_html_str:
        html_label(label_html_str)
    c1, c2 = st.columns([2.6, 1])
    with c1:
        st.slider(f"{key} (slider)", float(lo), float(hi), step=float(step),
                  format=fmt, key=skey, on_change=_from_slider,
                  label_visibility="collapsed")
    with c2:
        st.number_input(f"{key} (direct input)", float(lo), float(hi),
                        step=float(step), format=fmt, key=nkey,
                        on_change=_from_number, label_visibility="collapsed")
    return float(st.session_state[skey])


def input_sliders(default_vp=0.5, default_po=0.6, key_prefix=""):
    """The two design-variable controls: synced slider + direct-input box."""
    lo_vp, hi_vp = core.BOUNDS["vp_vs"]
    lo_po, hi_po = core.BOUNDS["po"]
    vp = synced_input(SYM["vp_vs"] + "&nbsp;&nbsp;(wick volume ratio)",
                      lo_vp, hi_vp, default_vp, 0.005, f"{key_prefix}vp")
    po = synced_input(SYM["po"] + "&nbsp;&nbsp;(porosity)",
                      lo_po, hi_po, default_po, 0.005, f"{key_prefix}po")
    return vp, po


LEVEL_STYLE = {
    "high":     (C_OK,     "High confidence",
                 "Inside the data hull; predictive uncertainty is low."),
    "moderate": (C_WARN,   "Moderate confidence",
                 "Near the edge of the sampled region - treat with care."),
    "low":      (C_ACCENT, "Low confidence / extrapolation",
                 "Outside the trusted region - the surrogate is guessing."),
}


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


# --------------------------------------------------------------------------- #
# Plot styling - WHITE opaque paper so labels are readable on any page theme
# --------------------------------------------------------------------------- #
def base_layout(fig: go.Figure, height=480, legend_below=True):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER,
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
        showline=True, linewidth=1.6, linecolor=AXLINE, mirror=True,
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
        paper_bgcolor=PAPER,
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
        name=f"FEM samples (n={assets.X_train.shape[0]})",
        hovertemplate="V<sub>p</sub>:V<sub>s</sub>=%{x:.3f}<br><i>\u03b5</i>=%{y:.3f}<extra></extra>",
    ))
    return fig
