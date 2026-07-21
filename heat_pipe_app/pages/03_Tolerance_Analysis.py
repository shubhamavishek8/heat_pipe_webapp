"""Page 3 - Manufacturing-tolerance propagation: how input scatter affects yield."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, inject_css, page_header, base_layout, input_sliders,
                       metric_card, html_label, PLOTLY_CONFIG, SYM, AX,
                       C_PRIMARY, C_ACCENT, C_OK, C_PURPLE)

st.set_page_config(page_title="Tolerance Analysis", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Manufacturing-Tolerance Analysis",
            "Propagate fabrication scatter on the design variables through the surrogate to a "
            "performance distribution and a yield (probability of meeting the pressure-drop limit).")

c = st.columns([1, 1, 1])
with c[0]:
    st.markdown("**Nominal design**")
    use_opt = st.checkbox("Use the constrained optimum as nominal", value=True)
    ptot_max = st.number_input("Pressure-drop limit (Pa)", value=core.PTOT_MAX_DEFAULT, step=100.0)
    if use_opt:
        sol = core.optimize_min_rth(A, ptot_max=ptot_max)
        vp0, po0 = (sol["vp_vs"], sol["po"]) if sol else (0.5, 0.6)
        st.markdown(f"<span style='color:#5a6675'>Optimum: {SYM['vp_vs']}={vp0:.4f}, "
                    f"{SYM['po']}={po0:.4f}</span>", unsafe_allow_html=True)
    else:
        vp0, po0 = input_sliders(0.5, 0.6, key_prefix="tol_")

with c[1]:
    st.markdown("**Tolerances (&plusmn;, absolute)**", unsafe_allow_html=True)
    html_label(SYM["vp_vs"] + " tolerance &plusmn;")
    tol_vp = st.number_input("vp tol", value=0.03, step=0.005, format="%.3f",
                             label_visibility="collapsed")
    html_label(SYM["po"] + " tolerance &plusmn;")
    tol_po = st.number_input("po tol", value=0.02, step=0.005, format="%.3f",
                             label_visibility="collapsed")
with c[2]:
    st.markdown("**Sampling**")
    dist = st.radio("Within-tolerance distribution", ["Uniform", "Normal (\u00b1=3\u03c3)"], index=1)
    n = st.select_slider("Monte-Carlo samples", options=[2000, 5000, 10000, 20000], value=10000)

rng = np.random.default_rng(42)
if dist.startswith("Uniform"):
    s_vp = rng.uniform(vp0 - tol_vp, vp0 + tol_vp, n)
    s_po = rng.uniform(po0 - tol_po, po0 + tol_po, n)
else:
    s_vp = rng.normal(vp0, tol_vp / 3.0, n)
    s_po = rng.normal(po0, tol_po / 3.0, n)
s_vp = np.clip(s_vp, *core.BOUNDS["vp_vs"])
s_po = np.clip(s_po, *core.BOUNDS["po"])

Y = core.predict(A, np.column_stack([s_vp, s_po]))
r_s, p_s = Y[:, 0], Y[:, 1]
yield_pct = (p_s <= ptot_max).mean() * 100.0

m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card(SYM["r_th"] + " mean", f"{r_s.mean():.3f}", sub=f"std {r_s.std():.3f}")
with m2:
    metric_card(SYM["p_tot"] + " mean", f"{p_s.mean():.2f}", sub=f"std {p_s.std():.2f}")
with m3:
    metric_card("Yield  P(" + SYM["p_tot"] + " &le; limit)", f"{yield_pct:.1f}%",
                value_color=(C_OK if yield_pct >= 99 else C_ACCENT))
with m4:
    metric_card(SYM["r_th"] + " 95th pct", f"{np.quantile(r_s,0.95):.3f}")

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure(go.Histogram(x=r_s, nbinsx=50, marker_color=C_PRIMARY))
    fig.update_layout(xaxis_title=AX["r_th"], yaxis_title="count")
    st.plotly_chart(base_layout(fig, height=360, legend_below=False),
                    use_container_width=True, config=PLOTLY_CONFIG)
with g2:
    fig = go.Figure(go.Histogram(x=p_s, nbinsx=50, marker_color=C_PURPLE))
    fig.add_vline(x=ptot_max, line=dict(color=C_ACCENT, width=3, dash="dash"),
                  annotation_text="limit")
    fig.update_layout(xaxis_title=AX["p_tot"], yaxis_title="count")
    st.plotly_chart(base_layout(fig, height=360, legend_below=False),
                    use_container_width=True, config=PLOTLY_CONFIG)

if yield_pct < 99:
    st.warning(f"At these tolerances, ~{100-yield_pct:.1f}% of parts would exceed the pressure-drop "
               f"limit. Tighten tolerances, or move the nominal design away from the constraint boundary.")
else:
    st.success(f"Yield ~ {yield_pct:.1f}% - the design tolerates this manufacturing scatter.")
