"""Page 7 - Sensitivity Explorer: one-at-a-time (OAT) response slices through a
chosen design point, with GPR predictive bands, plus local finite-difference
sensitivities. These are the 1-D cuts reviewers ask for: how each output moves
when ONE input varies and the other is held fixed."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, inject_css, page_header, input_sliders,
                       metric_card, base_layout, domain_badge, PLOTLY_CONFIG,
                       SYM, AX, C_PRIMARY, C_ACCENT, C_OK, MUTED, TEXT)

st.set_page_config(page_title="Sensitivity", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Sensitivity Explorer",
            f"One-at-a-time slices: vary a single input through the chosen design point "
            f"and watch {SYM['r_th']} and {SYM['p_tot']} respond, with predictive bands. "
            f"Local derivatives quantify which input dominates at that point.")

left, right = st.columns([1, 2.2], gap="large")

with left:
    st.subheader("Anchor point")
    vp, po = input_sliders(0.5, 0.6, key_prefix="sens_")
    ptot_max = st.number_input("Pressure-drop limit (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)
    domain_badge(core.domain_status(A, [vp, po]))

    # ---- local finite-difference sensitivities (central differences) ------
    h_vp = 0.01 * (core.UB[0] - core.LB[0])
    h_po = 0.01 * (core.UB[1] - core.LB[1])
    y0 = core.predict(A, [vp, po])[0]

    def _clamped(v, lo, hi):
        return float(np.clip(v, lo, hi))

    yv1 = core.predict(A, [_clamped(vp + h_vp, *core.BOUNDS["vp_vs"]), po])[0]
    yv0 = core.predict(A, [_clamped(vp - h_vp, *core.BOUNDS["vp_vs"]), po])[0]
    yp1 = core.predict(A, [vp, _clamped(po + h_po, *core.BOUNDS["po"])])[0]
    yp0 = core.predict(A, [vp, _clamped(po - h_po, *core.BOUNDS["po"])])[0]
    d_rth_dvp = (yv1[0] - yv0[0]) / (2 * h_vp)
    d_rth_dpo = (yp1[0] - yp0[0]) / (2 * h_po)
    d_pt_dvp = (yv1[1] - yv0[1]) / (2 * h_vp)
    d_pt_dpo = (yp1[1] - yp0[1]) / (2 * h_po)

    st.subheader("Local sensitivities")
    metric_card("\u2202" + SYM["r_th"] + "/\u2202" + SYM["vp_vs"],
                f"{d_rth_dvp:+.4f}", sub="K/W per unit ratio")
    metric_card("\u2202" + SYM["r_th"] + "/\u2202" + SYM["po"],
                f"{d_rth_dpo:+.4f}", sub="K/W per unit porosity")
    metric_card("\u2202" + SYM["p_tot"] + "/\u2202" + SYM["vp_vs"],
                f"{d_pt_dvp:+,.0f}", sub="Pa per unit ratio")
    metric_card("\u2202" + SYM["p_tot"] + "/\u2202" + SYM["po"],
                f"{d_pt_dpo:+,.0f}", sub="Pa per unit porosity")
    dom_r = SYM["vp_vs"] if abs(d_rth_dvp * (core.UB[0]-core.LB[0])) >= abs(d_rth_dpo * (core.UB[1]-core.LB[1])) else SYM["po"]
    dom_p = SYM["vp_vs"] if abs(d_pt_dvp * (core.UB[0]-core.LB[0])) >= abs(d_pt_dpo * (core.UB[1]-core.LB[1])) else SYM["po"]
    st.markdown(f"Range-normalised, {SYM['r_th']} is currently governed by {dom_r} and "
                f"{SYM['p_tot']} by {dom_p} at this point.", unsafe_allow_html=True)

N_SLICE = 120
gv = np.linspace(*core.BOUNDS["vp_vs"], N_SLICE)
gp = np.linspace(*core.BOUNDS["po"], N_SLICE)
u_vp = core.predict_with_uncertainty(A, np.column_stack([gv, np.full(N_SLICE, po)]), k=2.0)
u_po = core.predict_with_uncertainty(A, np.column_stack([np.full(N_SLICE, vp), gp]), k=2.0)
has_sd = A.supports_std


def _slice_fig(x, u, out, xlabel, anchor_x, anchor_y):
    lo_k, hi_k = (f"{out}_lo", f"{out}_hi")
    fig = go.Figure()
    if has_sd:
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                 y=np.concatenate([u[hi_k], u[lo_k][::-1]]),
                                 fill="toself", fillcolor="rgba(31,95,176,0.15)",
                                 line=dict(width=0), hoverinfo="skip",
                                 name="\u00b12\u03c3 band"))
    fig.add_trace(go.Scatter(x=x, y=u[out], mode="lines",
                             line=dict(color=C_PRIMARY, width=2.4), name="prediction"))
    fig.add_trace(go.Scatter(x=[anchor_x], y=[anchor_y], mode="markers",
                             marker=dict(size=12, color=C_ACCENT, symbol="x",
                                         line=dict(width=2)), name="anchor point"))
    fig.update_layout(xaxis_title=xlabel,
                      yaxis_title=AX["r_th"] if out == "r_th" else AX["p_tot"])
    return fig


with right:
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        f = _slice_fig(gv, u_vp, "r_th", AX["vp_vs"], vp, float(core.predict(A, [vp, po])[0, 0]))
        st.plotly_chart(base_layout(f, height=330), use_container_width=True, config=PLOTLY_CONFIG)
    with r1c2:
        f = _slice_fig(gp, u_po, "r_th", AX["po"], po, float(core.predict(A, [vp, po])[0, 0]))
        st.plotly_chart(base_layout(f, height=330), use_container_width=True, config=PLOTLY_CONFIG)
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        f = _slice_fig(gv, u_vp, "p_tot", AX["vp_vs"], vp, float(core.predict(A, [vp, po])[0, 1]))
        f.add_hline(y=ptot_max, line=dict(color=C_ACCENT, dash="dash"))
        st.plotly_chart(base_layout(f, height=330), use_container_width=True, config=PLOTLY_CONFIG)
    with r2c2:
        f = _slice_fig(gp, u_po, "p_tot", AX["po"], po, float(core.predict(A, [vp, po])[0, 1]))
        f.add_hline(y=ptot_max, line=dict(color=C_ACCENT, dash="dash"))
        st.plotly_chart(base_layout(f, height=330), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Top row: {SYM['r_th']} slices; bottom row: {SYM['p_tot']} slices with the "
               f"pressure-drop limit dashed. Shaded = GPR \u00b12\u03c3 band; widening bands "
               f"flag data-sparse stretches of the slice. The anchor is the X.",
               unsafe_allow_html=True)
