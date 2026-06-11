"""Page 1 - Forward prediction with predictive uncertainty, a domain guard,
and the equivalent thermal conductivity k_eq."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, inject_css, page_header, input_sliders,
                       domain_badge, metric_card, html_label, base_layout, add_training_points,
                       PLOTLY_CONFIG, SYM, AX, C_PRIMARY, C_ACCENT, C_OK, MUTED)

st.set_page_config(page_title="Predict", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

Q_WATT = 40.0   # total power, fixed per the project (Q = 40 W)

page_header("Forward Prediction",
            "Set a design point; get R_th, dP_tot and the equivalent thermal conductivity, "
            "each with a predictive interval and a trust signal.")

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Design point")
    vp, po = input_sliders(0.5, 0.6, key_prefix="pred_")
    k = st.select_slider("Predictive interval", options=[1, 2, 3], value=2,
                         format_func=lambda z: f"\u00b1{z}\u03c3  (~{ {1:68,2:95,3:99.7}[z] }%)")
    ptot_max = st.number_input("Pressure-drop constraint (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)

    with st.expander(f"Geometry for k\u209b\u2091 (Q = {Q_WATT:.0f} W fixed)", expanded=False):
        st.caption("Equivalent thermal conductivity depends on the heat-pipe geometry. "
                   "Set the effective length and outer diameter of your device.")
        D_mm = st.number_input("Outer diameter D (mm)", value=6.0, min_value=0.5,
                               step=0.5, format="%.2f")
        L_mm = st.number_input("Effective length L_eff (mm)", value=150.0, min_value=1.0,
                               step=5.0, format="%.1f")

    u = core.predict_with_uncertainty(A, [vp, po], k=float(k))
    status = core.domain_status(A, [vp, po])

    r_th = float(u["r_th"][0]); p_tot = float(u["p_tot"][0])

    # --- equivalent thermal conductivity -----------------------------------
    # dT = Q * R_th ;  k_eq = Q*L/(A_c*dT) = L/(A_c*R_th)
    A_c = np.pi * (D_mm / 1000.0) ** 2 / 4.0      # m^2
    L_eff = L_mm / 1000.0                          # m
    dT = Q_WATT * r_th                             # K
    k_eq = L_eff / (A_c * r_th) if r_th > 0 else float("nan")

    st.subheader("Prediction")
    feasible = p_tot <= ptot_max

    metric_card(SYM["r_th"] + "&nbsp;&nbsp;(K/W)", f"{r_th:.4f}",
                sub=f"\u00b1{k}\u03c3 band [{u['r_th_lo'][0]:.4f}, {u['r_th_hi'][0]:.4f}] (symmetric)")

    metric_card(SYM["p_tot"] + "&nbsp;&nbsp;(Pa)", f"{p_tot:.1f}",
                sub=f"\u00b1{k}\u03c3 band [{u['p_tot_lo'][0]:.1f}, {u['p_tot_hi'][0]:.1f}] "
                    f"(log-space, asymmetric)",
                value_color=(C_OK if feasible else C_ACCENT))

    metric_card(SYM["k_eq"] + "&nbsp;&nbsp;(W m\u207b\u00b9 K\u207b\u00b9)", f"{k_eq:,.0f}",
                sub=f"{SYM['dT']} = Q\u00b7{SYM['r_th']} = {dT:.2f} K  "
                    f"(A_c = {A_c*1e6:.2f} mm\u00b2, L_eff = {L_mm:.0f} mm)")

    st.markdown(
        f"<b>Constraint:</b> {'feasible' if feasible else 'violates limit'} "
        f"({SYM['p_tot']} {'&le;' if feasible else '&gt;'} {ptot_max:.0f} Pa)",
        unsafe_allow_html=True)

    domain_badge(status)

with right:
    metric = st.radio("Response surface", ["R_th", "dP_tot"], horizontal=True,
                      key="pred_metric", label_visibility="collapsed")
    is_rth = (metric == "R_th")
    VP, PO, R, P = cached_grid(80, 80)
    Z = R if is_rth else P
    colorscale = "Viridis" if is_rth else "Plasma"
    cbar_title = AX["r_th"] if is_rth else AX["p_tot"]

    fig = go.Figure(go.Contour(
        x=VP[0], y=PO[:, 0], z=Z, colorscale=colorscale,
        contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
        colorbar=dict(title=dict(text=cbar_title))))
    if not is_rth:
        fig.add_trace(go.Contour(
            x=VP[0], y=PO[:, 0], z=P, showscale=False,
            contours=dict(start=ptot_max, end=ptot_max, size=1, coloring="lines"),
            line=dict(color=C_ACCENT, width=3), name="limit", hoverinfo="skip"))
    add_training_points(fig, A, marker_color="white")
    fig.add_trace(go.Scatter(x=[vp], y=[po], mode="markers", name="your point",
                             marker=dict(size=16, color=C_ACCENT, symbol="x",
                                         line=dict(width=2))))
    fig.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"])
    st.plotly_chart(base_layout(fig, height=560), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Red line on the pressure-drop surface = the active limit. "
               "Open circles = the 49 FEM samples.")
