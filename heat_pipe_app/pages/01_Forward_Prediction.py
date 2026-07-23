"""Page 1 - Forward prediction with predictive uncertainty, a domain guard,
and the equivalent thermal conductivity k_eq (fixed geometry, Q = 40 W)."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, inject_css, page_header, input_sliders,
                       synced_input, domain_badge, metric_card, base_layout,
                       add_training_points, PLOTLY_CONFIG, SYM, AX, U,
                       C_PRIMARY, C_ACCENT, C_OK, MUTED)

st.set_page_config(page_title="Forward Prediction", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

# ---- shareable-URL state: seed widgets from ?vp=&po=&k=&ptot= on first load --
qp = st.query_params
if "vp" in qp and "pred_vp_sl" not in st.session_state:
    try:
        _v = float(np.clip(float(qp["vp"]), *core.BOUNDS["vp_vs"]))
        _p = float(np.clip(float(qp.get("po", 0.6)), *core.BOUNDS["po"]))
        st.session_state["pred_vp_sl"] = st.session_state["pred_vp_nb"] = _v
        st.session_state["pred_po_sl"] = st.session_state["pred_po_nb"] = _p
        if "k" in qp:
            st.session_state["pred_k"] = int(qp["k"])
        if "ptot" in qp:
            st.session_state["pred_ptot"] = float(qp["ptot"])
    except (ValueError, TypeError):
        pass

page_header("Forward Prediction",
            f"Set a design point and get {SYM['r_th']}, {SYM['k_eq']}, and {SYM['p_tot']} "
            f", each with a predictive interval and a "
            f"trust signal.")

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Design point")
    vp, po = input_sliders(0.5, 0.6, key_prefix="pred_")
    if A.supports_std:
        k = st.select_slider("Predictive interval", options=[1, 2, 3], value=2,
                             key="pred_k",
                             format_func=lambda z: f"\u00b1{z}\u03c3  (~{ {1:68,2:95,3:99.7}[z] }%)")
    else:
        k = 2
    if "pred_ptot" not in st.session_state:
        st.session_state["pred_ptot"] = core.PTOT_MAX_DEFAULT
    ptot_max = st.number_input("Pressure-drop constraint (Pa)",
                               step=100.0, key="pred_ptot")

    u = core.predict_with_uncertainty(A, [vp, po], k=float(k))
    status = core.domain_status(A, [vp, po])

    r_th = float(u["r_th"][0]); p_tot = float(u["p_tot"][0])

    # equivalent thermal conductivity with the FIXED parameters
    # A_c = 9.31e-6 m^2, L_eff = 0.14 m, Q = 40 W
    dT = core.Q_WATT * r_th
    k_eq = core.k_eq_from_rth(r_th)
    k_eq_lo = core.k_eq_from_rth(float(u["r_th_hi"][0]))   # high R_th -> low k_eq
    k_eq_hi = core.k_eq_from_rth(float(u["r_th_lo"][0]))

    st.subheader("Prediction")
    feasible = p_tot <= ptot_max
    has_sd = A.supports_std
    band_note = "" if has_sd else f"point estimate (model: {A.model_name}; predictive bands need a Gaussian Process)"

    metric_card(SYM["r_th"] + "&nbsp;&nbsp;(K/W)", f"{r_th:.3f}",
                sub=(f"\u00b1{k}\u03c3 band [{u['r_th_lo'][0]:.3f}, {u['r_th_hi'][0]:.3f}] (symmetric)"
                     if has_sd else band_note))

    metric_card(SYM["p_tot"] + "&nbsp;&nbsp;(Pa)", f"{p_tot:.2f}",
                sub=(f"\u00b1{k}\u03c3 band [{u['p_tot_lo'][0]:.2f}, {u['p_tot_hi'][0]:.2f}] "
                     f"(log-space, asymmetric)" if has_sd else band_note),
                value_color=(C_OK if feasible else C_ACCENT))

    k_eq_sub = (f"band [{k_eq_lo:,.0f}, {k_eq_hi:,.0f}] &nbsp;|&nbsp; " if has_sd else "")
    metric_card(SYM["k_eq"] + "&nbsp;&nbsp;(W m\u207b\u00b9 K\u207b\u00b9)", f"{k_eq:,.0f}",
                sub=k_eq_sub + f"{SYM['dT']} = Q\u00b7{SYM['r_th']} = {dT:.2f} K "
                    f"(Q = {core.Q_WATT:.0f} W, A<sub>c</sub> = 9.31\u00d710\u207b\u2076 m\u00b2, "
                    f"L<sub>eff</sub> = {core.L_EFF:.2f} m)")

    st.markdown(
        f"<b>Constraint:</b> {'Feasible' if feasible else 'Violates Limit'} "
        f"({SYM['p_tot']} {'&le;' if feasible else '&gt;'} {ptot_max:.0f} Pa)",
        unsafe_allow_html=True)

    domain_badge(status)

    with st.expander("Share this view"):
        if st.button("Encode this design point into the URL"):
            st.query_params.update({"vp": f"{vp:.3f}", "po": f"{po:.3f}",
                                    "k": str(int(k)), "ptot": f"{ptot_max:.0f}"})
            st.success("Done - the address bar now encodes this exact view. Copy the "
                       "URL from your browser and anyone opening it lands on this "
                       "design point.")
        st.caption("The link stores the design point, the interval and the constraint - "
                   "nothing else.")

with right:
    metric = st.radio("Response surface", [U["r_th"], U["p_tot"]], horizontal=True,
                      key="pred_metric", label_visibility="collapsed")
    is_rth = (metric == U["r_th"])
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
            colorscale=[[0, C_ACCENT], [1, C_ACCENT]],   # constant -> line stays RED
            contours=dict(start=ptot_max, end=ptot_max, size=1, coloring="lines"),
            line=dict(width=3), name=U["p_tot"] + " limit", hoverinfo="skip",
            showlegend=True))
    add_training_points(fig, A, marker_color="white")
    fig.add_trace(go.Scatter(x=[vp], y=[po], mode="markers", name="your point",
                             marker=dict(size=16, color=C_ACCENT, symbol="x",
                                         line=dict(width=2))))
    fig.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"])
    st.plotly_chart(base_layout(fig, height=560), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Red line on the pressure-drop surface = the active limit.")
