"""Page 2 - Constrained optimisation (SLSQP) and the R_th / dP_tot trade-off front."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, cached_pareto, inject_css, page_header,
                       base_layout, add_training_points, metric_card,
                       PLOTLY_CONFIG, SYM, AX, C_PRIMARY, C_ACCENT, C_OK)

st.set_page_config(page_title="Optimise", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Constrained Optimisation",
            "minimise R_th(Vp:Vs, porosity) subject to dP_tot <= limit.")

cc = st.columns([1.3, 1])
with cc[0]:
    ptot_max = st.slider("Pressure-drop constraint limit (Pa)", min_value=1500.0,
                         max_value=10000.0, value=core.PTOT_MAX_DEFAULT, step=100.0)
with cc[1]:
    method = st.radio("Optimisation method",
                      ["SLSQP (gradient-based)", "Grid search (exhaustive)"],
                      help="SLSQP converges to the exact constraint boundary; grid search "
                           "scans a lattice and is global but limited by its spacing.")

is_grid = method.startswith("Grid")
if is_grid:
    res = st.select_slider("Grid resolution (points per axis)",
                           options=[25, 50, 75, 100, 150, 200], value=100)
    sol = core.grid_search_min_rth(A, ptot_max=ptot_max, n_vp=res, n_po=res)
    other = core.optimize_min_rth(A, ptot_max=ptot_max)
else:
    sol = core.optimize_min_rth(A, ptot_max=ptot_max)
    other = core.grid_search_min_rth(A, ptot_max=ptot_max)

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Optimal design")
    if sol is None:
        st.error("No feasible solution at this limit - try loosening the pressure-drop limit.")
    else:
        metric_card(SYM["r_th"] + "*&nbsp;&nbsp;(K/W)", f"{sol['r_th']:.4f}")
        metric_card(SYM["p_tot"] + "*&nbsp;&nbsp;(Pa)", f"{sol['p_tot']:.1f}",
                    sub=f"slack {sol['slack']:.1f} Pa")
        st.markdown(
            f"{SYM['vp_vs']}* = <code>{sol['vp_vs']:.4f}</code> &nbsp;&nbsp; "
            f"{SYM['po']}* = <code>{sol['po']:.4f}</code>",
            unsafe_allow_html=True)
        active = abs(sol["slack"]) < 5.0
        if active:
            st.info("Constraint is **active** - the optimum sits on the pressure-drop boundary, "
                    "so the limit directly sets achievable thermal resistance.")
        else:
            st.info("Constraint is **inactive** - the unconstrained optimum already satisfies the limit.")
        if is_grid:
            st.caption(f"Solver: exhaustive grid search on a {sol['grid'][0]}x{sol['grid'][1]} "
                       f"lattice ({sol['n_feasible']} of {sol['n_eval']} cells feasible). "
                       f"Global on the lattice; accuracy limited by grid spacing.")
        else:
            st.caption("Solver: SLSQP, gradient-based with native inequality constraint, "
                       "best of the multi-start runs. Converges to the exact constraint boundary.")

        if other is not None:
            with st.expander("Cross-check vs the other method"):
                st.markdown(
                    f"**{other['method']}** optimum: "
                    f"{SYM['vp_vs']}=<code>{other['vp_vs']:.4f}</code>, "
                    f"{SYM['po']}=<code>{other['po']:.4f}</code>, "
                    f"{SYM['r_th']}=<code>{other['r_th']:.4f}</code>, "
                    f"{SYM['p_tot']}=<code>{other['p_tot']:.1f}</code> Pa.<br>"
                    f"Difference in {SYM['r_th']}: "
                    f"<code>{abs(other['r_th']-sol['r_th']):.2e}</code> K/W.",
                    unsafe_allow_html=True)
                st.caption("Close agreement is a sanity check that the optimum is real and not a "
                           "solver artefact.")

with right:
    st.subheader("Resistance / pressure-drop trade-off (Pareto front)")
    caps, Rf, Pf = cached_pareto(22)
    ok = ~np.isnan(Rf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Pf[ok], y=Rf[ok], mode="lines+markers",
                             line=dict(color=C_PRIMARY, width=2),
                             marker=dict(size=6), name="Pareto front"))
    if sol is not None:
        fig.add_trace(go.Scatter(x=[sol["p_tot"]], y=[sol["r_th"]], mode="markers",
                                 marker=dict(size=15, color=C_ACCENT, symbol="star"),
                                 name="current optimum"))
    fig.add_vline(x=ptot_max, line=dict(color=C_ACCENT, dash="dash"),
                  annotation_text="current limit", annotation_position="top")
    fig.update_layout(xaxis_title=AX["p_tot"], yaxis_title=AX["r_th"])
    st.plotly_chart(base_layout(fig), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Each point is the minimum resistance at a different pressure-drop cap. The slope is "
               "the constraint shadow price: resistance bought per extra Pa of allowable pressure drop.")

st.subheader("Feasible region & optimum")
VP, PO, R, P = cached_grid(90, 90)
feas = (P <= ptot_max).astype(float)
fig2 = go.Figure()
fig2.add_trace(go.Contour(x=VP[0], y=PO[:, 0], z=R, colorscale="Viridis",
                          contours=dict(showlabels=True,
                                        labelfont=dict(size=11, color="white")),
                          colorbar=dict(title=dict(text=AX["r_th"]))))
fig2.add_trace(go.Contour(x=VP[0], y=PO[:, 0], z=feas, showscale=False,
                          contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                          line=dict(color=C_ACCENT, width=3), hoverinfo="skip",
                          name="feasibility boundary"))
add_training_points(fig2, A, marker_color="white")
if sol is not None:
    fig2.add_trace(go.Scatter(x=[sol["vp_vs"]], y=[sol["po"]], mode="markers",
                              marker=dict(size=18, color=C_ACCENT, symbol="star",
                                          line=dict(width=1.5, color="white")),
                              name="optimum"))
fig2.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"])
st.plotly_chart(base_layout(fig2, height=520), use_container_width=True, config=PLOTLY_CONFIG)
st.caption("Background = resistance surface. Red line = pressure-drop limit (feasible side has lower "
           "pressure drop). Star = optimum.")
