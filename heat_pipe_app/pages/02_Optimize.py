"""Page 2 - Constrained optimisation with a user-selectable solver:
SLSQP (gradient-based) or step-defined grid search, plus the Pareto front."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, cached_pareto, inject_css, page_header,
                       base_layout, add_training_points, metric_card, synced_input,
                       html_label, PLOTLY_CONFIG, SYM, AX, U,
                       C_PRIMARY, C_ACCENT, C_OK, MUTED)

st.set_page_config(page_title="Optimise", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Constrained Optimisation",
            f"minimise {SYM['r_th']}({SYM['vp_vs']}, {SYM['po']}) subject to "
            f"{SYM['p_tot']} &le; limit.")

top_l, top_r = st.columns([1.6, 1], gap="large")
with top_l:
    ptot_max = synced_input("Pressure-drop constraint limit " + SYM["p_tot"] + " (Pa)",
                            1500.0, 10000.0, core.PTOT_MAX_DEFAULT, 50.0,
                            key="opt_limit", fmt="%.2f", num_step=0.01)
with top_r:
    method = st.radio("Optimisation method",
                      ["SLSQP (gradient-based)", "Grid search (exhaustive)"],
                      index=0,
                      help="SLSQP converges to the exact constraint boundary; grid search "
                           "evaluates a step-defined lattice over the data range, exactly "
                           "as in the offline pipeline.")
    use_grid = method.startswith("Grid")
    if use_grid:
        st.markdown("**Lattice spacing** (range fixed to the data bounds)")
        html_label(SYM["vp_vs"] + " step")
        step_vp = st.number_input("vp step", min_value=0.005, max_value=0.2,
                                  value=0.05, step=0.005, format="%.3f",
                                  label_visibility="collapsed")
        html_label(SYM["po"] + " step")
        step_po = st.number_input("po step", min_value=0.005, max_value=0.1,
                                  value=0.01, step=0.005, format="%.3f",
                                  label_visibility="collapsed")
    else:
        step_vp, step_po = 0.05, 0.01   # defaults used for the cross-check

if use_grid:
    sol = core.grid_search_min_rth(A, ptot_max=ptot_max,
                                   step_vp=step_vp, step_po=step_po)
    other = core.optimize_min_rth(A, ptot_max=ptot_max)
    other_name = "SLSQP"
else:
    sol = core.optimize_min_rth(A, ptot_max=ptot_max)
    other = core.grid_search_min_rth(A, ptot_max=ptot_max,
                                     step_vp=step_vp, step_po=step_po)
    other_name = "Grid search"

left, right = st.columns([1, 1.4], gap="large")

with left:
    st.subheader("Optimal design")
    if sol is None:
        st.error("No feasible solution at this limit - try loosening the pressure-drop limit.")
    else:
        k_eq_opt = core.k_eq_from_rth(sol["r_th"])
        metric_card(SYM["r_th"] + "*&nbsp;&nbsp;(K/W)", f"{sol['r_th']:.3f}")
        metric_card(SYM["p_tot"] + "*&nbsp;&nbsp;(Pa)", f"{sol['p_tot']:.2f}",
                    sub=f"slack {sol['slack']:.2f} Pa")
        metric_card(SYM["k_eq"] + "*&nbsp;&nbsp;(W m\u207b\u00b9 K\u207b\u00b9)",
                    f"{k_eq_opt:,.0f}",
                    sub=f"L<sub>eff</sub>/(A<sub>c</sub>\u00b7{SYM['r_th']}*) with "
                        f"Q = {core.Q_WATT:.0f} W, A<sub>c</sub> = 9.31\u00d710\u207b\u2076 m\u00b2, "
                        f"L<sub>eff</sub> = {core.L_EFF:.2f} m")
        st.markdown(
            f"{SYM['vp_vs']}* = <code>{sol['vp_vs']:.4f}</code> &nbsp;&nbsp; "
            f"{SYM['po']}* = <code>{sol['po']:.4f}</code>",
            unsafe_allow_html=True)

        active = abs(sol["slack"]) < 5.0
        if active:
            st.info("Constraint is **active** - the optimum sits on the pressure-drop "
                    "boundary, so the limit directly sets achievable thermal resistance.")
        else:
            st.info("Constraint is **inactive** - the unconstrained optimum already "
                    "satisfies the limit.")

        if use_grid:
            nx, ny = sol["grid"]
            st.caption(f"Solver: exhaustive grid search, steps "
                       f"\u0394(Vp:Vs) = {sol['step'][0]:.3f}, \u0394\u03b5 = {sol['step'][1]:.3f} "
                       f"\u2192 {nx} \u00d7 {ny} = {sol['n_eval']} designs evaluated, "
                       f"{sol['n_feasible']} feasible. Global on the lattice; accuracy "
                       f"limited by the step size.")
        else:
            st.caption("Solver: SLSQP, gradient-based with native inequality constraint, "
                       "best of the multi-start runs. Converges to the exact constraint "
                       "boundary.")

        if other is not None:
            with st.expander(f"Cross-check vs {other_name}"):
                st.markdown(
                    f"**{other_name} optimum:** "
                    f"{SYM['vp_vs']}= <code>{other['vp_vs']:.4f}</code> , "
                    f"{SYM['po']}= <code>{other['po']:.4f}</code> , "
                    f"{SYM['r_th']}= <code>{other['r_th']:.3f}</code> , "
                    f"{SYM['p_tot']}= <code>{other['p_tot']:.2f}</code> Pa. "
                    f"Difference in {SYM['r_th']}: "
                    f"<code>{abs(other['r_th']-sol['r_th']):.2e}</code> K/W.",
                    unsafe_allow_html=True)

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
    st.caption("Each point is the minimum resistance at a different pressure-drop cap. "
               "The slope is the constraint shadow price: resistance bought per extra Pa "
               "of allowable pressure drop.")

st.subheader("Feasible region & optimum")
VP, PO, R, P = cached_grid(90, 90)
feas = (P <= ptot_max).astype(float)
fig2 = go.Figure()
fig2.add_trace(go.Contour(x=VP[0], y=PO[:, 0], z=R, colorscale="Viridis",
                          contours=dict(showlabels=True,
                                        labelfont=dict(size=11, color="white")),
                          colorbar=dict(title=dict(text=AX["r_th"]))))
fig2.add_trace(go.Contour(x=VP[0], y=PO[:, 0], z=feas, showscale=False,
                          colorscale=[[0, C_ACCENT], [1, C_ACCENT]],   # constant -> RED
                          contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                          line=dict(width=3), hoverinfo="skip", showlegend=True,
                          name=U["p_tot"] + " limit (feasibility boundary)"))
add_training_points(fig2, A, marker_color="white")
if sol is not None:
    fig2.add_trace(go.Scatter(x=[sol["vp_vs"]], y=[sol["po"]], mode="markers",
                              marker=dict(size=18, color=C_ACCENT, symbol="star",
                                          line=dict(width=1.5, color="white")),
                              name="optimum"))
fig2.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"])
st.plotly_chart(base_layout(fig2, height=520), use_container_width=True, config=PLOTLY_CONFIG)
st.caption("Background = resistance surface. Red line = pressure-drop limit (feasible side "
           "has lower pressure drop). Star = optimum.")
