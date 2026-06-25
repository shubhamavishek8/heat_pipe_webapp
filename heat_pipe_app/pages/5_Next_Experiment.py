"""Page 5 - Next Experiment: 2D map of the GPR predictive uncertainty with
suggested locations for the next FEM simulation(s) - active-learning style.

Rationale: with a small sample, the single most valuable thing a new FEM run
can do is shrink the surrogate's uncertainty where it matters. The candidates
below are the feasible designs of maximum predictive std, picked greedily with
an exclusion radius so the suggestions are spread out rather than clustered on
one ridge.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, cached_sigma_grid, inject_css,
                       page_header, base_layout, add_training_points, metric_card,
                       PLOTLY_CONFIG, SYM, AX, C_PRIMARY, C_ACCENT, C_OK, C_WARN, MUTED)

st.set_page_config(page_title="Next Experiment", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

if not A.supports_std:
    page_header("Next Experiment (Active Learning)",
                "This page selects the next FEM run from the surrogate's predictive "
                "uncertainty.")
    st.info(f"The deployed surrogate ({A.model_name}) does not provide predictive "
            f"uncertainty, so an active-learning suggestion cannot be computed. This "
            f"page becomes available when the best model is a Gaussian Process "
            f"(which exposes predict(return_std=True)). All other pages work normally.")
    st.stop()

page_header("Next Experiment (Active Learning)",
            f"Where should the NEXT FEM simulation go? The map shows the GPR predictive "
            f"uncertainty \u03c3({SYM['r_th']}); the starred designs are the feasible points "
            f"where one more sample would most improve the surrogate.")

c = st.columns([1, 1, 1])
with c[0]:
    ptot_max = st.number_input("Pressure-drop limit " + "\u0394P\u209c\u2092\u209c".strip() + " (Pa)",
                               min_value=500.0, max_value=25000.0,
                               value=core.PTOT_MAX_DEFAULT, step=100.0)
with c[1]:
    n_cand = st.select_slider("Number of suggestions", options=[1, 2, 3, 4, 5], value=3)
with c[2]:
    restrict = st.checkbox("Restrict to feasible region", value=True,
                           help="If on, candidates must satisfy the pressure-drop limit. "
                                "If off, the whole design space competes - useful when the "
                                "goal is global model accuracy rather than the optimisation.")

# Dense lattices (cached): outputs and sigma share the same grid construction
N = 120
VP, PO, R, P = cached_grid(N, N)
_, _, SIG = cached_sigma_grid(N, N)

feas = P <= ptot_max
eligible = feas if restrict else np.ones_like(feas, dtype=bool)

if restrict and not feas.any():
    st.error("No feasible designs at this limit - loosen it or untick the restriction.")
    st.stop()

# Greedy max-sigma selection with an exclusion radius (10% of each range)
rx = 0.10 * (core.UB[0] - core.LB[0])
ry = 0.10 * (core.UB[1] - core.LB[1])
sig_work = np.where(eligible, SIG, -np.inf)
cands = []
for _ in range(int(n_cand)):
    if not np.isfinite(sig_work).any() or np.nanmax(sig_work) == -np.inf:
        break
    iy, ix = np.unravel_index(int(np.argmax(sig_work)), sig_work.shape)
    vp_c, po_c, s_c = float(VP[iy, ix]), float(PO[iy, ix]), float(SIG[iy, ix])
    y_c = core.predict(A, [vp_c, po_c])[0]
    cands.append({"vp": vp_c, "po": po_c, "sigma": s_c,
                  "r_th": float(y_c[0]), "p_tot": float(y_c[1])})
    # exclude an ellipse around the chosen point so suggestions spread out
    excl = ((VP - vp_c) / rx) ** 2 + ((PO - po_c) / ry) ** 2 <= 1.0
    sig_work[excl] = -np.inf

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Suggested FEM runs")
    sig_med = float(np.median(SIG))
    for i, cd in enumerate(cands, start=1):
        ratio = cd["sigma"] / sig_med if sig_med > 0 else float("nan")
        metric_card(
            f"#{i}&nbsp;&nbsp;{SYM['vp_vs']} = {cd['vp']:.3f}, "
            f"{SYM['po']} = {cd['po']:.3f}",
            f"\u03c3 = {cd['sigma']:.4f} K/W",
            sub=f"{ratio:.1f}\u00d7 the median \u03c3 &nbsp;|&nbsp; surrogate predicts "
                f"{SYM['r_th']} \u2248 {cd['r_th']:.4f} K/W, "
                f"{SYM['p_tot']} \u2248 {cd['p_tot']:.0f} Pa",
            value_color=C_WARN if i == 1 else None)
    st.caption("Greedy maximum-uncertainty selection with a 10%-of-range exclusion radius "
               "so candidates do not cluster on a single ridge. Re-running the FEM at #1 "
               "and retraining offline gives the largest expected reduction in surrogate "
               "error where it is currently worst.")

with right:
    fig = go.Figure(go.Contour(
        x=VP[0], y=PO[:, 0], z=SIG, colorscale="Magma",
        contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
        colorbar=dict(title=dict(text=AX["sigma"]))))
    # feasibility boundary
    fig.add_trace(go.Contour(
        x=VP[0], y=PO[:, 0], z=feas.astype(float), showscale=False,
        contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
        line=dict(color=C_ACCENT, width=3), hoverinfo="skip",
        name="feasibility boundary"))
    add_training_points(fig, A, marker_color="white")
    if cands:
        fig.add_trace(go.Scatter(
            x=[cd["vp"] for cd in cands], y=[cd["po"] for cd in cands],
            mode="markers+text", name="suggested runs",
            text=[str(i) for i in range(1, len(cands) + 1)],
            textposition="top center",
            textfont=dict(color="#ffd166", size=14, family="'Times New Roman', Times, serif"),
            marker=dict(size=16, color="#ffd166", symbol="star",
                        line=dict(width=1.5, color="#1a2230"))))
    fig.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"])
    st.plotly_chart(base_layout(fig, height=560), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Bright ridges = least-trusted regions (far from the {A.n} samples). Valleys sit "
               "on the FEM points. Red line = pressure-drop feasibility boundary; gold stars "
               "= suggested next simulations, numbered by priority.")
