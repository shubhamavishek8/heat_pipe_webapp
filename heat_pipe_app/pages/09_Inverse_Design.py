"""Page 09 - Inverse Design: specify a TARGET thermal resistance and get back
feasible candidate designs that achieve it. The forward pages answer
"design -> performance"; this answers the engineer's actual question,
"performance -> design". Because an iso-R_th line generally crosses the design
space, several designs achieve the same target - the page returns a spread of
them, ranked by constraint slack, each with a trust flag."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, inject_css, page_header,
                       metric_card, base_layout, add_training_points,
                       PLOTLY_CONFIG, SYM, AX, U, C_PRIMARY, C_ACCENT, C_OK,
                       C_WARN, MUTED)

st.set_page_config(page_title="Inverse Design", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Inverse Design",
            f"Choose a target {SYM['r_th']}; get feasible designs that deliver it. "
            f"Candidates lie on the target iso-line, satisfy the pressure-drop limit, and "
            f"are spread out so you can pick by manufacturability or trust.")

N = 220
VP, PO, R, P = cached_grid(N, N)

ctrl = st.columns([1.1, 1, 1, 1])
with ctrl[1]:
    ptot_max = st.number_input("Pressure-drop limit (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)
feas = P <= ptot_max
if not feas.any():
    st.error("No feasible designs at this limit - loosen it.")
    st.stop()
r_lo, r_hi = float(R[feas].min()), float(R[feas].max())
with ctrl[0]:
    target = st.number_input(f"Target R_th (K/W)  [achievable: {r_lo:.4f} - {r_hi:.4f}]",
                             min_value=0.0, value=round((r_lo + r_hi) / 2, 4),
                             step=0.001, format="%.4f")
with ctrl[2]:
    tol = st.number_input("Tolerance \u00b1 (K/W)", min_value=0.0005, max_value=0.05,
                          value=0.002, step=0.0005, format="%.4f")
with ctrl[3]:
    n_cand = st.select_slider("Candidates", options=[1, 2, 3, 4, 5], value=3)

if not (r_lo - tol <= target <= r_hi + tol):
    st.error(f"Target {target:.4f} K/W is outside the feasible achievable range "
             f"[{r_lo:.4f}, {r_hi:.4f}] at this pressure-drop limit. "
             f"{'Raise the target or loosen the limit.' if target < r_lo else 'Lower the target.'}")
    st.stop()

# ---- candidate extraction: on-target, feasible, spread out ----------------- #
on_target = (np.abs(R - target) <= tol) & feas
if not on_target.any():
    st.warning("No grid designs hit the target within this tolerance - widen the tolerance.")
    st.stop()

rx = 0.08 * (core.UB[0] - core.LB[0])
ry = 0.08 * (core.UB[1] - core.LB[1])
score = np.where(on_target, (ptot_max - P), -np.inf)   # rank by constraint slack
cands = []
work = score.copy()
for _ in range(int(n_cand)):
    if not np.isfinite(work).any() or np.nanmax(work) == -np.inf:
        break
    iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
    vp_c, po_c = float(VP[iy, ix]), float(PO[iy, ix])
    y_c = core.predict(A, [vp_c, po_c])[0]
    stt = core.domain_status(A, [vp_c, po_c])
    cands.append({"vp": vp_c, "po": po_c, "r_th": float(y_c[0]),
                  "p_tot": float(y_c[1]), "slack": float(ptot_max - y_c[1]),
                  "trust": stt["level"]})
    work[((VP - vp_c) / rx) ** 2 + ((PO - po_c) / ry) ** 2 <= 1.0] = -np.inf

left, right = st.columns([1, 1.5], gap="large")
TRUST_COLOR = {"high": C_OK, "moderate": C_WARN, "low": C_ACCENT}

with left:
    st.subheader("Candidate designs")
    for i, cd in enumerate(cands, start=1):
        keq = core.k_eq_from_rth(cd["r_th"])
        metric_card(
            f"#{i}&nbsp;&nbsp;{SYM['vp_vs']} = {cd['vp']:.3f}, {SYM['po']} = {cd['po']:.3f}",
            f"{cd['r_th']:.4f} K/W",
            sub=f"{SYM['p_tot']} = {cd['p_tot']:.0f} Pa (slack {cd['slack']:.0f}) &nbsp;|&nbsp; "
                f"{SYM['k_eq']} = {keq:,.0f} W m\u207b\u00b9 K\u207b\u00b9 &nbsp;|&nbsp; "
                f"<span style='color:{TRUST_COLOR[cd['trust']]};font-weight:700'>"
                f"{cd['trust']} trust</span>",
            value_color=(C_OK if abs(cd["r_th"] - target) <= tol else C_WARN))
    st.caption("Ranked by pressure-drop slack (largest margin first) and spread apart with "
               "an 8%-of-range exclusion radius. All candidates achieve the target within "
               "the tolerance and satisfy the limit; the trust flag comes from the domain "
               "guard (GPR \u03c3 percentile + convex hull).")

with right:
    fig = go.Figure(go.Contour(
        x=VP[0], y=PO[:, 0], z=R, colorscale="Viridis",
        contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
        colorbar=dict(title=dict(text=AX["r_th"]))))
    # target iso-line (gold) and feasibility boundary (red)
    fig.add_trace(go.Contour(
        x=VP[0], y=PO[:, 0], z=R, showscale=False,
        contours=dict(start=target, end=target, size=1, coloring="lines"),
        line=dict(color="#ffd166", width=4), hoverinfo="skip", name="target iso-line"))
    fig.add_trace(go.Contour(
        x=VP[0], y=PO[:, 0], z=feas.astype(float), showscale=False,
        contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
        line=dict(color=C_ACCENT, width=3), hoverinfo="skip", name="feasibility boundary"))
    add_training_points(fig, A, marker_color="white")
    if cands:
        fig.add_trace(go.Scatter(
            x=[c["vp"] for c in cands], y=[c["po"] for c in cands],
            mode="markers+text", name="candidates",
            text=[str(i) for i in range(1, len(cands) + 1)], textposition="top center",
            textfont=dict(color="#1a2230", size=14),
            marker=dict(size=16, color="#ffd166", symbol="star",
                        line=dict(width=1.5, color="#1a2230"))))
    fig.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"])
    st.plotly_chart(base_layout(fig, height=560), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Gold line = every design achieving the target resistance; the red boundary "
               "cuts it into feasible and infeasible stretches. Stars = returned candidates.")
