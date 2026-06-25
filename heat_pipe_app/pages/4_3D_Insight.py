"""Page 4 - 3D interactive insights: response surfaces AND the GPR
model-uncertainty surface over the design space."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, cached_grid, cached_sigma_grid, inject_css,
                       page_header, style_scene, PLOTLY_CONFIG, SYM, AX, U,
                       C_ACCENT, C_PRIMARY, TEXT, FONT_TNR)

st.set_page_config(page_title="3D Insight", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("3D Interactive Insight",
            f"Rotate, zoom and pan the surrogate surfaces over the design space: the two "
            f"responses ({SYM['r_th']}, {SYM['p_tot']}) and the model's own uncertainty "
            f"\u03c3({SYM['r_th']}) - where the surrogate can and cannot be trusted.")

ctrl = st.columns([1.4, 1, 1])
with ctrl[0]:
    which = st.radio("Surface", [U["r_th"], U["p_tot"], U["sigma"]],
                     horizontal=True, label_visibility="collapsed")
with ctrl[1]:
    ptot_max = st.number_input("Pressure-drop limit (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)
with ctrl[2]:
    show_pts = st.checkbox("Show FEM samples", value=True)
    show_opt = st.checkbox("Show optimum", value=True)

VP, PO, R, P = cached_grid(70, 70)
gx, gy = VP[0], PO[:, 0]

mode = "rth" if which == U["r_th"] else ("ptot" if which == U["p_tot"] else "sigma")

if mode == "sigma":
    _, _, Z = cached_sigma_grid(70, 70)
    zlabel, surf_scale = AX["sigma"], "Magma"
else:
    Z = R if mode == "rth" else P
    zlabel = AX["r_th"] if mode == "rth" else AX["p_tot"]
    surf_scale = "Viridis" if mode == "rth" else "Plasma"

fig = go.Figure()
fig.add_trace(go.Surface(
    x=gx, y=gy, z=Z, colorscale=surf_scale, opacity=0.95,
    colorbar=dict(title=dict(text=zlabel, font=dict(family=FONT_TNR, size=15, color=TEXT)),
                  tickfont=dict(family=FONT_TNR, size=12, color=TEXT), len=0.6),
    contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
    name=which,
    hovertemplate="V<sub>p</sub>:V<sub>s</sub>=%{x:.3f}<br><i>\u03b5</i>=%{y:.3f}<br>"
                  + zlabel + "=%{z:.3g}<extra></extra>"))

# Constraint plane (only meaningful on the pressure-drop surface)
if mode == "ptot":
    plane = np.full_like(Z, ptot_max)
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=plane, showscale=False, opacity=0.35,
        colorscale=[[0, C_ACCENT], [1, C_ACCENT]],
        name=f"limit = {ptot_max:.0f} Pa", hoverinfo="skip", showlegend=True))

# FEM training samples
if show_pts:
    if mode == "sigma":
        u_tr = core.predict_with_uncertainty(A, A.X_train, k=1.0)
        z_tr = u_tr["r_th_sigma"]
        pt_name = "FEM samples (\u03c3 at sample)"
    else:
        z_tr = A.y_train[:, 0 if mode == "rth" else 1]
        pt_name = f"FEM samples (n={A.n})"
    fig.add_trace(go.Scatter3d(
        x=A.X_train[:, 0], y=A.X_train[:, 1], z=z_tr,
        mode="markers", name=pt_name,
        marker=dict(size=4, color="#1a2230", line=dict(width=1, color="white")),
        hovertemplate="V<sub>p</sub>:V<sub>s</sub>=%{x:.3f}<br><i>\u03b5</i>=%{y:.3f}<br>"
                      + zlabel + "=%{z:.3g}<extra></extra>"))

# Constrained optimum marker
if show_opt:
    sol = core.optimize_min_rth(A, ptot_max=ptot_max)
    if sol is not None:
        if mode == "sigma":
            zopt = float(core.predict_with_uncertainty(
                A, [sol["vp_vs"], sol["po"]], k=1.0)["r_th_sigma"][0])
        else:
            zopt = sol["r_th"] if mode == "rth" else sol["p_tot"]
        fig.add_trace(go.Scatter3d(
            x=[sol["vp_vs"]], y=[sol["po"]], z=[zopt], mode="markers",
            name="constrained optimum",
            marker=dict(size=8, color=C_ACCENT, symbol="diamond",
                        line=dict(width=1, color="white")),
            hovertemplate=f"OPTIMUM<br>{zlabel}={zopt:.3g}<extra></extra>"))

fig.update_layout(scene=dict(zaxis=dict(
    title=dict(text=zlabel, font=dict(family=FONT_TNR, size=15, color=TEXT)))))
style_scene(fig, height=660)
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

if mode == "rth":
    st.caption("Resistance surface - the design objective. Lowest values sit toward high "
               "Vp:Vs and low porosity. Drag to rotate; the floor shows projected contours.")
elif mode == "ptot":
    st.caption("Pressure-drop surface with the red constraint plane. Wherever the surface "
               "rises above the plane the design is infeasible; the plane-surface "
               "intersection is the feasibility boundary seen in 2-D on the Optimise page.")
else:
    st.caption("GPR predictive standard deviation of the thermal resistance. Valleys sit on "
               "the 49 FEM samples (the model is pinned there); ridges between and beyond "
               "samples mark where predictions are least certain. Peaks inside the feasible "
               "region are the best candidate locations for the NEXT FEM simulation, and a "
               "high \u03c3 at the optimum warns that the optimum itself is uncertain.")
