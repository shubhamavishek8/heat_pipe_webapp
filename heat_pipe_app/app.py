"""app.py - Landing page for the Heat-Pipe Surrogate & Design Optimiser."""
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, inject_css, page_header, base_layout, add_training_points,
                       PLOTLY_CONFIG, SYM, AX, C_PRIMARY, C_ACCENT)

st.set_page_config(page_title="Heat-Pipe Surrogate Optimiser",
                   page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Heat-Pipe Surrogate &amp; Design Optimiser",
            "GPR surrogate of an FEM heat-pipe model - predict, optimise, and stress-test designs "
            "without rerunning the solver.")

st.markdown(
    f"""
This app wraps a **Gaussian Process Regression** surrogate trained on a finite-element
heat-pipe dataset. It maps two design variables to two performance outputs and lets you
explore the design space interactively. **No model is retrained** - every page loads the
exact surrogate selected by leave-one-out cross-validation in the offline pipeline.
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<table style="width:100%;border-collapse:collapse">
  <tr style="border-bottom:2px solid #dfe4ec">
    <th style="text-align:left;padding:6px">Design variables (inputs)</th>
    <th style="text-align:left;padding:6px">Performance outputs</th></tr>
  <tr style="border-bottom:1px solid #eef1f6">
    <td style="padding:6px">{SYM['vp_vs']} - primary/secondary wick volume ratio</td>
    <td style="padding:6px">{SYM['r_th']} - thermal resistance (K/W), <i>minimise</i></td></tr>
  <tr>
    <td style="padding:6px">{SYM['po']} - wick porosity</td>
    <td style="padding:6px">{SYM['p_tot']} - total pressure drop (Pa), <i>constrained</i></td></tr>
</table>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='margin-top:10px'><b>Optimisation goal:</b> minimise {SYM['r_th']} "
    f"subject to {SYM['p_tot']} &le; limit (default 4200 Pa).</p>",
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1.1], gap="large")

with c1:
    st.subheader("Pages")
    st.markdown(
        f"- **Predict** - forward prediction with uncertainty bands, a domain guard, and {SYM['k_eq']}\n"
        f"- **Optimise** - constrained min-{SYM['r_th']} (SLSQP) + {SYM['r_th']}/{SYM['p_tot']} trade-off front\n"
        "- **Tolerance Analysis** - manufacturing-tolerance propagation & yield\n"
        "- **3D Insight** - interactive response surface with the constraint plane",
        unsafe_allow_html=True,
    )

    st.subheader("Model provenance")
    m = A.manifest
    st.markdown(
        f"- **Surrogate:** `{m['best_model_name']}` (Mat\u00e9rn-2.5 + White kernel)\n"
        f"- **Selection basis:** {m['selection_basis']}\n"
        f"- **LOOCV overall R\u00b2:** `{m['loocv_overall_r2']:.4f}`\n"
        f"- **LOOCV MAE / RMSE:** `{m['loocv_overall_mae']:.3f}` / `{m['loocv_overall_mse']**0.5:.3f}`\n"
        f"- **Training samples:** n = {A.X_train.shape[0]}"
    )
    st.caption(
        "Small-sample caveat: with n=49 the surrogate is reliable only inside the sampled "
        "region. Every page shows a domain-of-validity indicator; heed it before trusting a number."
    )

with c2:
    st.subheader("Design space & FEM samples")
    lo_vp, hi_vp = core.BOUNDS["vp_vs"]
    lo_po, hi_po = core.BOUNDS["po"]
    fig = go.Figure()
    add_training_points(fig, A, marker_color=C_PRIMARY)
    fig.add_shape(type="rect", x0=lo_vp, x1=hi_vp, y0=lo_po, y1=hi_po,
                  line=dict(color=C_ACCENT, dash="dash"))
    fig.update_layout(xaxis_title=AX["vp_vs"], yaxis_title=AX["po"], showlegend=True)
    fig.update_xaxes(range=[lo_vp - 0.03, hi_vp + 0.03])
    fig.update_yaxes(range=[lo_po - 0.02, hi_po + 0.02])
    st.plotly_chart(base_layout(fig), use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Dashed box = design bounds (= dataset min/max). Predictions outside it are extrapolation.")
