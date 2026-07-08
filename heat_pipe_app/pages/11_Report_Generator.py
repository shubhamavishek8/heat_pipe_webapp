"""Page 11 - Report Generator: compile the study's headline results (provenance,
constrained optimum, local sensitivities, Pareto front) into a downloadable
one-page PDF - a reviewer-ready summary produced from the live surrogate."""
import numpy as np
import streamlit as st

import core
import report_utils
from app_utils import (get_assets, cached_pareto, inject_css, page_header,
                       metric_card, SYM, MUTED, C_OK)

st.set_page_config(page_title="Report", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Design Report Generator",
            f"One click compiles the surrogate provenance, the constrained optimum with "
            f"{SYM['k_eq']}, local sensitivities, and the Pareto front into a one-page PDF "
            f"generated from the live model - reviewer-ready and reproducible.")

c1, c2 = st.columns([1, 2], gap="large")
with c1:
    ptot_max = st.number_input("Pressure-drop limit for the report (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)
    go_btn = st.button("Generate PDF report", type="primary")

with c2:
    st.markdown(
        "**The report contains:**\n"
        "1. Surrogate provenance - best model, LOOCV scores, n, design space, fixed "
        "physical parameters\n"
        "2. The constrained optimum at your chosen limit - design, "
        "R\u209c\u2095*, \u0394P\u209c\u2092\u209c*, slack, k\u2091\u146f*, solver, "
        "grid cross-check, domain trust\n"
        "3. Local sensitivities at the optimum (central differences)\n"
        "4. The Pareto front as a vector chart with the limit and optimum marked")

if go_btn:
    with st.spinner("Solving and composing the report\u2026"):
        sol = core.optimize_min_rth(A, ptot_max=ptot_max)
        other = core.grid_search_min_rth(A, ptot_max=ptot_max) if sol is not None else None
        caps, Rf, Pf = cached_pareto(22)
        sens, status = None, None
        if sol is not None:
            vp, po = sol["vp_vs"], sol["po"]
            h_vp = 0.01 * (core.UB[0] - core.LB[0])
            h_po = 0.01 * (core.UB[1] - core.LB[1])
            cl = lambda v, lo, hi: float(np.clip(v, lo, hi))
            yv1 = core.predict(A, [cl(vp + h_vp, *core.BOUNDS["vp_vs"]), po])[0]
            yv0 = core.predict(A, [cl(vp - h_vp, *core.BOUNDS["vp_vs"]), po])[0]
            yp1 = core.predict(A, [vp, cl(po + h_po, *core.BOUNDS["po"])])[0]
            yp0 = core.predict(A, [vp, cl(po - h_po, *core.BOUNDS["po"])])[0]
            sens = {"dr_dvp": (yv1[0] - yv0[0]) / (2 * h_vp),
                    "dr_dpo": (yp1[0] - yp0[0]) / (2 * h_po),
                    "dp_dvp": (yv1[1] - yv0[1]) / (2 * h_vp),
                    "dp_dpo": (yp1[1] - yp0[1]) / (2 * h_po)}
            status = core.domain_status(A, [vp, po])
        pdf_bytes = report_utils.build_report_pdf(A, ptot_max, sol, other, caps, Rf,
                                                  sens, status)
    if sol is not None:
        metric_card("Optimum captured in the report",
                    f"{sol['r_th']:.4f} K/W",
                    sub=f"{SYM['vp_vs']} = {sol['vp_vs']:.4f}, {SYM['po']} = {sol['po']:.4f}, "
                        f"slack {sol['slack']:.1f} Pa", value_color=C_OK)
    st.download_button("Download design_report.pdf", pdf_bytes,
                       file_name="design_report.pdf", mime="application/pdf")
    st.caption("The PDF uses ASCII variable names (R_th, dP_tot) because core PDF fonts "
               "are latin-1; every number comes from the live surrogate at generation time.")
