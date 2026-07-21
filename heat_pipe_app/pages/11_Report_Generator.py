"""Page 11 - Report Generator: compile the study's headline results (provenance,
GPR performance metrics, the constrained optimum, a Monte-Carlo manufacturing
yield at that optimum, and the Pareto front) into a downloadable one-page PDF
typeset in a Times New Roman style (embedded Liberation Serif)."""
import numpy as np
import streamlit as st

import core
import report_utils
from app_utils import (get_assets, get_model_bank, cached_pareto, inject_css, page_header,
                       metric_card, SYM, MUTED, C_OK)

st.set_page_config(page_title="Report", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Design Report Generator",
            f"One click compiles the surrogate provenance, {A.model_name} performance "
            f"metrics, the constrained optimum with {SYM['k_eq']}, a manufacturing-yield "
            f"analysis at that optimum, and the Pareto front into a one-page PDF - "
            f"reviewer-ready and generated live from the model.")

c1, c2 = st.columns([1, 2], gap="large")
with c1:
    ptot_max = st.number_input("Pressure-drop limit for the report (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)
    tol_vp = st.number_input("Yield tolerance \u00b1 on Vp:Vs", min_value=0.001,
                             max_value=0.2, value=0.030, step=0.005, format="%.3f")
    tol_po = st.number_input("Yield tolerance \u00b1 on \u03b5", min_value=0.001,
                             max_value=0.1, value=0.020, step=0.005, format="%.3f")
    go_btn = st.button("Generate PDF report", type="primary")

with c2:
    st.markdown(
        f"**The report contains:**\n"
        f"1. Surrogate provenance - best model, n, design space, fixed physical parameters\n"
        f"2. Model provenance & performance - kernel, LOOCV R\u00b2, MAE, MSE, RMSE\n"
        f"3. The constrained optimum at your chosen limit - design, minimum "
        f"{SYM['r_th']}, {SYM['p_tot']}, slack, {SYM['k_eq']}, solver, grid cross-check, "
        f"domain trust\n"
        f"4. Manufacturing yield at the optimum - Monte-Carlo with your tolerances\n"
        f"5. The Pareto front - vector chart with major/minor grid, the limit and the "
        f"optimum marked",
        unsafe_allow_html=True)

if go_btn:
    with st.spinner("Solving, sampling and composing the report\u2026"):
        sol = core.optimize_min_rth(A, ptot_max=ptot_max)
        other = core.grid_search_min_rth(A, ptot_max=ptot_max) if sol is not None else None
        caps, Rf, Pf = cached_pareto(22)
        status = core.domain_status(A, [sol["vp_vs"], sol["po"]]) if sol is not None else None
        yld = (report_utils.yield_analysis(A, sol, ptot_max, tol_vp, tol_po)
               if sol is not None else None)
        # metrics: best-model manifest first, then the all-models manifest
        m = A.manifest
        metrics = {"r2": m.get("loocv_overall_r2"), "mae": m.get("loocv_overall_mae"),
                   "mse": m.get("loocv_overall_mse"), "basis": m.get("selection_basis"),
                   "sklearn_version": A.sklearn_version}
        bank = get_model_bank()
        if bank.available and any(metrics[k] is None for k in ("r2", "mae", "mse")):
            lo = bank.status.get(bank.best_name, {}).get("loocv", {})
            metrics["r2"] = metrics["r2"] if metrics["r2"] is not None else lo.get("overall_r2")
            metrics["mae"] = metrics["mae"] if metrics["mae"] is not None else lo.get("overall_mae")
            rm = lo.get("overall_rmse")
            if metrics["mse"] is None and rm is not None:
                metrics["mse"] = rm ** 2
        metrics["rmse"] = (metrics["mse"] ** 0.5) if metrics.get("mse") is not None else None
        try:
            metrics["kernel"] = str(A.model.kernel_)[:90]
        except Exception:
            metrics["kernel"] = None
        pdf_bytes = report_utils.build_report_pdf(A, ptot_max, sol, other, caps, Rf,
                                                  status, yld, metrics)
    if sol is not None:
        metric_card("Optimum captured in the report",
                    f"{sol['r_th']:.3f} K/W",
                    sub=f"{SYM['vp_vs']} = {sol['vp_vs']:.4f}, {SYM['po']} = {sol['po']:.4f}, "
                        f"slack {sol['slack']:.2f} Pa"
                        + (f" &nbsp;|&nbsp; yield {yld['yield_pct']:.1f}%" if yld else ""),
                    value_color=C_OK)
    st.download_button("Download design_report.pdf", pdf_bytes,
                       file_name="design_report.pdf", mime="application/pdf")
    st.caption("The PDF is typeset in Liberation Serif (metrically Times-compatible) with "
               "true italic variables and subscripts; every number comes from the live "
               "surrogate at generation time.")
