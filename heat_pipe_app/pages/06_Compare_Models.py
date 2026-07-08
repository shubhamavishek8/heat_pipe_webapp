"""Page 6 - Compare Models: for one design point, show the prediction from EVERY
saved surrogate side by side, so a user/reviewer can judge which model behaves
best. The best model (LOOCV winner from the manifest) is highlighted, and each
model's LOOCV R2 is shown next to its live prediction."""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, get_model_bank, inject_css, page_header,
                       input_sliders, metric_card, base_layout, domain_badge,
                       PLOTLY_CONFIG, SYM, AX, C_PRIMARY, C_ACCENT, C_OK, C_WARN,
                       MUTED, TEXT, FONT_TNR, BORDER, CARD_BG)

st.set_page_config(page_title="Compare Models", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()
bank = get_model_bank()

page_header("All-Model Comparison",
            f"One design point, every saved surrogate. Compare each model's "
            f"{SYM['r_th']} and {SYM['p_tot']} prediction against the others and against "
            f"their leave-one-out cross-validation scores, to see which model is most "
            f"trustworthy.")

if not bank.available:
    st.info("No `all_models_manifest.json` was found in artifacts/. This page needs the "
            "multi-model artifacts written by Section 18.9 of the training pipeline "
            "(model_*.pkl for each surrogate, surrogate_model.keras for the ANN, and the "
            "manifest). Add them to artifacts/ and reload. The other pages work without them.")
    st.stop()

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Design point")
    vp, po = input_sliders(0.5, 0.6, key_prefix="cmp_")
    status = core.domain_status(A, [vp, po])
    domain_badge(status)

preds = bank.predict_all([vp, po])
sig = bank.gpr_sigma([vp, po])

# assemble rows in the manifest's order, available first
rows = []
for name in bank.order:
    ent = bank.status[name]
    r = preds.get(name)
    rows.append({
        "name": name,
        "avail": ent["available"],
        "reason": ent.get("reason", ""),
        "source": ent.get("source"),
        "r_th": (r[0] if r else None),
        "p_tot": (r[1] if r else None),
        "r2": ent.get("loocv", {}).get("overall_r2"),
        "is_best": (name == bank.best_name),
    })
avail_rows = [r for r in rows if r["avail"]]
if not avail_rows:
    st.warning("The manifest was found, but none of the listed model files could be "
               "loaded. Check that the model_*.pkl files sit next to the manifest in "
               "artifacts/ (see the Source column below for per-model reasons).")

with right:
    st.subheader("Predicted thermal resistance by model")
    if avail_rows:
        names = [r["name"] + (" \u2605" if r["is_best"] else "") for r in avail_rows]
        rth = [r["r_th"] for r in avail_rows]
        colors = [C_ACCENT if r["is_best"] else C_PRIMARY for r in avail_rows]
        fig = go.Figure(go.Bar(x=names, y=rth, marker_color=colors,
                               text=[f"{v:.4f}" for v in rth], textposition="outside",
                               cliponaxis=False))
        fig.update_layout(xaxis_title="", yaxis_title=AX["r_th"])
        fig = base_layout(fig, height=380, legend_below=False)
        fig.update_yaxes(range=[0, max(rth) * 1.18])   # headroom so labels never clip
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("Predicted pressure drop by model")
        pt = [r["p_tot"] for r in avail_rows]
        fig2 = go.Figure(go.Bar(x=names, y=pt, marker_color=colors,
                                text=[f"{v:.0f}" for v in pt], textposition="outside",
                                cliponaxis=False))
        fig2.update_layout(xaxis_title="", yaxis_title=AX["p_tot"])
        fig2 = base_layout(fig2, height=380, legend_below=False)
        fig2.update_yaxes(range=[0, max(pt) * 1.18])
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)

# ----- comparison table --------------------------------------------------- #
st.subheader("Side-by-side table")
hdr = (f"<tr style='border-bottom:2px solid {BORDER}'>"
       f"<th style='text-align:left;padding:6px 10px'>Model</th>"
       f"<th style='padding:6px 10px'>{SYM['r_th']} (K/W)</th>"
       f"<th style='padding:6px 10px'>{SYM['p_tot']} (Pa)</th>"
       f"<th style='padding:6px 10px'>{SYM['k_eq']} (W m\u207b\u00b9 K\u207b\u00b9)</th>"
       f"<th style='padding:6px 10px'>\u0394 vs best</th>"
       f"<th style='padding:6px 10px'>LOOCV R\u00b2</th>"
       f"<th style='text-align:left;padding:6px 10px'>Source</th></tr>")
best_rth = next((r["r_th"] for r in rows if r["is_best"] and r["avail"]), None)
body = ""
for r in rows:
    hl = f"background:{CARD_BG};font-weight:700" if r["is_best"] else ""
    star = " \u2605 best" if r["is_best"] else ""
    if r["avail"]:
        keq = core.k_eq_from_rth(r["r_th"])
        r2 = f"{r['r2']:.4f}" if r["r2"] is not None else "\u2013"
        if best_rth and not r["is_best"]:
            dvb = f"{(r['r_th']-best_rth)/best_rth*100:+.2f}%"
        else:
            dvb = "\u2013" if not r["is_best"] else "ref"
        body += (f"<tr style='border-bottom:1px solid {BORDER};{hl}'>"
                 f"<td style='padding:6px 10px'>{r['name']}{star}</td>"
                 f"<td style='text-align:center'>{r['r_th']:.4f}</td>"
                 f"<td style='text-align:center'>{r['p_tot']:.1f}</td>"
                 f"<td style='text-align:center'>{keq:,.0f}</td>"
                 f"<td style='text-align:center'>{dvb}</td>"
                 f"<td style='text-align:center'>{r2}</td>"
                 f"<td style='padding:6px 10px;color:{MUTED}'>{r['source']}</td></tr>")
    else:
        body += (f"<tr style='border-bottom:1px solid {BORDER};color:{MUTED}'>"
                 f"<td style='padding:6px 10px'>{r['name']}</td>"
                 f"<td colspan='5' style='text-align:center'>unavailable</td>"
                 f"<td style='padding:6px 10px'>{r['reason']}</td></tr>")
st.markdown(f"<table style='width:100%;border-collapse:collapse;font-family:{FONT_TNR}'>"
            f"{hdr}{body}</table>", unsafe_allow_html=True)

if sig is not None:
    st.caption(f"GPR is the only model with a predictive interval: its \u00b12\u03c3 band on "
               f"{SYM['p_tot']} here is [{sig[0]:.1f}, {sig[1]:.1f}] Pa. The other models return "
               f"point estimates only. Star marks the pipeline's LOOCV-selected best model.")

unavailable = [r for r in rows if not r["avail"]]
if unavailable:
    names = ", ".join(r["name"] for r in unavailable)
    st.caption(f"Not shown: {names}. ANN needs TensorFlow and XGBoost needs the xgboost "
               f"package at load time; on the 1 GB Streamlit Community Cloud tier these are "
               f"omitted by default. Add them to requirements.txt to load those models live, "
               f"or drop a precomputed all_models_grid.npz into artifacts/ to compare them "
               f"with no heavy dependencies.")

st.caption("Note: LOOCV R\u00b2 (from training) measures held-out accuracy over the whole "
           "dataset; the per-point predictions above show how the models diverge at a "
           "specific design - useful where they disagree most, typically in data-sparse "
           "regions. Read them together.")
