"""Page 10 - Model Validation: parity (predicted vs actual) and residual plots.

Two modes, auto-detected:
  A) TRUE HELD-OUT parity - if the training pipeline exported per-point LOOCV
     predictions (loocv_predictions.csv in artifacts/), this page plots real
     leave-one-out parity: the gold-standard visual evidence behind the
     manifest's LOOCV R-squared.
  B) IN-SAMPLE fallback - without that file, it plots surrogate-vs-FEM at the
     training points, clearly labelled as in-sample (a GPR interpolates its
     training data almost exactly, so this mode mainly demonstrates consistency,
     not generalisation).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import core
from app_utils import (get_assets, get_model_bank, inject_css, page_header,
                       metric_card, base_layout, PLOTLY_CONFIG, SYM, AX,
                       C_PRIMARY, C_ACCENT, C_OK, MUTED, TEXT, FONT_TNR, BORDER)

st.set_page_config(page_title="Model Validation", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Model Validation (Parity & Residuals)",
            f"Predicted vs actual for {SYM['r_th']} and {SYM['p_tot']}: the visual evidence "
            f"behind the R\u00b2 numbers. True leave-one-out parity is shown when the "
            f"pipeline has exported per-point LOOCV predictions; otherwise an in-sample "
            f"check is shown with an explicit caveat.")


def _metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, float(np.mean(np.abs(y - yhat))), float(np.sqrt(np.mean((y - yhat) ** 2)))


def _parity_fig(y, yhat, label):
    lo = min(np.min(y), np.min(yhat)); hi = max(np.max(y), np.max(yhat))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                             line=dict(color=MUTED, dash="dash"), name="perfect (y = x)"))
    fig.add_trace(go.Scatter(x=y, y=yhat, mode="markers", name="points",
                             marker=dict(size=7, color=C_PRIMARY, symbol="circle-open",
                                         line=dict(width=1.6))))
    fig.update_layout(xaxis_title=f"FEM (actual) - {label}",
                      yaxis_title=f"Surrogate (predicted) - {label}")
    return fig


def _residual_fig(x, res, xlabel, ylabel):
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dash"))
    fig.add_trace(go.Scatter(x=x, y=res, mode="markers", name="residuals",
                             marker=dict(size=7, color=C_ACCENT, symbol="circle-open",
                                         line=dict(width=1.6))))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def _find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


# --------------------------------------------------------------------------- #
# Mode detection
# --------------------------------------------------------------------------- #
loocv_path = core._glob1("*loocv*pred*.csv", "*loo*pred*.csv")
df = None
if loocv_path:
    try:
        df = pd.read_csv(loocv_path)
    except Exception:
        df = None

if df is not None:
    # ---------------- Mode A: true held-out parity -------------------------- #
    st.success(f"Found per-point LOOCV predictions ({loocv_path.split('/')[-1]}) - showing "
               f"TRUE held-out parity.")
    cm = _find_col(df, ["model"])
    models = sorted(df[cm].unique()) if cm else ["(single model)"]
    pick = st.selectbox("Model", models,
                        index=models.index(A.model_name) if A.model_name in models else 0)
    d = df[df[cm] == pick] if cm else df
    c = {k: _find_col(d, v) for k, v in {
        "vp": ["vp_vs", "vp:vs", "vpvs"], "po": ["po", "porosity", "eps"],
        "ra": ["r_th_actual", "rth_actual"], "rp": ["r_th_pred", "rth_pred"],
        "pa": ["p_tot_actual", "ptot_actual"], "pp": ["p_tot_pred", "ptot_pred"]}.items()}
    if None in (c["ra"], c["rp"], c["pa"], c["pp"]):
        st.error("The LOOCV file needs columns: model, vp_vs, po, r_th_actual, r_th_pred, "
                 "p_tot_actual, p_tot_pred (case-insensitive).")
        st.stop()
    ra, rp = d[c["ra"]].to_numpy(float), d[c["rp"]].to_numpy(float)
    pa, pp = d[c["pa"]].to_numpy(float), d[c["pp"]].to_numpy(float)
    caveat = f"Held-out (leave-one-out) - each point was predicted by a model that never saw it."
else:
    # ---------------- Mode B: in-sample fallback ---------------------------- #
    if A.X_train is None:
        st.info("Neither a LOOCV-predictions file nor training data is available in "
                "artifacts/ - add Raw_Data13.csv/.xlsx (or a loocv_predictions.csv) to "
                "enable this page.")
        st.stop()
    st.warning("No per-point LOOCV export found - showing IN-SAMPLE parity (surrogate vs "
               "FEM at the training points). A GPR interpolates its training data almost "
               "exactly, so treat this as a consistency check, NOT generalisation evidence; "
               "the held-out story is the LOOCV metrics in the manifest. See the expander "
               "below to export true LOOCV parity from the pipeline.")
    pick = A.model_name
    y = core.predict(A, A.X_train)
    ra, rp = A.y_train[:, 0], y[:, 0]
    pa, pp = A.y_train[:, 1], y[:, 1]
    d = pd.DataFrame({"vp_vs": A.X_train[:, 0], "po": A.X_train[:, 1]})
    c = {"vp": "vp_vs", "po": "po"}
    caveat = "In-sample - predictions at the very points the model was trained on."

r2r, maer, rmser = _metrics(ra, rp)
r2p, maep, rmsep = _metrics(pa, pp)

m1, m2, m3, m4 = st.columns(4)
with m1: metric_card(SYM["r_th"] + " R\u00b2", f"{r2r:.4f}", sub=caveat.split(" - ")[0])
with m2: metric_card(SYM["r_th"] + " MAE / RMSE", f"{maer:.4f} / {rmser:.4f}", sub="K/W")
with m3: metric_card(SYM["p_tot"] + " R\u00b2", f"{r2p:.4f}", sub=caveat.split(" - ")[0])
with m4: metric_card(SYM["p_tot"] + " MAE / RMSE", f"{maep:.1f} / {rmsep:.1f}", sub="Pa")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(base_layout(_parity_fig(ra, rp, "R_th (K/W)"), height=380),
                    use_container_width=True, config=PLOTLY_CONFIG)
with c2:
    st.plotly_chart(base_layout(_parity_fig(pa, pp, "\u0394P_tot (Pa)"), height=380),
                    use_container_width=True, config=PLOTLY_CONFIG)

if c.get("vp") is not None and c["vp"] in (d.columns if hasattr(d, "columns") else []):
    st.subheader("Residuals vs inputs")
    x_vp = d[c["vp"]].to_numpy(float)
    r1, r2_ = st.columns(2)
    with r1:
        st.plotly_chart(base_layout(_residual_fig(x_vp, rp - ra, AX["vp_vs"],
                        "residual R_th (K/W)"), height=320),
                        use_container_width=True, config=PLOTLY_CONFIG)
    with r2_:
        st.plotly_chart(base_layout(_residual_fig(x_vp, pp - pa, AX["vp_vs"],
                        "residual \u0394P_tot (Pa)"), height=320),
                        use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"{caveat} Structure in the residuals (trends, funnels) indicates where the "
               f"surrogate is biased; random scatter around zero is what a healthy fit "
               f"looks like.")

# manifest LOOCV table for all models, if available
bank = get_model_bank()
if bank.available:
    st.subheader("LOOCV scores of all trained models (from the training manifest)")
    rows = ""
    for name in bank.order:
        lo = bank.status[name].get("loocv", {})
        star = " \u2605" if name == bank.best_name else ""
        rows += (f"<tr style='border-bottom:1px solid {BORDER}'>"
                 f"<td style='padding:5px 10px'>{name}{star}</td>"
                 f"<td style='text-align:center'>{lo.get('overall_r2', float('nan')):.4f}</td>"
                 f"<td style='text-align:center'>{lo.get('overall_mae', float('nan')):.4g}</td>"
                 f"<td style='text-align:center'>{lo.get('overall_rmse', float('nan')):.4g}</td></tr>")
    st.markdown(f"<table style='width:70%;border-collapse:collapse;font-family:{FONT_TNR}'>"
                f"<tr style='border-bottom:2px solid {BORDER}'>"
                f"<th style='text-align:left;padding:5px 10px'>Model</th><th>LOOCV R\u00b2</th>"
                f"<th>MAE</th><th>RMSE</th></tr>{rows}</table>", unsafe_allow_html=True)

with st.expander("How to export TRUE LOOCV parity from the training pipeline"):
    st.markdown(
        "Add this at the end of the pipeline's Section 18.6 (where the leave-one-out "
        "predictions for each model are computed), adapt the two marked names to your "
        "arrays, and copy the resulting `loocv_predictions.csv` into `artifacts/`:")
    st.code(
        "# loo_actual: (n,2) array of held-out FEM values in ORIGINAL units\n"
        "# loo_pred[model_name]: (n,2) array of LOOCV predictions in ORIGINAL units\n"
        "import pandas as _pd\n"
        "_rows = []\n"
        "for _m in ALL_MODELS:\n"
        "    for _i in range(len(X_original)):\n"
        "        _rows.append({'model': _m,\n"
        "                      'vp_vs': X_original[_i, 0], 'po': X_original[_i, 1],\n"
        "                      'r_th_actual': loo_actual[_i, 0],  # <- adapt\n"
        "                      'r_th_pred' : loo_pred[_m][_i, 0], # <- adapt\n"
        "                      'p_tot_actual': loo_actual[_i, 1],\n"
        "                      'p_tot_pred' : loo_pred[_m][_i, 1]})\n"
        "_pd.DataFrame(_rows).to_csv('loocv_predictions.csv', index=False)\n"
        "print('\\u25ba LOOCV parity export \\u2192 loocv_predictions.csv')",
        language="python")
