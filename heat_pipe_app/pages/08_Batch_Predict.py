"""Page 8 - Batch Prediction & Export: evaluate MANY design points at once
(upload a CSV or paste values), get predictions with uncertainty bands,
feasibility and domain trust for each, and download the results as CSV.
Turns the surrogate into a working tool for parametric studies and makes the
numbers behind the paper reproducible."""
import io
import numpy as np
import pandas as pd
import streamlit as st

import core
from app_utils import (get_assets, inject_css, page_header, PLOTLY_CONFIG,
                       SYM, MUTED, C_ACCENT)

st.set_page_config(page_title="Batch Predict", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Batch Prediction & Export",
            f"Evaluate many design points in one go. Upload a CSV (columns: vp_vs, po) or "
            f"paste values; get {SYM['r_th']}, {SYM['p_tot']} (with \u00b12\u03c3 bands), "
            f"{SYM['k_eq']}, feasibility and a domain-trust flag per point - and download "
            f"everything as CSV.")

MAX_ROWS = 2000

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.subheader("1 - Provide design points")
    up = st.file_uploader("CSV with columns vp_vs, po (extra columns are kept)",
                          type=["csv"])
    template = pd.DataFrame({"vp_vs": [0.30, 0.50, 0.73], "po": [0.55, 0.60, 0.43]})
    st.download_button("Download a template CSV",
                       template.to_csv(index=False).encode(),
                       file_name="batch_template.csv", mime="text/csv")
with c2:
    st.subheader("or paste values")
    pasted = st.text_area("One point per line: vp_vs, po",
                          placeholder="0.30, 0.55\n0.50, 0.60\n0.73, 0.43", height=130)
    ptot_max = st.number_input("Pressure-drop limit (Pa)",
                               value=core.PTOT_MAX_DEFAULT, step=100.0)


def _find_col(df, names):
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    return None


df_in, err = None, None
if up is not None:
    try:
        df_in = pd.read_csv(up)
        cvp = _find_col(df_in, ["vp_vs", "vp:vs", "vpvs", "vp/vs"])
        cpo = _find_col(df_in, ["po", "porosity", "epsilon", "eps"])
        if cvp is None or cpo is None:
            err = "Could not find vp_vs / po columns in the uploaded CSV."
        else:
            df_in = df_in.rename(columns={cvp: "vp_vs", cpo: "po"})
    except Exception as e:
        err = f"Could not read the CSV ({type(e).__name__})."
elif pasted.strip():
    try:
        rows = []
        for line in pasted.strip().splitlines():
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) != 2:
                raise ValueError(f"line '{line}' does not have exactly two numbers")
            rows.append(vals)
        df_in = pd.DataFrame(rows, columns=["vp_vs", "po"])
    except Exception as e:
        err = f"Could not parse the pasted values: {e}"

if err:
    st.error(err)
    st.stop()
if df_in is None:
    st.info("Upload a CSV or paste points above to run a batch prediction.")
    st.stop()
if len(df_in) > MAX_ROWS:
    st.warning(f"Input has {len(df_in)} rows; using the first {MAX_ROWS}.")
    df_in = df_in.head(MAX_ROWS)

X = df_in[["vp_vs", "po"]].to_numpy(dtype=float)
u = core.predict_with_uncertainty(A, X, k=2.0)

out = df_in.copy()
out["r_th_K_per_W"] = np.round(u["r_th"], 6)
if A.supports_std:
    out["r_th_lo_2sig"] = np.round(u["r_th_lo"], 6)
    out["r_th_hi_2sig"] = np.round(u["r_th_hi"], 6)
out["p_tot_Pa"] = np.round(u["p_tot"], 2)
if A.supports_std:
    out["p_tot_lo_2sig"] = np.round(u["p_tot_lo"], 2)
    out["p_tot_hi_2sig"] = np.round(u["p_tot_hi"], 2)
out["k_eq_W_per_mK"] = np.round([core.k_eq_from_rth(r) for r in u["r_th"]], 0)
out["feasible"] = u["p_tot"] <= ptot_max
levels, in_dom = [], []
for x in X:                                     # per-point trust flag
    s = core.domain_status(A, x)
    levels.append(s["level"])
    in_dom.append(s["in_box"])
out["domain_trust"] = levels
out["inside_bounds"] = in_dom

st.subheader("2 - Results")
n_feas = int(out["feasible"].sum())
st.markdown(f"**{len(out)} points evaluated** - {n_feas} feasible at "
            f"\u0394P\u209c\u2092\u209c \u2264 {ptot_max:.0f} Pa, "
            f"{len(out) - n_feas} infeasible. "
            f"{(np.array(levels) != 'high').sum()} point(s) carry a reduced-trust flag.",
            unsafe_allow_html=True)


def _style(row):
    if not row["feasible"]:
        return [f"color:{C_ACCENT}"] * len(row)
    if row["domain_trust"] != "high":
        return [f"color:#b9770e"] * len(row)
    return [""] * len(row)


st.dataframe(out.style.apply(_style, axis=1), use_container_width=True, height=380)
st.download_button("Download results as CSV",
                   out.to_csv(index=False).encode(),
                   file_name="batch_predictions.csv", mime="text/csv")
st.caption("Red rows violate the pressure-drop limit; amber rows sit in reduced-trust "
           "regions (near or beyond the sampled domain). Bands are GPR \u00b12\u03c3; "
           "p_tot bands are asymmetric because the model works in log space.")
