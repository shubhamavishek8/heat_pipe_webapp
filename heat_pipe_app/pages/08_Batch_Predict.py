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
                       SYM, MUTED, C_ACCENT, C_WARN)

st.set_page_config(page_title="Batch Predict", page_icon="\U0001f321\ufe0f", layout="wide")
inject_css()
A = get_assets()

page_header("Batch Prediction & Export",
            f"Evaluate many design points in one go. Upload a CSV (columns: vp_vs, po) or "
            f"paste values; get {SYM['r_th']}, {SYM['p_tot']} (with \u00b12\u03c3 bands), "
            f"{SYM['k_eq']}, feasibility and a domain-trust flag per point - and download "
            f"everything as CSV. Points outside the design bounds are skipped, not "
            f"extrapolated.")

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

df_in["vp_vs"] = np.round(df_in["vp_vs"].astype(float), 3)
df_in["po"] = np.round(df_in["po"].astype(float), 3)
X = df_in[["vp_vs", "po"]].to_numpy(dtype=float)
n = len(X)

# ---- bounds guard: do NOT evaluate points outside the design box ----------- #
(lb_vp, ub_vp) = core.BOUNDS["vp_vs"]
(lb_po, ub_po) = core.BOUNDS["po"]
eps = 1e-9
in_box = ((X[:, 0] >= lb_vp - eps) & (X[:, 0] <= ub_vp + eps) &
          (X[:, 1] >= lb_po - eps) & (X[:, 1] <= ub_po + eps))
n_out = int((~in_box).sum())

if n_out:
    detail = []
    for i in np.where(~in_box)[0][:6]:
        why = []
        if not (lb_vp - eps <= X[i, 0] <= ub_vp + eps):
            why.append(f"V\u209a:V\u209b = {X[i, 0]:.3f} \u2209 [{lb_vp:.3f}, {ub_vp:.3f}]")
        if not (lb_po - eps <= X[i, 1] <= ub_po + eps):
            why.append(f"\u03b5 = {X[i, 1]:.3f} \u2209 [{lb_po:.3f}, {ub_po:.3f}]")
        detail.append(f"- row {i}: " + "; ".join(why))
    more = f"\n- \u2026 and {n_out - 6} more" if n_out > 6 else ""
    st.warning(
        f"**{n_out} of {n} point(s) lie outside the design bounds** "
        f"(V\u209a:V\u209b \u2208 [{lb_vp:.3f}, {ub_vp:.3f}], "
        f"\u03b5 \u2208 [{lb_po:.3f}, {ub_po:.3f}]) and were **not evaluated** - predicting "
        f"there would be unreliable extrapolation:\n\n" + "\n".join(detail) + more)

# ---- evaluate ONLY the in-bounds points ------------------------------------ #
r = np.full(n, np.nan); r_lo = np.full(n, np.nan); r_hi = np.full(n, np.nan)
p = np.full(n, np.nan); p_lo = np.full(n, np.nan); p_hi = np.full(n, np.nan)
keq = np.full(n, np.nan)
trust = np.array(["outside bounds"] * n, dtype=object)
feas = np.array([None] * n, dtype=object)

if in_box.any():
    idxs = np.where(in_box)[0]
    u = core.predict_with_uncertainty(A, X[in_box], k=2.0)
    r[idxs] = np.round(u["r_th"], 3)
    p[idxs] = np.round(u["p_tot"], 2)
    keq[idxs] = np.round([core.k_eq_from_rth(v) for v in u["r_th"]], 2)
    if A.supports_std:
        r_lo[idxs] = np.round(u["r_th_lo"], 3); r_hi[idxs] = np.round(u["r_th_hi"], 3)
        p_lo[idxs] = np.round(u["p_tot_lo"], 2); p_hi[idxs] = np.round(u["p_tot_hi"], 2)
    for j, i in enumerate(idxs):
        feas[i] = bool(u["p_tot"][j] <= ptot_max)
        trust[i] = core.domain_status(A, X[i])["level"]

# ---- assemble the results table with readable, meaningful headers ---------- #
out = pd.DataFrame()
out["Vp:Vs"] = X[:, 0]
out["po (epsilon)"] = X[:, 1]
out["R_th (K/W)"] = r
if A.supports_std:
    out["R_th low -2sig (K/W)"] = r_lo
    out["R_th high +2sig (K/W)"] = r_hi
out["dP_tot (Pa)"] = p
if A.supports_std:
    out["dP_tot low -2sig (Pa)"] = p_lo
    out["dP_tot high +2sig (Pa)"] = p_hi
out["k_eq (W/m/K)"] = keq
out["feasible"] = feas
out["domain trust"] = trust
for extra in [c for c in df_in.columns if c not in ("vp_vs", "po")]:
    out[extra] = df_in[extra].values

st.subheader("2 - Results")
n_eval = int(in_box.sum())
n_feas = int(sum(1 for v in feas if v is True))
n_reduced = int(sum(1 for i, t in enumerate(trust) if in_box[i] and t != "high"))
st.markdown(
    f"**{n} point(s) submitted** - {n_eval} evaluated, {n_out} skipped (outside bounds). "
    f"Of the evaluated: {n_feas} feasible at \u0394P\u209c\u2092\u209c \u2264 {ptot_max:.2f} Pa, "
    f"{n_eval - n_feas} infeasible; {n_reduced} carry a reduced-trust flag.",
    unsafe_allow_html=True)

with st.expander("How are 'feasible' and 'domain trust' decided?"):
    st.markdown(
        f"""
**feasible** - purely the pressure-drop constraint: a point is feasible when its
*predicted* {SYM['p_tot']} is at or below your limit.

> `feasible = (predicted \u0394P_tot \u2264 limit)`

*Example (limit {ptot_max:.0f} Pa):* a point predicting {SYM['p_tot']} = 3062.59 Pa is
**feasible** (3062.59 \u2264 {ptot_max:.0f}); one predicting 17042.01 Pa is **infeasible**.

**domain trust** - how far the point sits from the data the surrogate actually
saw, combining two geometric checks and one statistical check:

1. **Inside the box?** Are both inputs within the trained bounds
   (V\u209a:V\u209b \u2208 [{lb_vp:.3f}, {ub_vp:.3f}], \u03b5 \u2208 [{lb_po:.3f}, {ub_po:.3f}])?
2. **Inside the convex hull?** Is the point inside the polygon spanned by the FEM
   sample locations (a tighter test than the box - the box can contain corners the
   samples never reach)?
3. **\u03c3 percentile.** Where does this point's GPR predictive standard deviation
   fall relative to the spread of \u03c3 over a dense in-domain grid? A high percentile
   means the model is unusually unsure here even if geometrically inside.

The three tiers:

| tier | condition |
|------|-----------|
| **low** | outside the box **or** \u03c3 in the top 1% (\u2265 99th percentile) |
| **moderate** | outside the convex hull (but inside the box) **or** \u03c3 \u2265 90th percentile |
| **high** | inside the hull **and** \u03c3 below the 90th percentile |

*Example:* (V\u209a:V\u209b = 0.20, \u03b5 = 0.50) is inside the box and hull with low \u03c3
\u2192 **high**. A point at \u03b5 = 0.20 fails check 1 (below the \u03b5 lower bound
{lb_po:.3f}); here it is reported as **outside bounds** and skipped entirely rather
than trusted.
""")

_cc = {"Vp:Vs": st.column_config.NumberColumn("V\u209a:V\u209b", format="%.3f"),
       "po (epsilon)": st.column_config.NumberColumn("\u03b5", format="%.3f"),
       "R_th (K/W)": st.column_config.NumberColumn("R\u209c\u2095 (K/W)", format="%.3f"),
       "R_th low -2sig (K/W)": st.column_config.NumberColumn("R\u209c\u2095 low (\u22122\u03c3)", format="%.3f"),
       "R_th high +2sig (K/W)": st.column_config.NumberColumn("R\u209c\u2095 high (+2\u03c3)", format="%.3f"),
       "dP_tot (Pa)": st.column_config.NumberColumn("\u0394P\u209c\u2092\u209c (Pa)", format="%.2f"),
       "dP_tot low -2sig (Pa)": st.column_config.NumberColumn("\u0394P\u209c\u2092\u209c low (\u22122\u03c3)", format="%.2f"),
       "dP_tot high +2sig (Pa)": st.column_config.NumberColumn("\u0394P\u209c\u2092\u209c high (+2\u03c3)", format="%.2f"),
       "k_eq (W/m/K)": st.column_config.NumberColumn("k_eq (W/m/K)", format="%.2f"),
       "feasible": st.column_config.TextColumn("feasible"),
       "domain trust": st.column_config.TextColumn("domain trust")}
st.dataframe(out, use_container_width=True, height=380,
             column_config={k: v for k, v in _cc.items() if k in out.columns})
st.download_button("Download results as CSV",
                   out.to_csv(index=False).encode(),
                   file_name="batch_predictions.csv", mime="text/csv")
st.caption("Low/high columns are the GPR \u00b12\u03c3 bounds (\u0394P\u209c\u2092\u209c "
           "asymmetric because the model works in log space). Rows outside the design "
           "bounds show blank predictions and 'outside bounds' trust - they are not "
           "evaluated.")
