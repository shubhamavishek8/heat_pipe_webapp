# Heat-Pipe Surrogate & Design Optimiser

A Streamlit web app that wraps a **Gaussian Process Regression (GPR)** surrogate of an
FEM heat-pipe model. It predicts thermal resistance (R_th) and total pressure drop
(ΔP_tot) from two design variables and lets you optimise, inverse-design, and
stress-test designs interactively. **No model is retrained at runtime** - the app loads
the exact surrogate selected by leave-one-out cross-validation in the offline pipeline.

## Inputs / outputs

| Inputs | Outputs |
|---|---|
| `vp_vs` - Vp/Vs wick volume ratio ∈ [0.05, 0.95] | `r_th` - thermal resistance (K/W), minimise |
| `po` - porosity ε ∈ [0.40, 0.77] | `p_tot` - total pressure drop (Pa), constrained ≤ 4200 |

## Fixed physical parameters

Equivalent thermal conductivity uses Q = 40 W, A_c = 9.31e-6 m^2, L_eff = 0.14 m:
k_eq = L_eff / (A_c * R_th).

## Pages

- **Predict** - forward prediction with a ±kσ predictive interval (linear for R_th,
  asymmetric log-space band for ΔP_tot) and a domain-of-validity guard.
- **Optimise** - constrained `min R_th s.t. ΔP_tot ≤ limit` via a user-selectable solver:
  multi-start SLSQP or a step-defined grid search (range fixed to the data bounds; the user
  sets the lattice spacing, default Δ(Vp:Vs)=0.05, Δε=0.01), plus an ε-constraint Pareto front.
- **Inverse Design** - the full region of designs meeting user targets on both outputs.
- **Robust Optimum** - risk-aware design: minimise `R_th + κ·σ(R_th)` under a chance
  constraint on ΔP_tot.
- **Tolerance Analysis** - manufacturing-tolerance propagation and yield.
- **Monte Carlo** - general per-input distribution propagation, optionally including the
  GPR's own predictive σ.

## Project layout

```
heat_pipe_app/
├── app.py                 # landing page + provenance (entry point)
├── core.py                # all numerics: prediction, σ-propagation, optimisers (no streamlit)
├── app_utils.py           # streamlit layer: cached loaders, theming, widgets
├── pages/                 # one file per feature page (auto-discovered by Streamlit)
├── artifacts/             # the saved GPR + scalers + manifest
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository (artifacts included - they are small).
2. On https://share.streamlit.io, create an app pointing at `app.py`.
3. No secrets or system packages are required.

### Key engineering notes

- **scikit-learn is pinned to 1.7.2** to match the version that pickled the artifacts.
- The 49 FEM training points are recovered from the GPR's internal `X_train_`, so no
  dataset CSV ships with the app.
- Grid evaluations and the Pareto sweep are cached (`st.cache_data`); the model and
  scalers load once (`st.cache_resource`).
- The surrogate is trustworthy only **inside** the sampled region (n=49). Every page
  surfaces a domain indicator - heed it before trusting an extrapolated number.
