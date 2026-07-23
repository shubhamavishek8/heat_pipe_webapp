# Hybrid Surrogate Model

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

## Updating the surrogate to a newly trained model (privacy-preserving)

The app is **self-configuring** - nothing dataset-specific is hardcoded. To deploy
a model trained on a new dataset, you never need to upload anything anywhere:

1. **Verify locally** (no data leaves your machine):
   ```bash
   python tools/inspect_artifacts.py /path/to/new/artifacts
   ```
   It reports the model type, the scikit-learn version it was pickled with, the
   input/output schema, the derived design-space bounds and sample count, and a
   COMPATIBLE / NOT COMPATIBLE verdict.

2. **Drop the new files** into `artifacts/` - the surrogate `.pkl`, the two scalers, the manifest JSON, AND the dataset file (e.g. `Raw_Data13.csv` or `Raw_Data13.xlsx`). The dataset file is the authoritative source for the design space, the sample count `n`, and the FEM-sample overlays everywhere in the app; columns are matched by name (Vp_Vs/po/R_th/P_tot, case-insensitive) or position. Filenames are auto-discovered; the model file is read from the manifest's `file` field.
   `scaler_y*.pkl`, and the manifest JSON). Filenames are auto-discovered; the
   model file is read from the manifest's `file` field when present.

3. **Pin scikit-learn** to the version the inspector printed (must match the
   version that pickled the model), e.g. `scikit-learn==1.7.2` in `requirements.txt`.

4. Push and redeploy. The app derives bounds and the sample count `n` directly
   from the model's training data; all pages, optimisers, sigma maps and the
   domain guard reconfigure automatically.

**Model-agnostic.** Any scikit-learn regressor (GPR, RandomForest, XGBoost, SVR, ...) works for prediction, optimisation and tolerance analysis. The manifest's `best_model_name`/`file` select which surrogate loads, and the app names it on the landing page. Predictive-uncertainty features (forecast bands, the uncertainty surface, the Next-Experiment page) activate only for a Gaussian Process (supports `predict(return_std=True)`) and otherwise switch off automatically with an in-app note. Training points/bounds/`n` come from a dataset CSV in `artifacts/` if present, else a GPR's `X_train_`, else a manifest `bounds`/`n_samples` field. Default pressure-drop constraint stays 4200 Pa.