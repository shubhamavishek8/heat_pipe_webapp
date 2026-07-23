"""
core.py - Single source of truth for the Hybrid Surrogate Model web app.

Everything numerically sensitive lives here and is imported by every page so
that no page can reimplement the prediction / uncertainty / constraint logic
loosely:

    * artifact loading (cached)                       -> load_assets()
    * forward prediction in ORIGINAL units            -> predict()
    * uncertainty propagation (linear + log-space)    -> predict_with_uncertainty()
    * design-domain / extrapolation guard             -> domain_status()
    * constrained optimisation (SLSQP multi-start)    -> optimize_min_rth()
    * epsilon-constraint Pareto front                 -> pareto_front()
    * robust (chance-constrained) optimisation        -> optimize_robust()

Pipeline facts hard-wired from the training script (ANN_ML_Optimization_v7.py):
    inputs  : x = [vp_vs, po]        (Vp/Vs wick ratio, porosity epsilon)
    outputs : y = [r_th, p_tot]      (thermal resistance, total pressure drop [Pa])
    transform: y was log1p() on column 1 (p_tot) BEFORE StandardScaler.
               -> invert with expm1() on column 1 after inverse_transform.
    surrogate: single multi-output GaussianProcessRegressor (Matern-2.5 + White).
               predict(return_std=True) returns std of shape (n, 2) whose two
               columns are IDENTICAL (a single shared GP variance that depends
               only on X and the kernel). Physical per-output sigma differs only
               through scaler_y.scale_ and the log transform on p_tot.
"""

from __future__ import annotations
import os
import json
import warnings
import numpy as np
import joblib
import glob
from scipy.optimize import minimize
from scipy.spatial import Delaunay

# --------------------------------------------------------------------------- #
# Constants pinned from the training pipeline
# --------------------------------------------------------------------------- #
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

LOG_TRANSFORM_COL = 1          # p_tot was log1p-transformed
INPUT_NAMES  = ["vp_vs", "po"]
OUTPUT_NAMES = ["r_th", "p_tot"]

# Pretty labels for the UI
PRETTY = {
    "vp_vs": "Vp/Vs  (wick volume ratio)",
    "po":    "\u03b5  (porosity)",
    "r_th":  "R\u209c\u2095  (thermal resistance, K/W)",
    "p_tot": "\u0394P\u209c\u2092\u209c  (total pressure drop, Pa)",
}

# Design-space bounds: DEFAULTS only - Assets.__init__ overwrites these with
# values derived from the loaded model's training data (per-input min/max).
BOUNDS = {
    "vp_vs": (0.05, 0.95),
    "po":    (0.40, 0.77),
}
LB = np.array([BOUNDS["vp_vs"][0], BOUNDS["po"][0]])
UB = np.array([BOUNDS["vp_vs"][1], BOUNDS["po"][1]])

PTOT_MAX_DEFAULT = 4358.0      # default constraint limit (Pa)
FEAS_TOL = 1e-3                # engineering tolerance on p_tot

# Fixed heat-pipe parameters for equivalent thermal conductivity (user-specified)
Q_WATT = 40.0                  # total power (W)
A_CROSS = 9.31e-6              # cross-sectional area (m^2)
L_EFF = 0.14                   # effective length (m)


def k_eq_from_rth(r_th: float) -> float:
    """Equivalent thermal conductivity k_eq = L_eff / (A_c * R_th)  [W/m/K].

    Derivation: R_th = dT/Q and k_eq = Q*L_eff/(A_c*dT) => Q cancels; the
    fixed Q = 40 W enters only through dT = Q*R_th shown alongside."""
    return L_EFF / (A_CROSS * r_th) if r_th > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Artifact auto-discovery - no filenames hard-coded; drop new artifacts in and go
# --------------------------------------------------------------------------- #
def _glob1(*patterns):
    """First file in ARTIFACT_DIR matching any of the glob patterns, else None."""
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(ARTIFACT_DIR, pat)))
        if hits:
            return hits[0]
    return None


def discover_artifacts():
    """Locate manifest, model and scaler files by convention so that swapping in
    a newly trained surrogate needs no code change. The model path is taken from
    the manifest's "file" field when present, else inferred. Returns
    (model_path, scaler_X_path, scaler_y_path, manifest_path)."""
    manifest_path = _glob1("*manifest*.json", "*.json")
    model_path = None
    if manifest_path and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as fh:
                mf = json.load(fh)
            if mf.get("file"):
                cand = os.path.join(ARTIFACT_DIR, os.path.basename(mf["file"]))
                if os.path.exists(cand):
                    model_path = cand
        except Exception:
            pass
    if model_path is None:
        model_path = _glob1("best_surrogate*.pkl", "*surrogate*.pkl", "model*.pkl")
    sx_path = _glob1("scaler_X*.pkl", "scaler_x*.pkl", "*scalerX*.pkl", "x_scaler*.pkl")
    sy_path = _glob1("scaler_y*.pkl", "*scalerY*.pkl", "y_scaler*.pkl")
    missing = [n for n, p in [("manifest JSON", manifest_path), ("surrogate .pkl", model_path),
                              ("scaler_X .pkl", sx_path), ("scaler_y .pkl", sy_path)] if p is None]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) in '{ARTIFACT_DIR}': {', '.join(missing)}. "
            f"Expected a manifest JSON, a surrogate .pkl, and scaler_X*/scaler_y* .pkl.")
    # Dataset file (e.g. Raw_Data13.csv / .xlsx) - authoritative source for the
    # design space, sample count and FEM-point overlays for ANY model type.
    data_path = _glob1("*raw*data*.csv", "*raw*data*.xlsx", "*data*.csv", "*data*.xlsx",
                       "*raw*.csv", "*raw*.xlsx", "*dataset*.csv", "*dataset*.xlsx",
                       "*.csv", "*.xlsx")
    return model_path, sx_path, sy_path, manifest_path, data_path


# --------------------------------------------------------------------------- #
# Asset loading (self-configuring: bounds and sample count derived from the data)
# --------------------------------------------------------------------------- #
class Assets:
    """Loads the surrogate, scalers and manifest, then DERIVES everything that
    depends on the dataset (design-space bounds, sample count) automatically.

    Model-agnostic: ANY scikit-learn regressor works for point predictions and
    every optimiser. Predictive-uncertainty features (bands, sigma maps, the
    Next-Experiment page) light up only when the surrogate supports
    predict(return_std=True) - i.e. a Gaussian Process - and otherwise degrade
    gracefully with a clear in-app note rather than crashing.

    Training-data source priority (for overlays / bounds / sample count):
        1. a dataset CSV in artifacts/      (works for every model type)
        2. the model's own X_train_/y_train_ (Gaussian Process only)
        3. manifest 'bounds' + 'n_samples'  (no overlays, but the app still runs)
    """

    def __init__(self):
        global BOUNDS, LB, UB

        model_path, sx_path, sy_path, manifest_path, csv_path = discover_artifacts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # tolerate sklearn version mismatch warning
            self.model    = joblib.load(model_path)
            self.scaler_X = joblib.load(sx_path)
            self.scaler_y = joblib.load(sy_path)
        with open(manifest_path) as fh:
            self.manifest = json.load(fh)

        self.model_name = self.manifest.get("best_model_name", type(self.model).__name__)
        self.model_class = type(self.model).__name__
        self.sklearn_version = getattr(self.model, "_sklearn_version", None)

        # ---- validate 2-in / 2-out via an actual prediction ------------------
        n_in = int(getattr(self.scaler_X, "n_features_in_", 2))
        probe = np.zeros((1, n_in))           # a valid point in SCALED input space
        y_probe = np.atleast_2d(self.model.predict(probe))
        if n_in != 2 or y_probe.shape[1] != 2:
            raise ValueError(
                f"This app supports exactly 2 inputs and 2 outputs; the loaded model has "
                f"{n_in} inputs / {y_probe.shape[1]} outputs.")

        # ---- predictive-uncertainty capability (no crash if unsupported) -----
        self.supports_std = True
        try:
            self.model.predict(probe, return_std=True)
        except Exception:
            self.supports_std = False

        # ---- training data: dataset file -> GPR internals -> manifest --------
        self.X_train, self.y_train, self.data_source = self._load_training_data(csv_path)
        self.n = int(self.X_train.shape[0]) if self.X_train is not None \
            else int(self.manifest.get("n_samples", 0)) or None

        # ---- design-space bounds: DATA FILE first (Raw_Data13), then manifest -
        mb = self.manifest.get("bounds")
        if self.X_train is not None:
            BOUNDS = {"vp_vs": (float(self.X_train[:, 0].min()), float(self.X_train[:, 0].max())),
                      "po":    (float(self.X_train[:, 1].min()), float(self.X_train[:, 1].max()))}
        elif isinstance(mb, dict) and all(k in mb for k in ("vp_vs", "po")):
            BOUNDS = {k: (float(mb[k][0]), float(mb[k][1])) for k in ("vp_vs", "po")}
        else:
            raise RuntimeError(
                "Could not determine design-space bounds. Put the dataset file (e.g. "
                "Raw_Data13.csv or .xlsx) in artifacts/, deploy a Gaussian Process "
                "(exposes X_train_), or add a 'bounds' field to the manifest.")
        LB = np.array([BOUNDS["vp_vs"][0], BOUNDS["po"][0]])
        UB = np.array([BOUNDS["vp_vs"][1], BOUNDS["po"][1]])
        self.bounds, self.LB, self.UB = BOUNDS, LB, UB

        # ---- reference sigma distribution + convex hull (only if available) --
        self._sigma_ref = None
        if self.supports_std:
            gx = np.linspace(*BOUNDS["vp_vs"], 40)
            gy = np.linspace(*BOUNDS["po"], 40)
            GX, GY = np.meshgrid(gx, gy)
            grid = np.column_stack([GX.ravel(), GY.ravel()])
            _, sig_scaled = self._raw_predict(grid, return_std=True)
            self._sigma_ref = np.sort(sig_scaled)
        self._tri = None
        if self.X_train is not None:
            try:
                self._tri = Delaunay(self.X_train)
            except Exception:
                self._tri = None

    # ---- training-data loader -------------------------------------------- #
    def _load_training_data(self, data_path):
        """Return (X_train, y_train, source_str) in ORIGINAL units, or (None, None, 'none').

        Reads the dataset file (Raw_Data13.csv or .xlsx). Columns are matched by
        the manifest's input/output names, then canonical aliases, then position
        (first two columns = inputs, next two = outputs)."""
        if data_path and os.path.exists(data_path):
            try:
                header, data = self._read_table(data_path)

                def col(names):
                    for nm in names:
                        if nm in header:
                            return header.index(nm)
                    return None
                in_names = [n.lower() for n in self.manifest.get("input_names", [])]
                out_names = [n.lower() for n in self.manifest.get("output_names", [])]
                ix0 = col(in_names[:1] or []);  ix0 = ix0 if ix0 is not None else col(["vp_vs", "vp:vs", "vpvs", "vp/vs"])
                ix1 = col(in_names[1:2] or []); ix1 = ix1 if ix1 is not None else col(["po", "porosity", "epsilon", "eps"])
                oy0 = col(out_names[:1] or []); oy0 = oy0 if oy0 is not None else col(["r_th", "rth", "thermal_resistance"])
                oy1 = col(out_names[1:2] or []); oy1 = oy1 if oy1 is not None else col(["p_tot", "ptot", "dp_tot", "pressure_drop", "delta_p"])
                if None not in (ix0, ix1, oy0, oy1):
                    X = data[:, [ix0, ix1]]; Y = data[:, [oy0, oy1]]
                else:                       # positional fallback: first 2 in, next 2 out
                    X = data[:, :2]; Y = data[:, 2:4]
                src = "XLSX" if data_path.lower().endswith((".xlsx", ".xls")) else "CSV"
                return X, Y, f"{src} ({os.path.basename(data_path)})"
            except Exception:
                pass
        # Gaussian Process exposes its own training data (scaled-log).
        if hasattr(self.model, "X_train_") and hasattr(self.model, "y_train_"):
            X = self.scaler_X.inverse_transform(self.model.X_train_)
            y = self.scaler_y.inverse_transform(self.model.y_train_).copy()
            y[:, LOG_TRANSFORM_COL] = np.expm1(y[:, LOG_TRANSFORM_COL])
            return X, y, "model.X_train_"
        return None, None, "none"

    @staticmethod
    def _read_table(path):
        """Read a CSV or XLSX into (header_lowercased_list, float_ndarray)."""
        if path.lower().endswith((".xlsx", ".xls")):
            try:
                import pandas as pd
                df = pd.read_excel(path)
            except Exception:
                import openpyxl
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                ws = wb.active
                rows = [[c.value for c in r] for r in ws.iter_rows()]
                import pandas as pd
                df = pd.DataFrame(rows[1:], columns=rows[0])
            header = [str(h).strip().lower() for h in df.columns]
            data = df.to_numpy(dtype=float)
            return header, data
        # CSV
        import csv as _csv
        with open(path, newline="") as fh:
            rows = [r for r in _csv.reader(fh) if r]
        header = [h.strip().lower() for h in rows[0]]
        data = np.array([[float(v) for v in r] for r in rows[1:]], dtype=float)
        return header, data

    # -- low-level prediction in SCALED space ------------------------------- #
    def _raw_predict(self, X_raw, return_std=False):
        X_raw = np.atleast_2d(np.asarray(X_raw, dtype=float))
        Xs = self.scaler_X.transform(X_raw)
        if return_std:
            y_s, std_s = self.model.predict(Xs, return_std=True)
            std_s = np.asarray(std_s)
            # std comes back (n, 2) with identical columns -> take the shared scalar
            sig_scaled = std_s[:, 0] if std_s.ndim == 2 else std_s
            return y_s, sig_scaled
        return self.model.predict(Xs)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict(assets: Assets, X_raw) -> np.ndarray:
    """Forward surrogate in ORIGINAL units. Returns (n, 2) = [r_th, p_tot]."""
    y_s = assets._raw_predict(X_raw, return_std=False)
    y = assets.scaler_y.inverse_transform(y_s)
    y[:, LOG_TRANSFORM_COL] = np.expm1(y[:, LOG_TRANSFORM_COL])
    return y


def predict_with_uncertainty(assets: Assets, X_raw, k: float = 2.0):
    """
    Forward prediction plus a k-sigma predictive interval in ORIGINAL units.

    r_th  : StandardScaler is linear  -> symmetric band, sigma = sig_scaled * scale_[0]
    p_tot : lives in log1p space      -> ASYMMETRIC band; transform the +/- k*sigma
            log-space bounds through expm1() (do NOT multiply the physical std).

    Returns dict with arrays of shape (n,):
        r_th, r_th_sigma, r_th_lo, r_th_hi,
        p_tot, p_tot_lo, p_tot_hi, sigma_scaled

    If the surrogate does not support predictive std (not a Gaussian Process),
    sigma is zero and the bands collapse to the mean; callers should check
    assets.supports_std before presenting an interval.
    """
    X_raw = np.atleast_2d(np.asarray(X_raw, dtype=float))
    sc = assets.scaler_y
    if not assets.supports_std:
        y_s = assets._raw_predict(X_raw, return_std=False)
        rth_mean = y_s[:, 0] * sc.scale_[0] + sc.mean_[0]
        ptot_mean = np.expm1(y_s[:, 1] * sc.scale_[1] + sc.mean_[1])
        z = np.zeros_like(rth_mean)
        return {"r_th": rth_mean, "r_th_sigma": z, "r_th_lo": rth_mean, "r_th_hi": rth_mean,
                "p_tot": ptot_mean, "p_tot_lo": ptot_mean, "p_tot_hi": ptot_mean,
                "sigma_scaled": z}
    y_s, sig_scaled = assets._raw_predict(X_raw, return_std=True)

    # r_th (linear)
    rth_mean  = y_s[:, 0] * sc.scale_[0] + sc.mean_[0]
    rth_sigma = sig_scaled * sc.scale_[0]
    rth_lo    = rth_mean - k * rth_sigma
    rth_hi    = rth_mean + k * rth_sigma

    # p_tot (log1p space -> asymmetric)
    mu_log  = y_s[:, 1] * sc.scale_[1] + sc.mean_[1]
    sig_log = sig_scaled * sc.scale_[1]
    ptot_mean = np.expm1(mu_log)
    ptot_lo   = np.expm1(mu_log - k * sig_log)
    ptot_hi   = np.expm1(mu_log + k * sig_log)

    return {
        "r_th": rth_mean, "r_th_sigma": rth_sigma, "r_th_lo": rth_lo, "r_th_hi": rth_hi,
        "p_tot": ptot_mean, "p_tot_lo": ptot_lo, "p_tot_hi": ptot_hi,
        "sigma_scaled": sig_scaled,
    }


# --------------------------------------------------------------------------- #
# Extrapolation / domain guard
# --------------------------------------------------------------------------- #
def domain_status(assets: Assets, x):
    """
    Classify how trustworthy a single query point is.

    Returns dict:
        in_box   : within axis-aligned design bounds
        in_hull  : inside the convex hull of the training points
        sigma    : scaled predictive std at the point
        pct      : percentile of that sigma vs the in-domain reference distribution
        level    : 'high' | 'moderate' | 'low'  (traffic light)
    """
    x = np.asarray(x, dtype=float).reshape(1, -1)
    in_box = bool(np.all(x[0] >= LB - 1e-12) and np.all(x[0] <= UB + 1e-12))
    if assets._tri is not None:
        in_hull = bool(assets._tri.find_simplex(x)[0] >= 0)
    else:
        in_hull = in_box

    # Soft signal from GPR predictive std, when available; else geometry only.
    if assets.supports_std and assets._sigma_ref is not None:
        _, sig = assets._raw_predict(x, return_std=True)
        sig = float(sig[0])
        ref = assets._sigma_ref
        pct = float(np.searchsorted(ref, sig) / len(ref) * 100.0)
        if (not in_box) or pct >= 99.0:
            level = "low"
        elif (not in_hull) or pct >= 90.0:
            level = "moderate"
        else:
            level = "high"
    else:
        sig, pct = float("nan"), float("nan")
        level = "high" if in_hull else ("moderate" if in_box else "low")

    return {"in_box": in_box, "in_hull": in_hull, "sigma": sig, "pct": pct, "level": level}


# --------------------------------------------------------------------------- #
# Optimisation helpers
# --------------------------------------------------------------------------- #
def _r_th(assets, x):
    return float(predict(assets, x)[0, 0])


def _p_tot(assets, x):
    return float(predict(assets, x)[0, 1])


def optimize_min_rth(assets: Assets, ptot_max: float = PTOT_MAX_DEFAULT,
                     n_starts: int = 12, seed: int = 0):
    """
    Headline solver: min r_th(vp_vs, po) s.t. p_tot <= ptot_max, box bounds.
    SLSQP from n_starts Latin-hypercube-ish random starts; best feasible wins.
    Returns dict or None if no feasible solution found.
    """
    rng = np.random.default_rng(seed)
    starts = np.column_stack([
        rng.uniform(LB[0], UB[0], n_starts),
        rng.uniform(LB[1], UB[1], n_starts),
    ])
    bounds = [(LB[0], UB[0]), (LB[1], UB[1])]
    cons = [{"type": "ineq", "fun": lambda x: ptot_max - _p_tot(assets, x)}]

    best_x, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(lambda x: _r_th(assets, x), x0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            if not res.success:
                continue
            xc = np.clip(res.x, LB, UB)
            r, p = predict(assets, xc)[0]
            if p <= ptot_max + FEAS_TOL and r < best_r:
                best_x, best_r = xc, r
        except Exception:
            continue

    if best_x is None:
        return None
    r, p = predict(assets, best_x)[0]
    return {"vp_vs": float(best_x[0]), "po": float(best_x[1]),
            "r_th": float(r), "p_tot": float(p),
            "slack": float(ptot_max - p), "ptot_max": float(ptot_max),
            "method": "SLSQP (multi-start)", "n_starts": int(n_starts)}


def grid_search_min_rth(assets: Assets, ptot_max: float = PTOT_MAX_DEFAULT,
                        step_vp: float = 0.05, step_po: float = 0.01):
    """
    Exhaustive grid search defined by STEP SIZE (gap between lattice points),
    matching the original pipeline's range-and-distance formulation. The range
    is fixed to the data bounds; the user controls only the spacing.

    step_vp : lattice spacing along Vp:Vs   (default 0.05, as in the script)
    step_po : lattice spacing along porosity (default 0.01, as in the script)

    The upper bound is always included even when (ub - lb) is not an exact
    multiple of the step. Deterministic and global on the lattice; accuracy is
    limited by the spacing rather than solver tolerance. Same result schema as
    optimize_min_rth.
    """
    def axis(lb, ub, step):
        vals = np.arange(lb, ub + 1e-12, step)
        if vals.size == 0 or vals[-1] < ub - 1e-9:
            vals = np.append(vals, ub)
        return vals

    gx = axis(LB[0], UB[0], float(step_vp))
    gy = axis(LB[1], UB[1], float(step_po))
    VP, PO = np.meshgrid(gx, gy)
    pts = np.column_stack([VP.ravel(), PO.ravel()])
    y = predict(assets, pts)
    R = y[:, 0].reshape(VP.shape)
    P = y[:, 1].reshape(VP.shape)

    feasible = P <= ptot_max + FEAS_TOL
    if not feasible.any():
        return None
    R_masked = np.where(feasible, R, np.inf)
    iy, ix = np.unravel_index(int(np.argmin(R_masked)), R_masked.shape)
    vp, po = float(VP[iy, ix]), float(PO[iy, ix])
    r, p = float(R[iy, ix]), float(P[iy, ix])
    return {"vp_vs": vp, "po": po, "r_th": r, "p_tot": p,
            "slack": float(ptot_max - p), "ptot_max": float(ptot_max),
            "method": "Grid search",
            "step": (float(step_vp), float(step_po)),
            "grid": (int(gx.size), int(gy.size)),
            "n_eval": int(VP.size), "n_feasible": int(feasible.sum())}


def pareto_front(assets: Assets, n: int = 25, seed: int = 0):
    """
    Epsilon-constraint Pareto front for (min r_th, min p_tot) -- no pymoo needed.
    Sweep the p_tot cap across the achievable range and solve min r_th at each.
    Returns arrays (caps, r_th_at_cap, p_tot_at_cap) for feasible solves only.
    """
    p_lo = float(assets.y_train[:, 1].min())
    p_hi = float(assets.y_train[:, 1].max())
    caps = np.linspace(p_lo, p_hi, n)
    R, P = [], []
    for c in caps:
        sol = optimize_min_rth(assets, ptot_max=c, n_starts=8, seed=seed)
        if sol is not None:
            R.append(sol["r_th"]); P.append(sol["p_tot"])
        else:
            R.append(np.nan); P.append(np.nan)
    return caps, np.array(R), np.array(P)


def optimize_robust(assets: Assets, ptot_max: float = PTOT_MAX_DEFAULT,
                    kappa: float = 2.0, conf_z: float = 2.0,
                    n_starts: int = 12, seed: int = 0):
    """
    Risk-aware optimum.

    Objective (pessimistic / UCB for a minimisation):
        minimise   r_th_mean + kappa * r_th_sigma
    Chance constraint on p_tot (conservative feasibility):
        expm1( mu_log(p_tot) + conf_z * sigma_log(p_tot) ) <= ptot_max
        i.e. the upper conf_z-sigma bound of p_tot must satisfy the limit.

    kappa  -> how strongly to penalise objective uncertainty
    conf_z -> standard-normal quantile for the chance constraint
              (1.645 ~ 95%, 2.0 ~ 97.7%, 2.326 ~ 99%)
    """
    rng = np.random.default_rng(seed)
    starts = np.column_stack([
        rng.uniform(LB[0], UB[0], n_starts),
        rng.uniform(LB[1], UB[1], n_starts),
    ])
    bounds = [(LB[0], UB[0]), (LB[1], UB[1])]

    def robust_obj(x):
        u = predict_with_uncertainty(assets, x, k=1.0)
        return float(u["r_th"][0] + kappa * u["r_th_sigma"][0])

    def ptot_upper(x):
        # conf_z-sigma upper bound of p_tot in physical units
        return float(predict_with_uncertainty(assets, x, k=conf_z)["p_tot_hi"][0])

    cons = [{"type": "ineq", "fun": lambda x: ptot_max - ptot_upper(x)}]

    best_x, best_obj = None, np.inf
    for x0 in starts:
        try:
            res = minimize(robust_obj, x0, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            if not res.success:
                continue
            xc = np.clip(res.x, LB, UB)
            if ptot_upper(xc) <= ptot_max + FEAS_TOL and robust_obj(xc) < best_obj:
                best_x, best_obj = xc, robust_obj(xc)
        except Exception:
            continue

    if best_x is None:
        return None
    u = predict_with_uncertainty(assets, best_x, k=conf_z)
    return {
        "vp_vs": float(best_x[0]), "po": float(best_x[1]),
        "r_th": float(u["r_th"][0]), "r_th_sigma": float(u["r_th_sigma"][0]),
        "r_th_ucb": float(u["r_th"][0] + kappa * u["r_th_sigma"][0]),
        "p_tot": float(u["p_tot"][0]), "p_tot_upper": float(u["p_tot_hi"][0]),
        "ptot_max": float(ptot_max), "kappa": kappa, "conf_z": conf_z,
    }


# --------------------------------------------------------------------------- #
# Vectorised grid evaluation (for maps / inverse design)
# --------------------------------------------------------------------------- #
def evaluate_grid(assets: Assets, n_vp: int = 80, n_po: int = 80):
    """Dense grid evaluation over the design space. Returns VP, PO, R_th, P_tot."""
    gx = np.linspace(*BOUNDS["vp_vs"], n_vp)
    gy = np.linspace(*BOUNDS["po"], n_po)
    VP, PO = np.meshgrid(gx, gy)
    pts = np.column_stack([VP.ravel(), PO.ravel()])
    y = predict(assets, pts)
    R = y[:, 0].reshape(VP.shape)
    P = y[:, 1].reshape(VP.shape)
    return VP, PO, R, P


# =========================================================================== #
# Multi-model comparison  (v7.2 "save all models")
# =========================================================================== #
# Reads all_models_manifest.json (written by the pipeline's Section 18.9) and
# loads EVERY saved surrogate for side-by-side prediction. Capability-aware:
#   - scikit-learn models (RF/SVR/Ridge/GPR)  -> joblib, always loadable
#   - XGBoost (.pkl)                          -> needs the `xgboost` package
#   - ANN (.keras)                            -> needs `tensorflow`
# A model whose dependency is absent is reported as unavailable (with a reason)
# instead of crashing the page. An OPTIONAL precomputed grid (all_models_grid.npz)
# is used as a dependency-free fallback so ANN/XGBoost can still be compared
# without installing heavy libraries (see export block in the docs).
# =========================================================================== #
def _sanitize(name):
    return str(name).replace(" ", "_")


class ModelBank:
    """Loads all saved surrogates for comparison and predicts with each."""

    def __init__(self, assets: "Assets"):
        self.scaler_X = assets.scaler_X
        self.scaler_y = assets.scaler_y
        self.log_col = LOG_TRANSFORM_COL
        self.best_name = assets.model_name
        self.available = False
        self.order = []                 # display order of model names
        self.status = {}                # name -> dict(available, reason, loocv, is_ann, source)
        self._models = {}               # name -> loaded live object
        self._grid = None               # optional np.lib.npyio.NpzFile
        self._grid_interp = {}          # name -> (interp_r_th, interp_p_tot)

        amf = _glob1("all_models_manifest.json", "*all*model*manifest*.json")
        if not amf:
            return                      # page will show a "not available" note
        with open(amf) as fh:
            mf = json.load(fh)
        self.available = True
        prep = mf.get("preprocessing", {})
        self.log_col = int(prep.get("log_transform_col", LOG_TRANSFORM_COL))
        self.best_name = mf.get("best_model_loocv", self.best_name)
        self.n = mf.get("n_training_samples")
        models = mf.get("models", {})
        self.order = list(models.keys())

        # optional precomputed grid (dependency-free fallback)
        gpath = _glob1("all_models_grid.npz", "*model*grid*.npz")
        if gpath:
            try:
                self._grid = np.load(gpath, allow_pickle=False)
                self._build_grid_interp()
            except Exception:
                self._grid = None

        for name, info in models.items():
            entry = {"available": False, "reason": "", "source": None,
                     "is_ann": bool(info.get("is_ann", False)),
                     "loocv": info.get("loocv", {})}
            path = os.path.join(ARTIFACT_DIR, os.path.basename(info.get("file", "")))
            if not info.get("file") or not os.path.exists(path):
                entry["reason"] = "model file not found in artifacts/"
            elif entry["is_ann"] or path.lower().endswith((".keras", ".h5")):
                try:
                    from tensorflow import keras       # noqa
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self._models[name] = keras.models.load_model(path, compile=False)
                    entry.update(available=True, source="live (Keras)")
                except Exception:
                    entry["reason"] = "requires tensorflow"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self._models[name] = joblib.load(path)
                    entry.update(available=True, source="live")
                except ModuleNotFoundError as e:
                    entry["reason"] = f"requires the '{e.name}' package"
                except Exception as e:
                    entry["reason"] = f"load error ({type(e).__name__})"

            # fall back to the precomputed grid if the live load failed
            if not entry["available"] and _sanitize(name) in self._grid_interp:
                entry.update(available=True, source="precomputed grid")
            self.status[name] = entry

    # -- precomputed-grid handling ----------------------------------------- #
    def _build_grid_interp(self):
        from scipy.interpolate import RegularGridInterpolator
        g = self._grid
        if "vp_grid" not in g or "po_grid" not in g:
            return
        vp, po = g["vp_grid"], g["po_grid"]
        for key in g.files:
            if not key.startswith("pred__"):
                continue
            name_key = key[len("pred__"):]
            arr = g[key]                              # (n_po, n_vp, 2), original units
            if arr.ndim != 3 or arr.shape[2] != 2:
                continue
            ir = RegularGridInterpolator((po, vp), arr[:, :, 0], bounds_error=False, fill_value=None)
            ip = RegularGridInterpolator((po, vp), arr[:, :, 1], bounds_error=False, fill_value=None)
            self._grid_interp[name_key] = (ir, ip)

    def _predict_grid(self, name, x_raw):
        ir, ip = self._grid_interp[_sanitize(name)]
        pt = np.array([[x_raw[1], x_raw[0]]])         # (po, vp) order
        return float(ir(pt)[0]), float(ip(pt)[0])

    # -- prediction -------------------------------------------------------- #
    def predict_all(self, x_raw):
        """Return {model_name: (r_th, p_tot)} for every AVAILABLE model."""
        x = np.atleast_2d(np.asarray(x_raw, dtype=float))
        Xs = self.scaler_X.transform(x)
        out = {}
        for name in self.order:
            entry = self.status.get(name, {})
            if not entry.get("available"):
                continue
            if entry["source"] == "precomputed grid":
                out[name] = self._predict_grid(name, x[0])
                continue
            m = self._models[name]
            ys = m.predict(Xs, verbose=0) if entry["is_ann"] else m.predict(Xs)
            y = self.scaler_y.inverse_transform(np.atleast_2d(ys))[0].astype(float).copy()
            y[self.log_col] = np.expm1(y[self.log_col])
            out[name] = (float(y[0]), float(y[1]))
        return out

    def gpr_sigma(self, x_raw):
        """GPR-only predictive +/-2 sigma on p_tot (original units), or None."""
        m = self._models.get("GPR")
        if m is None or not hasattr(m, "predict"):
            return None
        try:
            x = np.atleast_2d(np.asarray(x_raw, dtype=float))
            mean, std = m.predict(self.scaler_X.transform(x), return_std=True)
            std = np.atleast_2d(std)
            sig_log = std[0, self.log_col] * self.scaler_y.scale_[self.log_col]
            mu_log = self.scaler_y.inverse_transform(np.atleast_2d(mean))[0, self.log_col]
            return (float(np.expm1(mu_log - 2 * sig_log)), float(np.expm1(mu_log + 2 * sig_log)))
        except Exception:
            return None
