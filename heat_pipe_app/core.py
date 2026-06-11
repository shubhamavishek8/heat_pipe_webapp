"""
core.py - Single source of truth for the Heat-Pipe Surrogate web app.

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

# Design-space bounds (user-supplied; equal to dataset min/max)
BOUNDS = {
    "vp_vs": (0.05, 0.95),
    "po":    (0.40, 0.77),
}
LB = np.array([BOUNDS["vp_vs"][0], BOUNDS["po"][0]])
UB = np.array([BOUNDS["vp_vs"][1], BOUNDS["po"][1]])

PTOT_MAX_DEFAULT = 4200.0      # default constraint limit (Pa)
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
# Asset loading
# --------------------------------------------------------------------------- #
class Assets:
    """Container for the loaded surrogate, scalers, manifest and recovered data."""

    def __init__(self):
        with warnings.catch_warnings():
            # InconsistentVersionWarning is expected if the runtime sklearn != 1.7.2.
            warnings.simplefilter("ignore")
            self.model    = joblib.load(os.path.join(ARTIFACT_DIR, "best_surrogate_GPR.pkl"))
            self.scaler_X = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_X2.pkl"))
            self.scaler_y = joblib.load(os.path.join(ARTIFACT_DIR, "scaler_y2.pkl"))
        with open(os.path.join(ARTIFACT_DIR, "best_surrogate_manifest.json")) as fh:
            self.manifest = json.load(fh)

        # Recover the 49 training points from the GPR itself (no CSV needed).
        self.X_train = self.scaler_X.inverse_transform(self.model.X_train_)
        y_log = self.scaler_y.inverse_transform(self.model.y_train_)
        y_orig = y_log.copy()
        y_orig[:, LOG_TRANSFORM_COL] = np.expm1(y_orig[:, LOG_TRANSFORM_COL])
        self.y_train = y_orig

        # Reference scaled-sigma distribution over a dense in-domain grid,
        # used to classify predictive confidence (soft extrapolation signal).
        gx = np.linspace(*BOUNDS["vp_vs"], 40)
        gy = np.linspace(*BOUNDS["po"], 40)
        GX, GY = np.meshgrid(gx, gy)
        grid = np.column_stack([GX.ravel(), GY.ravel()])
        _, sig_scaled = self._raw_predict(grid, return_std=True)
        self._sigma_ref = np.sort(sig_scaled)        # ascending
        # Delaunay triangulation of training inputs for convex-hull membership.
        try:
            self._tri = Delaunay(self.X_train)
        except Exception:
            self._tri = None

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
    """
    X_raw = np.atleast_2d(np.asarray(X_raw, dtype=float))
    y_s, sig_scaled = assets._raw_predict(X_raw, return_std=True)
    sc = assets.scaler_y

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
        in_hull  : inside the convex hull of the 49 training points
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
