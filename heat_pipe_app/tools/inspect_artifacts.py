#!/usr/bin/env python3
"""
inspect_artifacts.py - run this LOCALLY to check that a newly trained surrogate
is compatible with the web app, WITHOUT uploading anything anywhere.

Usage:
    python tools/inspect_artifacts.py            # inspects ../artifacts
    python tools/inspect_artifacts.py /path/to/artifacts

It reports the discovered files, model type, the scikit-learn version the model
was pickled with (so you can pin requirements.txt correctly), the input/output
shapes, whether predictive std is available, the derived design-space bounds and
sample count, and a final COMPATIBLE / NOT COMPATIBLE verdict. Nothing leaves
your machine.
"""
import os
import sys
import json
import glob
import warnings

import numpy as np
import joblib


def glob1(d, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(d, pat)))
        if hits:
            return hits[0]
    return None


def main(artifact_dir):
    print("=" * 70)
    print("ARTIFACT INSPECTION  -  ", os.path.abspath(artifact_dir))
    print("=" * 70)

    manifest_path = glob1(artifact_dir, "*manifest*.json", "*.json")
    model_path = None
    manifest = {}
    if manifest_path:
        try:
            manifest = json.load(open(manifest_path))
            if manifest.get("file"):
                cand = os.path.join(artifact_dir, os.path.basename(manifest["file"]))
                model_path = cand if os.path.exists(cand) else None
        except Exception as e:
            print("  ! manifest unreadable:", e)
    if model_path is None:
        model_path = glob1(artifact_dir, "best_surrogate*.pkl", "*surrogate*.pkl", "model*.pkl")
    sx_path = glob1(artifact_dir, "scaler_X*.pkl", "scaler_x*.pkl", "*scalerX*.pkl", "x_scaler*.pkl")
    sy_path = glob1(artifact_dir, "scaler_y*.pkl", "*scalerY*.pkl", "y_scaler*.pkl")

    print("\n[1] Discovered files")
    for name, p in [("manifest", manifest_path), ("model", model_path),
                    ("scaler_X", sx_path), ("scaler_y", sy_path)]:
        print(f"    {name:9s}: {os.path.basename(p) if p else '*** MISSING ***'}")
    if not all([manifest_path, model_path, sx_path, sy_path]):
        print("\nVERDICT: NOT COMPATIBLE - one or more required files are missing.")
        return 1

    print("\n[2] Manifest")
    for k, v in manifest.items():
        print(f"    {k}: {v}")

    pickled_ver = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = joblib.load(model_path)
        sX = joblib.load(sx_path)
        sY = joblib.load(sy_path)
        for wi in w:
            msg = str(wi.message)
            if "from version" in msg and "when using version" in msg:
                try:
                    pickled_ver = msg.split("from version")[1].split("when using")[0].strip()
                except Exception:
                    pass
    pickled_ver = pickled_ver or getattr(model, "_sklearn_version", "unknown")
    import sklearn
    print("\n[3] Versions")
    print(f"    model pickled with scikit-learn : {pickled_ver}")
    print(f"    scikit-learn in THIS environment: {sklearn.__version__}")
    print(f"    -> pin this line in requirements.txt:  scikit-learn=={pickled_ver}")

    print("\n[4] Model")
    print(f"    type            : {type(model).__name__}")
    has_train = hasattr(model, "X_train_") and hasattr(model, "y_train_")
    print(f"    has X_train_/y_train_ (GPR): {has_train}")
    supports_std = False
    if has_train:
        try:
            Xr = sX.inverse_transform(model.X_train_)
            model.predict(sX.transform(Xr[:1]), return_std=True)
            supports_std = True
        except TypeError:
            supports_std = False
        except Exception as e:
            print("    ! predict(return_std=True) raised:", e)
    print(f"    predict(return_std=True) works: {supports_std}")

    if not has_train:
        print("\nVERDICT: NOT COMPATIBLE - the model must be a fitted GaussianProcessRegressor "
              "exposing X_train_/y_train_. Re-export the GPR surrogate.")
        return 1

    Xr = sX.inverse_transform(model.X_train_)
    yr = np.asarray(model.y_train_)
    n_in, n_out = Xr.shape[1], yr.shape[1]
    print("\n[5] Schema & data")
    print(f"    inputs : {n_in}   outputs: {n_out}   samples (n): {Xr.shape[0]}")
    if n_in != 2 or n_out != 2:
        print("\nVERDICT: NOT COMPATIBLE - the app supports exactly 2 inputs and 2 outputs.")
        return 1

    # derived bounds (what the app will use)
    print("\n[6] Derived design-space bounds (auto-used by the app)")
    print(f"    vp_vs : [{Xr[:,0].min():.4f}, {Xr[:,0].max():.4f}]")
    print(f"    po    : [{Xr[:,1].min():.4f}, {Xr[:,1].max():.4f}]")
    # original-unit outputs (assumes col 1 = log1p p_tot, per the convention)
    yo = sY.inverse_transform(yr).copy()
    yo[:, 1] = np.expm1(yo[:, 1])
    print("\n[7] Output ranges (col0 = r_th linear, col1 = p_tot via expm1)")
    print(f"    r_th  : [{yo[:,0].min():.4f}, {yo[:,0].max():.4f}]")
    print(f"    p_tot : [{yo[:,1].min():.1f}, {yo[:,1].max():.1f}]")
    print(f"    (default pressure-drop constraint is 4200 Pa - confirm it sits in this range)")

    ok = supports_std and has_train and n_in == 2 and n_out == 2
    print("\n" + "=" * 70)
    print("VERDICT:", "COMPATIBLE - drop these into artifacts/ and the app self-configures."
          if ok else "NOT COMPATIBLE - see notes above.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    sys.exit(main(d))
