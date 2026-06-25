#!/usr/bin/env python3
"""
inspect_artifacts.py - run this LOCALLY to check that a newly trained surrogate
is compatible with the web app, WITHOUT uploading anything anywhere.

Usage:
    python tools/inspect_artifacts.py            # inspects ../artifacts
    python tools/inspect_artifacts.py /path/to/artifacts

ANY scikit-learn regressor is compatible for prediction and optimisation;
uncertainty features simply switch off for non-Gaussian-Process models. Nothing
leaves your machine.
"""
import os
import sys
import csv as _csv
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
    print("=" * 72)
    print("ARTIFACT INSPECTION  -  ", os.path.abspath(artifact_dir))
    print("=" * 72)

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
    csv_path = glob1(artifact_dir, "*data*.csv", "*raw*.csv", "*dataset*.csv", "*.csv")

    print("\n[1] Discovered files")
    for name, p in [("manifest", manifest_path), ("model", model_path),
                    ("scaler_X", sx_path), ("scaler_y", sy_path), ("dataset CSV", csv_path)]:
        print(f"    {name:11s}: {os.path.basename(p) if p else '-- none --'}")
    if not all([manifest_path, model_path, sx_path, sy_path]):
        print("\nVERDICT: NOT COMPATIBLE - a required file (manifest/model/scalers) is missing.")
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
    print(f"    best model (per manifest)        : {manifest.get('best_model_name', type(model).__name__)}")
    print(f"    model class                      : {type(model).__name__}")
    print(f"    model pickled with scikit-learn  : {pickled_ver}")
    print(f"    scikit-learn in THIS environment : {sklearn.__version__}")
    print(f"    -> pin in requirements.txt:  scikit-learn=={pickled_ver}")

    n_in = int(getattr(sX, "n_features_in_", 2))
    try:
        y_probe = np.atleast_2d(model.predict(np.zeros((1, n_in))))
        n_out = y_probe.shape[1]
    except Exception as e:
        print("\n    ! model.predict failed:", e)
        print("\nVERDICT: NOT COMPATIBLE - the model cannot predict on a 2-feature input.")
        return 1

    try:
        model.predict(np.zeros((1, n_in)), return_std=True)
        supports_std = True
    except Exception:
        supports_std = False

    print("\n[4] Model capability")
    print(f"    inputs / outputs                 : {n_in} / {n_out}")
    print(f"    predictive std (Gaussian Process): {supports_std}")
    if n_in != 2 or n_out != 2:
        print("\nVERDICT: NOT COMPATIBLE - the app supports exactly 2 inputs and 2 outputs.")
        return 1

    src, X = "none", None
    if csv_path:
        try:
            rows = list(_csv.reader(open(csv_path, newline="")))
            data = np.array([[float(v) for v in r] for r in rows[1:] if r], dtype=float)
            X = data[:, :2]; src = "CSV"
        except Exception:
            X = None
    if X is None and hasattr(model, "X_train_"):
        X = sX.inverse_transform(model.X_train_); src = "model.X_train_ (Gaussian Process)"

    print("\n[5] Training-data source & derived design space")
    print(f"    source : {src}")
    mb = manifest.get("bounds")
    if X is not None:
        print(f"    samples (n) : {X.shape[0]}")
        print(f"    vp_vs bounds: [{X[:,0].min():.4f}, {X[:,0].max():.4f}]")
        print(f"    po    bounds: [{X[:,1].min():.4f}, {X[:,1].max():.4f}]")
    elif isinstance(mb, dict):
        print(f"    bounds from manifest: {mb}  | n_samples: {manifest.get('n_samples','unknown')}")
    else:
        print("    -- no training data available --")

    can_bounds = (X is not None) or isinstance(mb, dict)
    print("\n" + "=" * 72)
    if not can_bounds:
        print("VERDICT: NOT COMPATIBLE - cannot determine design-space bounds.\n"
              "  Add a dataset CSV to artifacts/, deploy a Gaussian Process (exposes\n"
              "  X_train_), or add a 'bounds' field to the manifest.")
        return 1
    note = ("Full functionality (Gaussian Process: predictive bands, uncertainty surface, "
            "Next-Experiment all active)." if supports_std else
            "Prediction, optimisation and tolerance analysis active; uncertainty features OFF "
            "for this model type.")
    print("VERDICT: COMPATIBLE - drop these into artifacts/ and the app self-configures.")
    print("  " + note)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    sys.exit(main(d))
