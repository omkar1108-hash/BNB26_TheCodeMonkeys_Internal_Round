"""
Train the fusion model, fit temperature scaling and the conformal threshold, and save artifacts.

    python -m evaluation.train            # generate data if needed, train, calibrate, save
    python -m evaluation.train --regen    # regenerate the synthetic datasets first

Three independent datasets (different seeds) are used so that nothing is evaluated on data
it was trained or calibrated on:  train -> fit XGBoost,  calib -> temperature + conformal,
test -> reported metrics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from backend.config import TARGET_CLASSES, DATASETS_DIR
from backend.fusion.calibration import fit_temperature, conformal_qhat, TemperatureScaler, CALIBRATION_PATH
from backend.fusion.classifier import MODEL_PATH
from evaluation.features import ensure_datasets
from evaluation.metrics import compute_ece

XGB_PARAMS = dict(
    n_estimators=150, max_depth=3, min_child_weight=2, learning_rate=0.1, subsample=0.85, colsample_bytree=0.85,
    objective="multi:softprob", eval_metric="mlogloss", random_state=0, n_jobs=1,
)
ALPHA = 0.10


def fit_xgb(X, y, columns=None):
    Xs = X if columns is None else X[:, columns]
    classes = np.unique(y)                      # a held-out fold may be missing a class
    remap = {int(c): i for i, c in enumerate(classes)}
    y_enc = np.array([remap[int(v)] for v in y])
    model = xgb.XGBClassifier(num_class=len(classes), **XGB_PARAMS)
    model.fit(Xs, y_enc, sample_weight=compute_sample_weight("balanced", y_enc))
    model.trl_classes = classes
    return model


def proba_matrix(model, X, columns=None):
    Xs = X if columns is None else X[:, columns]
    raw = model.predict_proba(Xs)
    out = np.zeros((len(Xs), len(TARGET_CLASSES)))
    for col, idx in enumerate(getattr(model, "trl_classes", model.classes_)):
        out[:, int(idx)] = raw[:, col]
    return out


def calibrate(model, calib_X, calib_y, columns=None):
    p = proba_matrix(model, calib_X, columns)
    T = fit_temperature(p, calib_y)
    pc = TemperatureScaler(T).calibrate_matrix(p)
    return T, conformal_qhat(pc, calib_y, ALPHA)


def main(regen: bool = False, apply_artifacts: bool = False):
    tables = ensure_datasets(DATASETS_DIR, regenerate=regen)
    tr, ca, te = tables["train"], tables["calib"], tables["test"]
    print(f"train={len(tr.y)}  calib={len(ca.y)}  test={len(te.y)} bundles")

    model = fit_xgb(tr.X, tr.y)
    T, qhat = calibrate(model, ca.X, ca.y)

    p_raw = proba_matrix(model, te.X)
    p_cal = TemperatureScaler(T).calibrate_matrix(p_raw)
    pred = p_cal.argmax(1)
    ece_raw = compute_ece(p_raw.max(1), (p_raw.argmax(1) == te.y).astype(float))
    ece_cal = compute_ece(p_cal.max(1), (pred == te.y).astype(float))
    cover = float(np.mean([1.0 - p_cal[i, te.y[i]] <= qhat for i in range(len(te.y))]))

    report = {
        "n_train": int(len(tr.y)), "n_calib": int(len(ca.y)), "n_test": int(len(te.y)),
        "test_accuracy": round(float(accuracy_score(te.y, pred)), 4),
        "test_macro_f1": round(float(f1_score(te.y, pred, average="macro")), 4),
        "temperature": round(T, 4), "conformal_qhat": round(qhat, 4), "alpha": ALPHA,
        "conformal_empirical_coverage_on_test": round(cover, 4),
        "ece_before_temperature": round(ece_raw, 4), "ece_after_temperature": round(ece_cal, 4),
        "note": "Synthetic benchmark; test bundles are fresh draws with the SAME techniques as training. "
                "See evaluation.run_eval for unseen-technique (LOTO) results.",
    }

    cand_model = MODEL_PATH.parent / "xgb_bundle_candidate.json"
    cand_calib = CALIBRATION_PATH.parent / "calibration_candidate.json"
    cand_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(cand_model))
    calib_dict = {
        "temperature": T, "conformal_qhat": qhat, "alpha": ALPHA, "fitted": True,
        "fitted_on": "synthetic_calib (seed 99)",
    }
    cand_calib.write_text(json.dumps(calib_dict, indent=2))

    if apply_artifacts:
        model.save_model(str(MODEL_PATH))
        CALIBRATION_PATH.write_text(json.dumps(calib_dict, indent=2))
        print(f"Applied to {MODEL_PATH.name} and {CALIBRATION_PATH.name}")
    else:
        print(f"Saved {cand_model.name} and {cand_calib.name} (original artifacts preserved pending approval)")

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--apply", action="store_true", help="Apply candidate artifacts to production xgb_bundle.json")
    args = ap.parse_args()
    main(regen=args.regen, apply_artifacts=args.apply)
