import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from backend.config import TARGET_CLASSES, DATASETS_DIR
from backend.models.bundle import BundleInput
from backend.verdict.engine import analyze_bundle


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE).
    Measures the absolute difference between predicted confidence and empirical accuracy.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_coverage_vs_accuracy(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    thresholds: List[float]
) -> List[Dict[str, float]]:
    """
    Computes coverage and selective accuracy under varying abstention thresholds.
    """
    curve = []
    n = len(confidences)
    if n == 0:
        return curve

    for tau in thresholds:
        accepted = confidences >= tau
        coverage = float(np.mean(accepted))
        if np.sum(accepted) > 0:
            acc = float(np.mean(accuracies[accepted]))
        else:
            acc = 1.0  # complete abstention
        curve.append({
            "threshold": tau,
            "coverage": round(coverage, 3),
            "selective_accuracy": round(acc, 3)
        })
    return curve


def evaluate_manifest(
    manifest_path: Path,
    exclude_fallbacks: bool = True
) -> Dict[str, Any]:
    """
    Evaluates model predictions on a dataset manifest.
    Strictly excludes records flagged with is_fallback == True to maintain benchmark integrity.
    """
    with open(manifest_path, "r") as f:
        records = json.load(f)

    y_true = []
    y_pred = []
    confidences = []
    probs_matrix = []
    valid_records_count = 0
    fallback_records_excluded = 0

    for rec in records:
        meta_dict = None
        if rec.get("metadata_path") and Path(rec["metadata_path"]).exists():
            try:
                with open(rec["metadata_path"]) as mf:
                    meta_dict = json.load(mf)
            except Exception:
                pass

        bundle = BundleInput(
            bundle_id=rec["bundle_id"],
            image_path=rec.get("image_path"),
            audio_path=rec.get("audio_path"),
            text=rec.get("text"),
            metadata=meta_dict,
            claimed_speaker=rec.get("claimed_speaker")
        )

        res = analyze_bundle(bundle)

        # Enforce evaluation policy: Exclude fallbacks from metrics
        if exclude_fallbacks and res.is_fallback:
            fallback_records_excluded += 1
            continue

        valid_records_count += 1
        y_true.append(rec["label"])
        y_pred.append(res.label)
        confidences.append(res.confidence)
        
        # Collect class probabilities
        probs = [res.probabilities.get(cls, 0.0) for cls in TARGET_CLASSES]
        probs_matrix.append(probs)

    if valid_records_count == 0:
        return {
            "status": "No non-fallback records evaluated",
            "fallback_records_excluded": fallback_records_excluded
        }

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    conf_np = np.array(confidences)
    accuracies = (y_true_np == y_pred_np).astype(np.float64)

    # Metrics
    macro_f1 = float(f1_score(y_true_np, y_pred_np, labels=TARGET_CLASSES, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_np, y_pred_np, labels=TARGET_CLASSES).tolist()
    ece = compute_ece(conf_np, accuracies)
    
    # Multiclass AUROC
    try:
        y_true_bin = label_binarize(y_true_np, classes=TARGET_CLASSES)
        auroc = float(roc_auc_score(y_true_bin, np.array(probs_matrix), multi_class="ovr", average="macro"))
    except Exception:
        auroc = 0.50

    # Coverage vs Accuracy curve
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cov_acc = compute_coverage_vs_accuracy(conf_np, accuracies, thresholds)

    return {
        "evaluated_bundles": valid_records_count,
        "fallback_records_excluded": fallback_records_excluded,
        "overall_accuracy": round(float(np.mean(accuracies)), 4),
        "macro_f1": round(macro_f1, 4),
        "auroc_ovr": round(auroc, 4),
        "expected_calibration_error": round(ece, 4),
        "confusion_matrix": {
            "labels": TARGET_CLASSES,
            "matrix": cm
        },
        "coverage_vs_accuracy": cov_acc
    }


if __name__ == "__main__":
    manifest_file = DATASETS_DIR / "synthetic_bundles" / "manifest.json"
    if manifest_file.exists():
        # For testing, evaluate on all records (including fallbacks) to inspect initial baseline
        results = evaluate_manifest(manifest_file, exclude_fallbacks=False)
        print("Evaluation Results:", json.dumps(results, indent=2))
