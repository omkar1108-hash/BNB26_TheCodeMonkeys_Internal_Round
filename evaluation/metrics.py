from typing import List, Dict
import numpy as np


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
