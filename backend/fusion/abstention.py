from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from backend.config import CONFIDENCE_THRESHOLD, ENTROPY_ABSTAIN_THRESHOLD


def compute_entropy(probabilities: Dict[str, float]) -> float:
    """
    Shannon entropy in bits. Higher entropy signifies uniform/uncertain predictions.
    """
    probs = [p for p in probabilities.values() if p > 1e-9]
    return float(-sum(p * math.log2(p) for p in probs))


def conformal_prediction_set(
    probabilities: Dict[str, float],
    qhat: Optional[float]
) -> List[str]:
    """Classes whose nonconformity score (1 - p) is within the conformal threshold."""
    if qhat is None:
        return []
    return [c for c, p in sorted(probabilities.items(), key=lambda x: -x[1]) if (1.0 - p) <= qhat]


def evaluate_abstention(
    probabilities: Dict[str, float],
    total_modalities: float,
    confidence_cutoff: float = CONFIDENCE_THRESHOLD,
    entropy_cutoff: float = ENTROPY_ABSTAIN_THRESHOLD,
    conformal_set: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], str, float]:
    """
    Evaluates conformal/threshold abstention.
    Returns: (abstained, abstain_reason, final_label, final_confidence)
    """
    if total_modalities == 0:
        return True, "No bundle modalities provided.", "insufficient_evidence", 0.0
        
    entropy = compute_entropy(probabilities)
    sorted_classes = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_label, top_prob = sorted_classes[0]
    
    # Check 1: Missing cross-validation evidence for authentic verdict
    if top_label == "authentic" and total_modalities < 2:
        return (
            True,
            "Authenticity requires at least 2 distinct corroborating modalities.",
            "insufficient_evidence",
            top_prob
        )
        
    # Check 1b: the model itself judged the evidence insufficient
    if top_label == "insufficient_evidence":
        return (
            True,
            "The fusion model judged the available evidence insufficient for a verdict.",
            "insufficient_evidence",
            top_prob
        )

    # Check 1c: conformal prediction set is ambiguous (more than one plausible class)
    if conformal_set and len(conformal_set) > 1:
        names = ", ".join(c.replace("_", " ") for c in conformal_set)
        return (
            True,
            f"Conformal prediction set is ambiguous at the calibrated 90% level: {{{names}}}.",
            "insufficient_evidence",
            top_prob
        )

    # Check 2: High Entropy / Ambiguity
    if entropy > entropy_cutoff:
        return (
            True,
            f"High prediction entropy ({entropy:.2f} bits exceeds safety threshold {entropy_cutoff:.2f}).",
            "insufficient_evidence",
            top_prob
        )
        
    # Check 3: Below Confidence Threshold
    if top_prob < confidence_cutoff:
        return (
            True,
            f"Top class probability ({top_prob:.2%}) is below confidence cutoff ({confidence_cutoff:.2%}).",
            "insufficient_evidence",
            top_prob
        )
        
    return False, None, top_label, top_prob

