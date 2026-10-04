"""Single place where features become a verdict (classifier -> calibration -> abstention)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from backend.fusion.classifier import classifier
from backend.fusion.calibration import TemperatureScaler, load_calibration
from backend.fusion.abstention import evaluate_abstention, conformal_prediction_set
from backend.fusion.feature_builder import FEATURE_INDEX


@dataclass
class Decision:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    abstained: bool
    abstain_reason: Optional[str]
    conformal_set: List[str] = field(default_factory=list)
    model_info: Dict[str, object] = field(default_factory=dict)


TRAINING_CLUSTERS = [
    # [image, audio, text, metadata, conflicts]
    {"image": 0.10, "audio": 0.15, "text": 0.15, "metadata": 0.10, "conflicts": 0.0},   # authentic
    {"image": 0.75, "audio": 0.15, "text": 0.15, "metadata": 0.10, "conflicts": 0.0},   # image manip
    {"image": 0.10, "audio": 0.70, "text": 0.15, "metadata": 0.10, "conflicts": 0.0},   # audio manip
    {"image": 0.10, "audio": 0.15, "text": 0.70, "metadata": 0.10, "conflicts": 0.0},   # text manip
    {"image": 0.10, "audio": 0.15, "text": 0.15, "metadata": 0.65, "conflicts": 0.0},   # metadata manip
    {"image": 0.10, "audio": 0.15, "text": 0.15, "metadata": 0.10, "conflicts": 2.0},   # coordinated
    {"image": 0.70, "audio": 0.70, "text": 0.15, "metadata": 0.10, "conflicts": 0.0},   # multi-manip
    {"image": 0.70, "audio": 0.15, "text": 0.15, "metadata": 0.65, "conflicts": 0.0},   # image + meta manip
]
OOD_DISTANCE_THRESHOLD = 0.58


def ood_guard(fv: np.ndarray) -> Optional[str]:
    """
    Out-of-distribution detector: if a bundle's detector profile is far from ALL
    known training clusters across available modalities and conflicts, abstain
    (insufficient_evidence) rather than guessing.
    """
    g = lambda n: float(fv[FEATURE_INDEX[n]])
    present = [m for m in ("image", "audio", "text", "metadata") if g("has_" + m) > 0]
    if len(present) < 2:
        return None  # handled by standard insufficient evidence check
    
    total_conflicts = min(g("total_conflicts"), 2.0)
    dists = []
    for cl in TRAINING_CLUSTERS:
        diffs = [(g(m + "_score") - cl[m]) ** 2 for m in present]
        diffs.append(((total_conflicts - cl["conflicts"]) / 2.0) ** 2)
        dists.append(float(np.sqrt(np.mean(diffs))))
    
    min_dist = min(dists)
    if min_dist > OOD_DISTANCE_THRESHOLD:
        return (f"Out-of-distribution detector profile (distance {min_dist:.2f} > {OOD_DISTANCE_THRESHOLD}): "
                "signature does not match any known authentic or manipulation pattern.")
    return None


def reliability_guard(fv: np.ndarray, label: str) -> Optional[str]:
    """
    Evidence-bar policy (applied after the model): never accuse on a single weak or fragile signal.
    If the ONLY evidence for 'manipulated' is a pixel-forensics anomaly in an image that is either
    only moderately anomalous or heavily re-compressed, and nothing else corroborates it, abstain
    and ask for an original file instead. Returns an abstention reason, or None.
    """
    if label != "manipulated":
        return None
    g = lambda n: float(fv[FEATURE_INDEX[n]])
    others_clean = all(
        g(s) * g(h) <= 0.45
        for s, h in (("audio_score", "has_audio"), ("text_score", "has_text"), ("metadata_score", "has_metadata"))
    )
    img = g("image_score") * g("has_image")
    if others_clean and g("total_conflicts") == 0 and img > 0.0:
        if g("image_reliability") < 0.55:
            return ("The only anomaly is in a heavily re-compressed image, where pixel-level forensics are unreliable; "
                    "an original (uncompressed) file is needed to decide.")
        if img < 0.60:
            return ("The only anomaly is a moderate pixel-forensics signal in a single image with no corroborating evidence; "
                    "that is not enough to accuse. An original file or a second artifact would settle it.")
    return None


def decide(feature_vector: np.ndarray) -> Decision:
    cal = load_calibration()
    scaler = TemperatureScaler(float(cal.get("temperature", 1.2)))
    raw = classifier.predict_proba(feature_vector)
    probs = scaler.calibrate_probabilities(raw)
    qhat = cal.get("conformal_qhat")
    cset = conformal_prediction_set(probs, qhat)
    # Corroborating sources: every present modality, plus extra independent text sources.
    total_present = float(feature_vector[FEATURE_INDEX["total_modalities_present"]])
    n_text = float(feature_vector[FEATURE_INDEX["n_text_sources"]])
    total_present = total_present + max(0.0, n_text - 1.0)
    abstained, reason, label, conf = evaluate_abstention(
        probs, total_present, conformal_set=cset
    )
    if not abstained:
        guard = reliability_guard(feature_vector, label)
        if guard:
            abstained, reason, label = True, guard, "insufficient_evidence"
        else:
            ood_reason = ood_guard(feature_vector)
            if ood_reason:
                abstained, reason, label = True, ood_reason, "insufficient_evidence"
    info = dict(classifier.info())
    info.update({
        "temperature": float(cal.get("temperature", 1.2)),
        "conformal_alpha": cal.get("alpha"),
        "calibration_fitted": bool(cal.get("fitted", False)),
    })
    return Decision(label, conf, probs, abstained, reason, cset, info)
