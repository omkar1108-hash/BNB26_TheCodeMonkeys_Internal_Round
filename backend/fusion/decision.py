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
    info = dict(classifier.info())
    info.update({
        "temperature": float(cal.get("temperature", 1.2)),
        "conformal_alpha": cal.get("alpha"),
        "calibration_fitted": bool(cal.get("fitted", False)),
    })
    return Decision(label, conf, probs, abstained, reason, cset, info)
