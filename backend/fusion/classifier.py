"""Layer 3 fusion classifier.

Primary path : a trained XGBoost 4-class model loaded from
               ``backend/fusion/artifacts/xgb_bundle.json`` (produced by
               ``python -m evaluation.train``).
Fallback path: a transparent rule-based scorer, used only when no trained model
               file is present. When it is used the verdict is flagged
               ``is_fallback`` so evaluation code can exclude it.
"""
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from backend.config import TARGET_CLASSES
from backend.fusion.feature_builder import FEATURE_NAMES, FEATURE_INDEX

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "xgb_bundle.json"


def rule_based_proba(fv: np.ndarray) -> Dict[str, float]:
    """Transparent heuristic scorer (fallback only)."""
    g = lambda name: float(fv[FEATURE_INDEX[name]])
    total_present = g("total_modalities_present")
    if total_present == 0:
        return {"authentic": 0.05, "manipulated": 0.05,
                "coordinated_synthetic": 0.05, "insufficient_evidence": 0.85}

    max_anomaly = max(
        g("image_score") * g("has_image"),
        g("audio_score") * g("has_audio"),
        g("text_score") * g("has_text"),
        g("metadata_score") * g("has_metadata"),
    )
    cross = (
        g("contradiction_count") * 1.5
        + g("high_severity_contradictions") * 2.0
        + g("metadata_conflict_count") * 1.0
    )
    if g("has_clip_similarity"):
        cross += (1.0 - g("image_text_similarity")) * 1.5

    logit_auth = 2.0 * (1.0 - max_anomaly) - 3.0 * cross
    if total_present < 2:
        logit_auth -= 2.5
    logit_manip = 3.5 * max_anomaly - 1.0 * cross
    logit_coord = 2.8 * cross + 1.2 * (1.0 - max_anomaly) - 1.0
    logit_insuff = 2.0 * (4.0 - total_present) / 4.0 - 0.5 * (max_anomaly + cross)

    logits = np.array([logit_auth, logit_manip, logit_coord, logit_insuff], dtype=np.float64)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return {TARGET_CLASSES[i]: float(probs[i]) for i in range(len(TARGET_CLASSES))}


class BundleClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.classes = TARGET_CLASSES
        self.model = None
        self.model_path = model_path
        self.load_error: Optional[str] = None
        self._load()

    def _load(self):
        if not self.model_path.exists():
            self.load_error = f"No trained model at {self.model_path.name}; run `python -m evaluation.train`."
            return
        try:
            import xgboost as xgb
            clf = xgb.XGBClassifier()
            clf.load_model(str(self.model_path))
            self.model = clf
        except Exception as e:  # pragma: no cover
            self.load_error = f"Could not load XGBoost model: {e}"

    @property
    def using_fallback(self) -> bool:
        return self.model is None

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """Vectorised probabilities, columns ordered as TARGET_CLASSES."""
        X = np.atleast_2d(X)
        if self.model is None:
            return np.array([[rule_based_proba(r)[c] for c in self.classes] for r in X])
        X_in = X
        if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ < X.shape[1]:
            X_in = X[:, :self.model.n_features_in_]
        raw = self.model.predict_proba(X_in)
        out = np.zeros((len(X), len(self.classes)))
        for col, cls_idx in enumerate(self.model.classes_):
            out[:, int(cls_idx)] = raw[:, col]
        return out

    def predict_proba(self, feature_vector: np.ndarray) -> Dict[str, float]:
        row = self.predict_proba_matrix(feature_vector)[0]
        return {c: float(row[i]) for i, c in enumerate(self.classes)}

    def info(self) -> Dict[str, object]:
        return {
            "type": "rule_based_fallback" if self.model is None else "xgboost",
            "n_features": len(FEATURE_NAMES),
            "load_error": self.load_error,
        }


classifier = BundleClassifier()
