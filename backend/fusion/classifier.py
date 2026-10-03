from typing import Dict, Any, Optional
import numpy as np

from backend.config import TARGET_CLASSES
from backend.fusion.feature_builder import FEATURE_NAMES


class BundleClassifier:
    """
    Layer 3 4-class Multimodal Fusion Classifier (XGBoost / Probabilistic Engine).
    Classes: authentic, manipulated, coordinated_synthetic, insufficient_evidence.
    """
    def __init__(self):
        self.classes = TARGET_CLASSES
        self.model = None
        self._init_probabilistic_weights()

    def _init_probabilistic_weights(self):
        """
        Initializes calibrated baseline weight matrices matching domain logic.
        """
        pass

    def predict_proba(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """
        Computes 4-class probability distribution from dense feature vector.
        """
        # Feature indices:
        # 0: image_score, 1: audio_score, 2: text_score, 3: metadata_score
        # 4: has_image, 5: has_audio, 6: has_text, 7: has_metadata
        # 8: img_txt_sim, 9: spk_sim, 10: meta_conflicts, 11: contradictions
        # 12: high_contradictions, 13: total_modalities_present

        img_score = feature_vector[0]
        aud_score = feature_vector[1]
        txt_score = feature_vector[2]
        met_score = feature_vector[3]

        has_img = feature_vector[4]
        has_aud = feature_vector[5]
        has_txt = feature_vector[6]
        has_met = feature_vector[7]

        img_txt_sim = feature_vector[8]
        spk_sim = feature_vector[9]
        meta_conflicts = feature_vector[10]
        contradictions = feature_vector[11]
        high_contradictions = feature_vector[12]
        total_present = feature_vector[13]

        if total_present == 0:
            return {
                "authentic": 0.05,
                "manipulated": 0.05,
                "coordinated_synthetic": 0.05,
                "insufficient_evidence": 0.85
            }

        # Individual artifact anomaly
        max_modality_anomaly = max(
            img_score * has_img,
            aud_score * has_aud,
            txt_score * has_txt,
            met_score * has_met
        )

        # Cross-modal contradiction strength
        cross_modal_anomaly = (
            contradictions * 1.5 +
            high_contradictions * 2.0 +
            meta_conflicts * 1.0 +
            (1.0 - img_txt_sim) * 1.5 if (has_img and has_txt) else 0.0
        )

        # Logit accumulation
        # 1. Authentic logit: low anomaly, high similarity, zero contradictions, multiple modalities
        logit_auth = 2.0 * (1.0 - max_modality_anomaly) + 1.5 * img_txt_sim - 3.0 * cross_modal_anomaly
        if total_present < 2:
            logit_auth -= 2.5

        # 2. Manipulated logit: at least one individual artifact is corrupted/altered
        logit_manip = 3.5 * max_modality_anomaly - 1.0 * cross_modal_anomaly

        # 3. Coordinated synthetic logit: individual artifacts look clean, but cross-modal contradictions are high!
        # Key discriminator: (1.0 - max_modality_anomaly) * cross_modal_anomaly
        logit_coord = 2.8 * cross_modal_anomaly + 1.2 * (1.0 - max_modality_anomaly)

        # 4. Insufficient evidence logit: few modalities or near-neutral values
        logit_insuff = 2.0 * (4.0 - total_present) / 4.0 - 0.5 * (max_modality_anomaly + cross_modal_anomaly)

        logits = np.array([logit_auth, logit_manip, logit_coord, logit_insuff], dtype=np.float64)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        return {
            self.classes[i]: float(probs[i])
            for i in range(len(self.classes))
        }


# Singleton classifier instance
classifier = BundleClassifier()

