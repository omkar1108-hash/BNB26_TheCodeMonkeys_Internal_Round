from typing import Dict, List, Optional
import numpy as np

from backend.models.bundle import ModalityEvidence, CrossModalEvidence


FEATURE_NAMES = [
    "image_score",
    "audio_score",
    "text_score",
    "metadata_score",
    "has_image",
    "has_audio",
    "has_text",
    "has_metadata",
    "image_text_similarity",
    "speaker_similarity",
    "metadata_conflict_count",
    "contradiction_count",
    "high_severity_contradictions",
    "total_modalities_present"
]


def build_fusion_features(
    modality_results: Dict[str, ModalityEvidence],
    cross_modal: Optional[CrossModalEvidence]
) -> np.ndarray:
    """
    Builds a fixed-dimensional dense feature vector for Layer 3 fusion.
    Handles missing modalities gracefully via binary presence indicator flags.
    Missing modalities lower overall model certainty rather than causing failures.
    """
    img_ev = modality_results.get("image")
    aud_ev = modality_results.get("audio")
    txt_ev = modality_results.get("text")
    met_ev = modality_results.get("metadata")

    has_image = 1.0 if (img_ev and img_ev.present) else 0.0
    has_audio = 1.0 if (aud_ev and aud_ev.present) else 0.0
    has_text = 1.0 if (txt_ev and txt_ev.present) else 0.0
    has_metadata = 1.0 if (met_ev and met_ev.present) else 0.0

    image_score = img_ev.score if (img_ev and img_ev.present) else 0.0
    audio_score = aud_ev.score if (aud_ev and aud_ev.present) else 0.0
    text_score = txt_ev.score if (txt_ev and txt_ev.present) else 0.0
    metadata_score = met_ev.score if (met_ev and met_ev.present) else 0.0

    img_txt_sim = 0.5
    spk_sim = 0.5
    meta_conflicts = 0.0
    contradictions = 0.0
    high_contradictions = 0.0

    if cross_modal:
        if cross_modal.image_text_similarity is not None:
            img_txt_sim = float(cross_modal.image_text_similarity)
        if cross_modal.speaker_similarity is not None:
            spk_sim = float(cross_modal.speaker_similarity)
        meta_conflicts = float(len(cross_modal.metadata_conflicts))
        contradictions = float(len(cross_modal.contradictions))
        high_contradictions = float(
            sum(1 for c in cross_modal.contradictions if c.severity == "high")
        )

    total_present = has_image + has_audio + has_text + has_metadata

    vector = np.array([
        image_score,
        audio_score,
        text_score,
        metadata_score,
        has_image,
        has_audio,
        has_text,
        has_metadata,
        img_txt_sim,
        spk_sim,
        meta_conflicts,
        contradictions,
        high_contradictions,
        total_present
    ], dtype=np.float32)

    return vector

