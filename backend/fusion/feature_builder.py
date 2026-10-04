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
    "total_modalities_present",
    # --- appended features (indices 14+) ---
    "n_text_sources",
    "speaker_conflict",
    "date_conflicts",
    "location_conflicts",
    "has_clip_similarity",
    # modality-agnostic aggregates: let the model generalise to detectors/techniques it never saw fire
    "max_modality_anomaly",
    "n_anomalous_modalities",
    "total_conflicts",
    "image_reliability",
]

# Which feature columns each modality owns. Used for ablations and counterfactuals.
MODALITY_COLUMNS = {
    "image": ["image_score", "has_image"],
    "audio": ["audio_score", "has_audio"],
    "text": ["text_score", "has_text"],
    "metadata": ["metadata_score", "has_metadata"],
}
CROSS_MODAL_COLUMNS = [
    "image_text_similarity", "speaker_similarity", "metadata_conflict_count",
    "contradiction_count", "high_severity_contradictions", "n_text_sources",
    "speaker_conflict", "date_conflicts", "location_conflicts", "has_clip_similarity",
    "total_conflicts",
]
FEATURE_INDEX = {n: i for i, n in enumerate(FEATURE_NAMES)}


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
    n_text_sources = 0.0
    speaker_conflict = 0.0
    date_conflicts = 0.0
    location_conflicts = 0.0
    has_clip = 0.0

    if cross_modal:
        if cross_modal.image_text_similarity is not None:
            # Intentionally NOT used by the model: it was trained on synthetic bundles that carry no real
            # CLIP signal, so feeding one in would be an out-of-distribution input. The CLIP score is
            # surfaced as evidence and as an authenticity guard instead (see verdict engine).
            pass
        if cross_modal.speaker_similarity is not None:
            spk_sim = float(cross_modal.speaker_similarity)
        meta_conflicts = float(len(cross_modal.metadata_conflicts))
        contradictions = float(len(cross_modal.contradictions))
        high_contradictions = float(
            sum(1 for c in cross_modal.contradictions if c.severity == "high")
        )
        n_text_sources = float(cross_modal.n_text_sources)
        speaker_conflict = 1.0 if any(c.field == "speaker" for c in cross_modal.contradictions) else 0.0
        date_conflicts = float(
            sum(1 for c in cross_modal.contradictions if c.field == "date")
            + sum(1 for m in cross_modal.metadata_conflicts if m.startswith("Temporal conflict"))
        )
        location_conflicts = float(
            sum(1 for c in cross_modal.contradictions if c.field == "location")
            + sum(1 for m in cross_modal.metadata_conflicts if m.startswith("Location conflict"))
        )

    total_present = has_image + has_audio + has_text + has_metadata

    scores_present = [image_score * has_image, audio_score * has_audio,
                      text_score * has_text, metadata_score * has_metadata]
    max_anomaly = max(scores_present)
    n_anomalous = float(sum(1 for sc in scores_present if sc > 0.45))
    total_conflicts = contradictions + meta_conflicts
    # JPEG quality -> how much to trust pixel forensics (1 = pristine, 0 = heavily laundered)
    image_reliability = 1.0
    if img_ev and img_ev.present:
        q = img_ev.features.get("jpeg_quality") if img_ev.features else None
        image_reliability = 0.8 if q is None else float(np.clip((q - 50.0) / 40.0, 0.0, 1.0))

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
        total_present,
        n_text_sources,
        speaker_conflict,
        date_conflicts,
        location_conflicts,
        has_clip,
        max_anomaly,
        n_anomalous,
        total_conflicts,
        image_reliability,
    ], dtype=np.float32)

    return vector

