import uuid
from typing import Dict, Any, List, Optional

from backend.models.bundle import (
    BundleInput,
    VerdictResult,
    ModalityEvidence,
    CrossModalEvidence,
    ExtractedClaims
)
from backend.analyzers.image_detector import analyze_image
from backend.analyzers.audio_detector import analyze_audio
from backend.analyzers.text_detector import analyze_text
from backend.analyzers.metadata_detector import analyze_metadata
from backend.cross_modal.image_text import compute_image_caption_similarity
from backend.cross_modal.audio_speaker import verify_claimed_speaker
from backend.cross_modal.metadata_rules import check_metadata_conflicts
from backend.claims.extractor import extract_claims
from backend.claims.conflict import find_bundle_contradictions
from backend.fusion.feature_builder import build_fusion_features
from backend.fusion.classifier import classifier
from backend.fusion.calibration import TemperatureScaler
from backend.fusion.abstention import evaluate_abstention
from backend.verdict.explainer import rank_evidence
from backend.verdict.llm_report import generate_llm_report


def analyze_bundle(bundle: BundleInput) -> VerdictResult:
    """
    Executes the full 4-layer TrustLayer verification pipeline on a multimodal bundle.
    """
    bundle_id = bundle.bundle_id or str(uuid.uuid4())

    # -------------------------------------------------------------
    # Layer 1: Per-Modality Detectors
    # -------------------------------------------------------------
    image_ev = analyze_image(bundle.image_path)
    audio_ev = analyze_audio(bundle.audio_path)
    text_ev = analyze_text(bundle.text)
    metadata_ev = analyze_metadata(bundle.image_path, bundle.metadata)

    modality_results: Dict[str, ModalityEvidence] = {
        "image": image_ev,
        "audio": audio_ev,
        "text": text_ev,
        "metadata": metadata_ev
    }

    # -------------------------------------------------------------
    # Layer 2: Cross-Modal Verification & Claims
    # -------------------------------------------------------------
    img_txt_sim, img_txt_fallback, img_txt_reason = compute_image_caption_similarity(
        bundle.image_path, bundle.text
    )
    spk_sim, spk_fallback, spk_reason = verify_claimed_speaker(
        bundle.audio_path, bundle.claimed_speaker
    )

    claims_map: Dict[str, ExtractedClaims] = {}
    if bundle.text:
        claims_map["text"] = extract_claims(bundle.text)

    contradictions = find_bundle_contradictions(claims_map)

    # Check metadata conflicts
    text_dates = claims_map["text"].dates if "text" in claims_map else []
    text_locations = claims_map["text"].locations if "text" in claims_map else []
    meta_conflicts = check_metadata_conflicts(
        bundle.metadata, text_dates, text_locations
    )

    cross_modal_fallback = img_txt_fallback or spk_fallback or any(c.is_fallback for c in claims_map.values())
    cross_modal = CrossModalEvidence(
        image_text_similarity=img_txt_sim,
        speaker_similarity=spk_sim,
        metadata_conflicts=meta_conflicts,
        contradictions=contradictions,
        is_fallback=cross_modal_fallback,
        fallback_reason="Some cross-modal verifiers used lightweight fallbacks" if cross_modal_fallback else None
    )

    # -------------------------------------------------------------
    # Layer 3: Multimodal Fusion, Calibration & Abstention
    # -------------------------------------------------------------
    feature_vector = build_fusion_features(modality_results, cross_modal)
    raw_probabilities = classifier.predict_proba(feature_vector)
    
    scaler = TemperatureScaler(temperature=1.20)
    calibrated_probabilities = scaler.calibrate_probabilities(raw_probabilities)

    total_present = float(sum(1 for ev in modality_results.values() if ev.present))
    abstained, abstain_reason, final_label, final_confidence = evaluate_abstention(
        calibrated_probabilities, total_present
    )

    # -------------------------------------------------------------
    # Layer 4: Evidence Ranking & Grounded Intelligence Report
    # -------------------------------------------------------------
    ranked_evidence = rank_evidence(modality_results, cross_modal, final_label)
    report_text, report_verified = generate_llm_report(
        final_label,
        final_confidence,
        modality_results,
        cross_modal,
        ranked_evidence
    )

    overall_fallback = any(ev.is_fallback for ev in modality_results.values()) or cross_modal.is_fallback or (not report_verified)

    return VerdictResult(
        bundle_id=bundle_id,
        label=final_label,
        confidence=final_confidence,
        probabilities=calibrated_probabilities,
        abstained=abstained,
        abstain_reason=abstain_reason,
        ranked_evidence=ranked_evidence,
        report=report_text,
        report_verified=report_verified,
        is_fallback=overall_fallback,
        modality_results=modality_results,
        cross_modal_results=cross_modal
    )


# Backwards compatibility for legacy file list analyzer
def generate_verdict(files, claims, conflicts):
    file_count = len(files)
    conflict_count = len(conflicts)

    if file_count == 0 or len(claims) == 0:
        return {
            "label": "insufficient_evidence",
            "confidence": 0.20,
            "reason": "Insufficient claims or source files were available."
        }

    if conflict_count == 0:
        if file_count >= 2:
            return {
                "label": "authentic",
                "confidence": 0.85,
                "reason": "The analyzed sources contain consistent structured claims with no detected cross-file conflicts."
            }
        return {
            "label": "insufficient_evidence",
            "confidence": 0.40,
            "reason": "Only one source is available, so cross-source consistency cannot be established."
        }

    conflict_fields = {c.get("field") for c in conflicts if c.get("field")}
    if file_count >= 3 and len(conflict_fields) >= 2:
        return {
            "label": "coordinated_synthetic",
            "confidence": 0.90,
            "reason": "Multiple sources contain significant contradictions across multiple factual fields."
        }

    return {
        "label": "manipulated",
        "confidence": 0.75,
        "reason": "At least one significant cross-source contradiction was detected."
    }