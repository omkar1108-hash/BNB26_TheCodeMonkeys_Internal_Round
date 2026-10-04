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
from backend.analyzers.text_detector import analyze_text, analyze_text_sources
from backend.analyzers.metadata_detector import analyze_metadata
from backend.cross_modal.image_text import compute_image_text_similarity
from backend.cross_modal.audio_speaker import verify_claimed_speaker
from backend.cross_modal.metadata_rules import check_metadata_conflicts
from backend.claims.extractor import extract_claims
from backend.claims.conflict import find_bundle_contradictions
from backend.claims.evidence_graph import build_bundle_graph
from backend.fusion.feature_builder import build_fusion_features
from backend.fusion.decision import decide
from backend.verdict.explainer import rank_evidence
from backend.verdict.llm_report import generate_llm_report
from backend.verdict.counterfactual import compute_counterfactuals
from backend.verdict.guidance import suggest_next_steps


CLIP_MISMATCH_GUARD = 0.30


def collect_text_sources(bundle: BundleInput) -> Dict[str, str]:
    """All named text sources in the bundle (caption/message/report/...)."""
    sources: Dict[str, str] = {}
    if bundle.text and bundle.text.strip():
        sources["text"] = bundle.text.strip()
    for name, content in (bundle.documents or {}).items():
        if content and content.strip():
            key = name if name not in sources else f"{name}_2"
            sources[key] = content.strip()
    return sources


def compute_bundle_signals(bundle: BundleInput) -> Dict[str, Any]:
    """Layers 1-2: per-modality detectors + cross-modal verification (no fusion)."""
    text_sources = collect_text_sources(bundle)
    combined_text = "\n\n".join(text_sources.values()) if text_sources else None

    # -------------------------------------------------------------
    # Layer 1: Per-Modality Detectors
    # -------------------------------------------------------------
    image_ev = analyze_image(bundle.image_path)
    audio_ev = analyze_audio(bundle.audio_path)
    text_ev = analyze_text_sources(text_sources)
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
    img_txt_sim, img_txt_fallback, img_txt_reason = compute_image_text_similarity(
        bundle.image_path, list(text_sources.values())
    )
    spk_sim, spk_fallback, spk_reason = verify_claimed_speaker(
        bundle.audio_path, bundle.claimed_speaker
    )

    claims_map: Dict[str, ExtractedClaims] = {}
    for name, content in text_sources.items():
        claims_map[name] = extract_claims(content)
    if bundle.claimed_speaker:
        claims_map["audio"] = ExtractedClaims(claimed_speaker=bundle.claimed_speaker)

    contradictions = find_bundle_contradictions(claims_map)

    # Capture-metadata vs each text source (calendar date and place)
    meta_conflicts: List[str] = []
    meta_for_rules = dict(bundle.metadata or {})
    for name, claims in claims_map.items():
        if name == "audio":
            continue
        meta_conflicts.extend(check_metadata_conflicts(
            meta_for_rules, claims.dates, claims.locations, source_label=name
        ))

    text_claims = [c for n, c in claims_map.items() if n != "audio"]
    extractor_fallback = any(c.is_fallback for c in text_claims)
    cross_modal_fallback = bool(img_txt_fallback or spk_fallback)
    cross_modal = CrossModalEvidence(
        image_text_similarity=img_txt_sim,
        speaker_similarity=spk_sim,
        metadata_conflicts=meta_conflicts,
        contradictions=contradictions,
        n_text_sources=len(text_sources),
        is_fallback=cross_modal_fallback or extractor_fallback,
        fallback_reason=(
            "; ".join(r for r in (img_txt_reason, spk_reason) if r)
            or ("Claim extraction used the rule-based fallback" if extractor_fallback else None)
            if (cross_modal_fallback or extractor_fallback) else None
        )
    )

    return {
        "text_sources": text_sources,
        "modality_results": modality_results,
        "claims_map": claims_map,
        "contradictions": contradictions,
        "cross_modal": cross_modal,
        "extractor_fallback": extractor_fallback,
        "cross_modal_fallback": cross_modal_fallback,
        "text_claims": text_claims,
    }


def analyze_bundle(bundle: BundleInput) -> VerdictResult:
    """
    Executes the full 4-layer TrustLayer verification pipeline on a multimodal bundle.
    """
    bundle_id = bundle.bundle_id or str(uuid.uuid4())
    sig = compute_bundle_signals(bundle)
    text_sources = sig["text_sources"]
    modality_results = sig["modality_results"]
    claims_map = sig["claims_map"]
    contradictions = sig["contradictions"]
    cross_modal = sig["cross_modal"]
    extractor_fallback = sig["extractor_fallback"]
    cross_modal_fallback = sig["cross_modal_fallback"]
    text_claims = sig["text_claims"]

    # -------------------------------------------------------------
    # Layer 3: Multimodal Fusion, Calibration & Abstention
    # -------------------------------------------------------------
    feature_vector = build_fusion_features(modality_results, cross_modal)
    decision = decide(feature_vector)
    final_label = decision.label
    final_confidence = decision.confidence
    abstained = decision.abstained
    abstain_reason = decision.abstain_reason

    # Authenticity guard: if CLIP says the text does not describe the image, do not certify the
    # bundle as authentic (a real photo with a false caption is still misleading context).
    sim = cross_modal.image_text_similarity
    if (not abstained) and final_label == "authentic" and sim is not None and sim < CLIP_MISMATCH_GUARD:
        abstained, final_label = True, "insufficient_evidence"
        abstain_reason = (
            f"The text does not appear to describe the image (CLIP agreement {sim:.2f}); "
            "an authentic verdict needs the caption and image to match."
        )

    # -------------------------------------------------------------
    # Layer 4: Evidence, Counterfactuals, Guidance & Grounded Report
    # -------------------------------------------------------------
    ranked_evidence = rank_evidence(modality_results, cross_modal, final_label)
    counterfactuals = compute_counterfactuals(modality_results, cross_modal, final_label)
    next_steps = suggest_next_steps(
        final_label, abstained, decision.conformal_set,
        modality_results, cross_modal, bool(bundle.claimed_speaker)
    )
    evidence_graph = build_bundle_graph(
        claims_map, bundle.metadata, modality_results,
        bundle.claimed_speaker, contradictions
    )
    report_text, report_verified = generate_llm_report(
        final_label,
        final_confidence,
        modality_results,
        cross_modal,
        ranked_evidence
    )

    detector_fallback = any(ev.is_fallback for ev in modality_results.values())
    classifier_fallback = decision.model_info.get("type") == "rule_based_fallback"
    # signal_fallback = a fallback that could change the verdict (excludes LLM wording/extraction)
    signal_fallback = bool(detector_fallback or cross_modal_fallback or classifier_fallback)
    overall_fallback = signal_fallback or extractor_fallback or (not report_verified)

    model_info = dict(decision.model_info)
    learned_used = {
        m: ev.features["learned_fake_prob"] for m, ev in modality_results.items()
        if ev.present and "learned_fake_prob" in ev.features
    }
    from backend.learned import registry as _learned_registry
    model_info.update({
        "learned_detectors_used": learned_used,
        "learned_models": _learned_registry.status() if _learned_registry.enabled() else "off",
        "clip_used": cross_modal.image_text_similarity is not None,
        "signal_fallback": signal_fallback,
        "claim_extractor": "rule_based" if extractor_fallback else ("ollama" if text_claims else "none"),
        "report": "ollama_verified" if report_verified else "template_fallback",
    })

    return VerdictResult(
        bundle_id=bundle_id,
        label=final_label,
        confidence=final_confidence,
        probabilities=decision.probabilities,
        abstained=abstained,
        abstain_reason=abstain_reason,
        ranked_evidence=ranked_evidence,
        report=report_text,
        report_verified=report_verified,
        is_fallback=overall_fallback,
        modality_results=modality_results,
        cross_modal_results=cross_modal,
        counterfactuals=counterfactuals,
        next_steps=next_steps,
        evidence_graph=evidence_graph,
        conformal_set=decision.conformal_set,
        model_info=model_info,
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