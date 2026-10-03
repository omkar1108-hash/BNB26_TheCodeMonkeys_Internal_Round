from typing import List, Dict, Optional
from backend.models.bundle import ModalityEvidence, CrossModalEvidence, EvidenceItem


def rank_evidence(
    modality_results: Dict[str, ModalityEvidence],
    cross_modal: Optional[CrossModalEvidence],
    predicted_label: str
) -> List[EvidenceItem]:
    """
    Ranks findings by anomaly strength and feature importance.
    Highlights key contradictions and per-modality artifacts.
    """
    candidates = []

    # 1. Contradictions from cross-modal analysis (Crucial for Coordinated Synthetic)
    if cross_modal and cross_modal.contradictions:
        for c in cross_modal.contradictions:
            importance = 0.95 if c.severity == "high" else 0.70
            candidates.append({
                "category": "cross_modal",
                "modality": "cross_modal",
                "description": c.description,
                "importance": importance,
                "is_fallback": False
            })

    # 2. Metadata conflicts
    if cross_modal and cross_modal.metadata_conflicts:
        for mc in cross_modal.metadata_conflicts:
            candidates.append({
                "category": "metadata",
                "modality": "metadata",
                "description": mc,
                "importance": 0.85,
                "is_fallback": False
            })

    # 3. Low cross-modal image-text similarity
    if cross_modal and cross_modal.image_text_similarity is not None:
        sim = cross_modal.image_text_similarity
        if sim < 0.40:
            candidates.append({
                "category": "cross_modal",
                "modality": "image_text",
                "description": f"Significant visual-textual semantic misalignment (similarity: {sim:.2f}).",
                "importance": 0.80,
                "is_fallback": cross_modal.is_fallback
            })

    # 4. Per-modality anomalies
    for mod_name, ev in modality_results.items():
        if ev.present and ev.score > 0.45:
            candidates.append({
                "category": "modality",
                "modality": mod_name,
                "description": f"[{mod_name.upper()}] {ev.evidence}",
                "importance": float(ev.score),
                "is_fallback": ev.is_fallback
            })
        elif ev.present and ev.score <= 0.25 and predicted_label == "authentic":
            candidates.append({
                "category": "modality",
                "modality": mod_name,
                "description": f"[{mod_name.upper()}] Artifact displays authentic physical capture signatures.",
                "importance": float(1.0 - ev.score),
                "is_fallback": ev.is_fallback
            })

    # Sort descending by importance
    candidates.sort(key=lambda x: x["importance"], reverse=True)

    ranked = []
    for idx, c in enumerate(candidates, start=1):
        ranked.append(EvidenceItem(
            rank=idx,
            category=c["category"],
            modality=c["modality"],
            description=c["description"],
            importance=c["importance"],
            is_fallback=c["is_fallback"]
        ))

    return ranked

