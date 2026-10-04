"""Counterfactual explanations: 'what would the verdict be without artifact X?'"""
from typing import Dict, List, Optional

from backend.models.bundle import (
    ModalityEvidence, CrossModalEvidence, Counterfactual
)
from backend.fusion.feature_builder import build_fusion_features
from backend.fusion.decision import decide


def _absent(modality: str) -> ModalityEvidence:
    return ModalityEvidence(modality=modality, present=False, score=0.0, evidence="removed")


def _without(
    modality: str,
    modality_results: Dict[str, ModalityEvidence],
    cross: Optional[CrossModalEvidence],
):
    mods = dict(modality_results)
    mods[modality] = _absent(modality)
    cm = cross.model_copy(deep=True) if cross else None
    if cm is not None:
        if modality == "image":
            cm.image_text_similarity = None
        elif modality == "audio":
            cm.speaker_similarity = None
            cm.contradictions = [c for c in cm.contradictions if c.field != "speaker"]
        elif modality == "metadata":
            cm.metadata_conflicts = []
        elif modality == "text":
            cm.image_text_similarity = None
            cm.contradictions = []
            cm.metadata_conflicts = []
            cm.n_text_sources = 0
    return mods, cm


def compute_counterfactuals(
    modality_results: Dict[str, ModalityEvidence],
    cross: Optional[CrossModalEvidence],
    base_label: str,
) -> List[Counterfactual]:
    present = [m for m, ev in modality_results.items() if ev.present]
    if len(present) < 2:
        return []
    out: List[Counterfactual] = []
    for m in present:
        mods, cm = _without(m, modality_results, cross)
        d = decide(build_fusion_features(mods, cm))
        flips = d.label != base_label
        note = (
            f"Without the {m} artifact the verdict becomes {d.label.replace('_', ' ')}."
            if flips else
            f"Removing the {m} artifact does not change the verdict."
        )
        out.append(Counterfactual(
            removed=m, label=d.label, confidence=d.confidence,
            flips_verdict=flips, note=note,
        ))
    return out
