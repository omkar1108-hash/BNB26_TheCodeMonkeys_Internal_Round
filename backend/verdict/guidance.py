"""'What would settle it?' guidance, produced when evidence is thin or ambiguous."""
from typing import Dict, List, Optional

from backend.models.bundle import ModalityEvidence, CrossModalEvidence


def suggest_next_steps(
    label: str,
    abstained: bool,
    conformal_set: List[str],
    modality_results: Dict[str, ModalityEvidence],
    cross: Optional[CrossModalEvidence],
    has_claimed_speaker: bool,
) -> List[str]:
    steps: List[str] = []
    present = {m for m, ev in modality_results.items() if ev.present}
    n_sources = cross.n_text_sources if cross else 0

    if not abstained and label in ("authentic",) and len(present) >= 2:
        return steps  # confident, corroborated: nothing to ask for

    if "image" not in present:
        steps.append("Provide the original image file (not a screenshot) so pixel-level forensics and EXIF can be checked.")
    if "metadata" not in present:
        steps.append("Provide capture metadata (EXIF or a sidecar JSON with DateTimeOriginal and Location) to verify when and where it was captured.")
    if "text" not in present:
        steps.append("Provide the accompanying caption or message so its claims can be cross-checked.")
    elif n_sources < 2:
        steps.append("Provide a second independent text source (report, caption or article) so dates, places and people can be compared across sources.")
    if "audio" in present and not has_claimed_speaker:
        steps.append("State who the audio is claimed to be from so the speaker claim can be compared with the text.")
    if "audio" not in present and abstained:
        steps.append("If an audio or video recording exists, provide it for spoof detection and a speaker check.")

    fb = [m for m, ev in modality_results.items() if ev.present and ev.is_fallback]
    if fb:
        steps.append(f"Heuristic fallbacks were used for: {', '.join(fb)}. Re-run with the full model weights for stronger evidence.")
    if cross and cross.is_fallback:
        steps.append("Some cross-modal checks used fallbacks (for example CLIP weights or Ollama unavailable).")

    if len(conformal_set) > 1:
        names = " / ".join(c.replace("_", " ") for c in conformal_set)
        steps.insert(0, f"The evidence is consistent with more than one outcome ({names}); the items below would help separate them.")
    return steps[:5]
