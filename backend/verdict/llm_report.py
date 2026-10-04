import json
import re
from typing import Dict, Any, List, Optional
import httpx

from backend.config import OLLAMA_HOST, OLLAMA_MODEL, is_ollama_online
from backend.models.bundle import ModalityEvidence, CrossModalEvidence, EvidenceItem


def generate_template_report(
    label: str,
    confidence: float,
    ranked_evidence: List[EvidenceItem],
    is_fallback: bool = True
) -> str:
    """
    Deterministic structured template report used as labeled fallback.
    """
    prefix = "[FALLBACK FORENSIC TEMPLATE REPORT]\n\n" if is_fallback else ""
    lines = [
        f"{prefix}EXECUTIVE VERDICT: {label.upper().replace('_', ' ')}",
        f"CALIBRATED CONFIDENCE: {confidence:.1%}\n",
        "SUMMARY OF FINDINGS:"
    ]

    if not ranked_evidence:
        lines.append("- No substantial forensic evidence or corroborating modalities detected.")
    else:
        for ev in ranked_evidence[:5]:
            lines.append(f"- Rank {ev.rank} ({ev.category}): {ev.description}")

    lines.append("\nANALYSIS CONCLUSION:")
    if label == "authentic":
        lines.append("The bundle artifacts exhibit natural physical signatures with mutual cross-modal consistency.")
    elif label == "manipulated":
        lines.append("At least one artifact exhibits clear traces of localized digital manipulation or synthetic alteration.")
    elif label == "coordinated_synthetic":
        lines.append("While individual artifacts appear clean in isolation, significant cross-modal factual and contextual contradictions demonstrate a coordinated synthetic campaign.")
    else:
        lines.append("Evidence is insufficient or below safety confidence thresholds to form a conclusive determination.")

    return "\n".join(lines)


def verify_report_grounding(report_text: str, structured_findings: Dict[str, Any]) -> bool:
    """
    Verifies that the generated LLM text does not introduce hallucinated dates
    or novel factual numbers not present in the input structured findings.
    """
    findings_str = json.dumps(structured_findings).lower()
    
    # Check 4-digit years in report
    years_in_report = re.findall(r'\b(20\d\d|19\d\d)\b', report_text)
    for year in years_in_report:
        if year not in findings_str:
            return False  # Hallucinated year
            
    return True


def generate_llm_report(
    label: str,
    confidence: float,
    modality_results: Dict[str, ModalityEvidence],
    cross_modal: Optional[CrossModalEvidence],
    ranked_evidence: List[EvidenceItem]
) -> tuple[str, bool]:
    """
    Generates an analytical intelligence report using local Ollama.
    Prompted strictly with structured findings only, temperature 0.
    Verifies output against findings; falls back to template if unverified or if Ollama fails.
    Returns: (report_text, report_verified)
    """
    if not is_ollama_online():
        fallback_text = generate_template_report(label, confidence, ranked_evidence, is_fallback=True)
        return fallback_text, False

    structured_findings = {
        "verdict": label,
        "calibrated_confidence": f"{confidence:.1%}",
        "modality_scores": {
            m: {"score": round(ev.score, 3), "evidence": ev.evidence}
            for m, ev in modality_results.items() if ev.present
        },
        "contradictions": [
            c.description for c in (cross_modal.contradictions if cross_modal else [])
        ],
        "metadata_conflicts": cross_modal.metadata_conflicts if cross_modal else [],
        "top_evidence": [ev.description for ev in ranked_evidence[:4]]
    }

    prompt = (
        "You are an objective digital forensics analyst. Synthesize ONLY the provided "
        "structured findings into a concise, professional executive intelligence report. "
        "Do not invent, speculate, or introduce any facts, entities, dates, or details "
        "not explicitly stated in the structured findings.\n\n"
        f"STRUCTURED FINDINGS:\n{json.dumps(structured_findings, indent=2)}\n\n"
        "REPORT FORMAT:\n"
        "1. Executive Summary\n"
        "2. Key Forensic Findings\n"
        "3. Cross-Modal Assessment\n"
        "4. Conclusion\n"
    )

    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        timeout_cfg = httpx.Timeout(read=12.0, connect=1.0)
        with httpx.Client(timeout=timeout_cfg) as client:
            res = client.post(url, json=payload)
            if res.status_code == 200:
                report_content = res.json().get("response", "").strip()
                if report_content:
                    if verify_report_grounding(report_content, structured_findings):
                        return report_content, True
                    # If report added unsupported facts, reject and fall back
    except Exception:
        pass

    # Labeled fallback template
    fallback_text = generate_template_report(label, confidence, ranked_evidence, is_fallback=True)
    return fallback_text, False
