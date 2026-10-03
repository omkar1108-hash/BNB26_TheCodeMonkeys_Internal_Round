import math
import re
from typing import Optional, Dict, Any
import numpy as np

from backend.models.bundle import ModalityEvidence


def compute_stylometry(text: str) -> Dict[str, float]:
    """
    Computes lexical diversity, sentence length variance, and punctuation entropy.
    LLM-generated text typically displays lower vocabulary bursts, uniform sentence length,
    and predictable punctuation distributions.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return {
            "type_token_ratio": 0.0,
            "avg_sentence_len": 0.0,
            "sentence_len_var": 0.0,
            "punctuation_entropy": 0.0,
            "word_count": 0.0
        }
        
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if sentences:
        sentence_lens = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        avg_len = float(np.mean(sentence_lens))
        len_var = float(np.var(sentence_lens))
    else:
        avg_len = float(len(words))
        len_var = 0.0
        
    # Punctuation distribution entropy
    punct_matches = re.findall(r'[,;:—\-\(\)]', text)
    if punct_matches:
        counts = [punct_matches.count(p) for p in set(punct_matches)]
        total = sum(counts)
        probs = [c / total for c in counts]
        punct_entropy = -sum(p * math.log2(p) for p in probs)
    else:
        punct_entropy = 0.0
        
    return {
        "type_token_ratio": float(ttr),
        "avg_sentence_len": float(avg_len),
        "sentence_len_var": float(len_var),
        "punctuation_entropy": float(punct_entropy),
        "word_count": float(len(words))
    }


def estimate_perplexity_score(text: str, stylometry: Dict[str, float]) -> float:
    """
    Calculates an estimated generation anomaly score based on stylometric signatures
    (burstiness and lexical predictability).
    Higher score indicates higher likelihood of AI-generated/synthetic text.
    """
    score = 0.20
    
    # AI generated text tends to have very uniform sentence lengths (low variance)
    # and moderate-to-high TTR without rare idiosyncratic vocabulary
    len_var = stylometry["sentence_len_var"]
    word_count = stylometry["word_count"]
    ttr = stylometry["type_token_ratio"]
    
    if word_count > 20:
        if len_var < 8.0:
            score += 0.35  # unnaturally uniform sentences
        elif len_var > 35.0:
            score -= 0.10  # natural human burstiness
            
        if 0.65 < ttr < 0.85:
            score += 0.20
            
    return float(np.clip(score, 0.05, 0.95))


def analyze_text(text: Optional[str]) -> ModalityEvidence:
    """
    Layer 1 text detector analyzing stylometry and synthetic generation patterns.
    """
    if not text or not text.strip():
        return ModalityEvidence(
            modality="text",
            present=False,
            score=0.0,
            evidence="No text/caption artifact provided in bundle.",
            features={},
            is_fallback=False
        )
        
    try:
        stylometry = compute_stylometry(text)
        anomaly_score = estimate_perplexity_score(text, stylometry)
        
        evidence_parts = []
        if stylometry["sentence_len_var"] < 8.0 and stylometry["word_count"] > 25:
            evidence_parts.append(f"Low sentence length variance ({stylometry['sentence_len_var']:.1f}) exhibits synthetic uniformity.")
        else:
            evidence_parts.append(f"Sentence cadence displays organic human rhythm (var: {stylometry['sentence_len_var']:.1f}).")
            
        evidence_parts.append(f"Vocabulary lexical diversity (TTR: {stylometry['type_token_ratio']:.2f}).")
        
        stylometry["text_anomaly"] = anomaly_score
        
        return ModalityEvidence(
            modality="text",
            present=True,
            score=anomaly_score,
            evidence=" ".join(evidence_parts),
            features=stylometry,
            is_fallback=False
        )
        
    except Exception as e:
        return ModalityEvidence(
            modality="text",
            present=True,
            score=0.50,
            evidence=f"Text analysis fallback triggered: {str(e)}",
            features={"text_anomaly": 0.50},
            is_fallback=True,
            fallback_reason=f"Text parsing error: {str(e)}"
        )

