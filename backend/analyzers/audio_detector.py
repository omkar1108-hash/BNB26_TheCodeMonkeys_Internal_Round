import wave
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from backend.models.bundle import ModalityEvidence
from backend.config import DEVICE


def extract_audio_spectral_features(file_path: str) -> Dict[str, float]:
    """
    Extract spectral and temporal features using standard wave and numpy.
    Works natively on standard WAV PCM files.
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        raw_bytes = wf.readframes(n_frames)
        
        dtype = np.int16 if sampwidth == 2 else (np.int32 if sampwidth == 4 else np.uint8)
        audio = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)
        
        if n_channels > 1:
            audio = audio[::n_channels]  # downmix mono
            
        if len(audio) == 0:
            return {"zero_crossing_rate": 0.0, "spectral_centroid": 0.0, "rolloff": 0.0, "rms": 0.0}
            
        # Normalize
        max_val = np.max(np.abs(audio)) + 1e-9
        audio = audio / max_val
        
        # Zero crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
        
        # RMS energy
        rms = float(np.sqrt(np.mean(audio ** 2)))
        
        # FFT Spectrum
        fft_vals = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / framerate)
        
        # Spectral Centroid
        sum_fft = np.sum(fft_vals) + 1e-9
        spectral_centroid = float(np.sum(freqs * fft_vals) / sum_fft)
        
        # Spectral Rolloff (85% energy)
        cumulative_energy = np.cumsum(fft_vals)
        threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = np.searchsorted(cumulative_energy, threshold)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        
        return {
            "zero_crossing_rate": zcr,
            "spectral_centroid": spectral_centroid,
            "rolloff": rolloff,
            "rms": rms,
            "framerate": float(framerate)
        }


def analyze_audio(audio_path: Optional[str]) -> ModalityEvidence:
    """
    Layer 1 audio detector.
    Analyzes acoustic features (centroid, rolloff, ZCR) and flags synthetic / spoof signatures.
    """
    if not audio_path or not Path(audio_path).exists():
        return ModalityEvidence(
            modality="audio",
            present=False,
            score=0.0,
            evidence="No audio artifact provided in bundle.",
            features={},
            is_fallback=False
        )
        
    path = Path(audio_path)
    
    # Non-WAV or synthetic stub handling
    if path.suffix.lower() != ".wav":
        return ModalityEvidence(
            modality="audio",
            present=True,
            score=0.50,
            evidence=f"Audio format '{path.suffix}' detected; fallback acoustic estimation applied.",
            features={"audio_anomaly": 0.50},
            is_fallback=True,
            fallback_reason="Non-WAV audio passed without ffmpeg transcoding"
        )
        
    try:
        features = extract_audio_spectral_features(str(path))
        
        # Deepfake synthetic voice vocoders (HiFi-GAN, WaveGlow) frequently leave
        # unnatural high-frequency cutoff or abnormally high zero-crossing jitter
        zcr = features["zero_crossing_rate"]
        centroid = features["spectral_centroid"]
        rolloff = features["rolloff"]
        
        # Heuristic anomaly indicator calibrated for 16kHz speech
        anomaly_score = 0.15
        evidence_parts = []
        
        if rolloff < 2800 and features.get("framerate", 16000) >= 16000:
            anomaly_score += 0.35
            evidence_parts.append("Low spectral rolloff suggests vocoder band-limiting.")
        elif rolloff > 7000:
            evidence_parts.append("Wide natural acoustic frequency bandwidth.")
            
        if zcr > 0.30:
            anomaly_score += 0.30
            evidence_parts.append("High zero-crossing rate indicating unnatural high-frequency vocoder buzz.")
        else:
            evidence_parts.append("Zero-crossing rate within normal speech dynamics.")
            
        anomaly_score = float(np.clip(anomaly_score, 0.05, 0.95))
        features["audio_anomaly"] = anomaly_score
        
        return ModalityEvidence(
            modality="audio",
            present=True,
            score=anomaly_score,
            evidence=" ".join(evidence_parts),
            features=features,
            is_fallback=False
        )
        
    except Exception as e:
        return ModalityEvidence(
            modality="audio",
            present=True,
            score=0.50,
            evidence=f"Audio analysis fallback triggered: {str(e)}",
            features={"audio_anomaly": 0.50},
            is_fallback=True,
            fallback_reason=f"Audio analysis failure: {str(e)}"
        )

