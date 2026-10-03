import io
import math
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

from backend.models.bundle import ModalityEvidence
from backend.config import DEVICE


def compute_ela(image: Image.Image, quality: int = 90) -> tuple[float, Image.Image]:
    """
    Error Level Analysis (ELA).
    Saves image at a known quality level and computes pixel difference with original.
    Higher mean difference / variance indicates potential digital manipulation or resaving.
    """
    with io.BytesIO() as buffer:
        rgb_image = image.convert("RGB")
        rgb_image.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        diff = ImageChops.difference(rgb_image, resaved)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max(max_diff, 1)
        diff_enhanced = ImageEnhance.Brightness(diff).enhance(scale)
        
        diff_np = np.array(diff, dtype=np.float32)
        ela_score = float(np.mean(diff_np) / 255.0)
        return ela_score, diff_enhanced


def compute_fft_frequency_score(image: Image.Image) -> tuple[float, Dict[str, float]]:
    """
    2D Fast Fourier Transform frequency domain analysis.
    Synthetic/generated images often exhibit anomalous high-frequency falloff
    or periodic grid artifacts from upsampling/deconvolution layers.
    """
    gray = image.convert("L").resize((256, 256))
    img_np = np.array(gray, dtype=np.float32)
    
    # 2D FFT and zero-frequency shift
    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-9)
    
    rows, cols = img_np.shape
    crow, ccol = rows // 2, cols // 2
    
    # Mask low frequency center (radius 30)
    y, x = np.ogrid[:rows, :cols]
    r = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    low_freq_mask = r <= 30
    high_freq_mask = r > 60
    
    low_energy = float(np.mean(magnitude_spectrum[low_freq_mask]))
    high_energy = float(np.mean(magnitude_spectrum[high_freq_mask]))
    
    # Ratio of high to low energy
    ratio = high_energy / max(low_energy, 1e-6)
    
    # Normalize anomaly score: natural images typically have ratio around 0.35-0.55
    anomaly = float(np.clip(abs(ratio - 0.45) * 2.5, 0.0, 1.0))
    
    return anomaly, {
        "low_energy": low_energy,
        "high_energy": high_energy,
        "freq_ratio": ratio
    }


def analyze_image(image_path: Optional[str]) -> ModalityEvidence:
    """
    Layer 1 image analyzer combining ELA and 2D FFT frequency analysis.
    """
    if not image_path or not Path(image_path).exists():
        return ModalityEvidence(
            modality="image",
            present=False,
            score=0.0,
            evidence="No image artifact provided in bundle.",
            features={},
            is_fallback=False
        )
    
    try:
        with Image.open(image_path) as img:
            ela_score, _ = compute_ela(img)
            fft_score, freq_metrics = compute_fft_frequency_score(img)
            
            # Combined image anomaly score
            combined_score = float(np.clip(0.55 * ela_score * 3.0 + 0.45 * fft_score, 0.0, 1.0))
            
            evidence_parts = []
            if ela_score > 0.08:
                evidence_parts.append(f"Elevated compression error level variance ({ela_score:.3f}) suggests editing/splicing.")
            else:
                evidence_parts.append(f"Uniform compression error level ({ela_score:.3f}).")
                
            if fft_score > 0.60:
                evidence_parts.append(f"Anomalous high-frequency spectral signature ({freq_metrics['freq_ratio']:.3f}) indicative of generative upsampling.")
            else:
                evidence_parts.append("Frequency spectrum conforms to typical photographic capture.")
                
            return ModalityEvidence(
                modality="image",
                present=True,
                score=combined_score,
                evidence=" ".join(evidence_parts),
                features={
                    "ela_score": ela_score,
                    "fft_score": fft_score,
                    "freq_ratio": freq_metrics["freq_ratio"],
                    "image_anomaly": combined_score
                },
                is_fallback=False
            )
            
    except Exception as e:
        return ModalityEvidence(
            modality="image",
            present=True,
            score=0.5,
            evidence=f"Image inspection fallback triggered due to read error: {str(e)}",
            features={"image_anomaly": 0.5},
            is_fallback=True,
            fallback_reason=f"Image processing failure: {str(e)}"
        )

