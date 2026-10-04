import io
import math
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ImageChops, ImageEnhance
import numpy as np

from backend.models.bundle import ModalityEvidence
from backend.config import DEVICE


_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def estimate_jpeg_quality(image: Image.Image) -> Optional[float]:
    """Estimate IJG JPEG quality (1-100) from the luminance quantisation table; None if not a JPEG."""
    tables = getattr(image, "quantization", None)
    if not tables:
        return None
    table = np.array(tables[0], dtype=np.float64)
    if table.size != 64:
        return None
    r = float(np.mean(np.maximum(table, 1.0) / _STD_LUMA))
    q = 50.0 / r if r > 1.0 else 100.0 - 50.0 * r
    return float(np.clip(q, 1.0, 100.0))


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


def compute_local_inconsistency(image: Image.Image, block: int = 32, jpeg_quality: Optional[float] = None) -> tuple[float, Dict[str, float]]:
    """
    Sensor-noise consistency across the image. A camera leaves a roughly uniform noise
    residual everywhere; splicing, blurring, noise injection or re-compression of a region
    changes that region's residual. We estimate the noise level per block (MAD of the
    Laplacian, which ignores sparse true edges) and flag blocks that are robust outliers.
    Returns (score in [0,1], metrics).
    """
    gray = np.array(image.convert("RGB").resize((256, 256)).convert("L"), dtype=np.float32)
    lap = np.pad(
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:], 1
    )
    n = 256 // block
    sig = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            b = lap[i * block:(i + 1) * block, j * block:(j + 1) * block]
            sig[i, j] = np.median(np.abs(b - np.median(b))) * 1.4826
    s = np.log(sig.ravel() + 0.5)
    med = float(np.median(s))
    mad = max(float(np.median(np.abs(s - med))) * 1.4826, 0.10)  # floor: ignore <10% variation
    z = np.abs(s - med) / mad
    peak = float(np.max(z))
    # ~4 is ordinary variation for an untouched image. Re-compressed / rescaled images (resharing)
    # vary more by themselves, so the evidence bar rises as JPEG quality drops: we would rather
    # miss a subtle edit in a laundered file than accuse an innocent one.
    rel = 0.8 if jpeg_quality is None else float(np.clip((jpeg_quality - 50.0) / 40.0, 0.0, 1.0))
    threshold = 5.0 + 12.0 * (1.0 - rel)
    score = float(np.clip((peak - threshold) / 20.0, 0.0, 1.0))
    return score, {"noise_block_peak_z": peak, "evidence_threshold_z": threshold}


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
            jpeg_q = estimate_jpeg_quality(img)
            local_score, local_metrics = compute_local_inconsistency(img, jpeg_quality=jpeg_q)
            global_score = float(np.clip(0.55 * ela_score * 3.0 + 0.45 * fft_score, 0.0, 1.0))
            combined_score = float(np.clip(max(global_score, 0.30 + 0.65 * local_score) if local_score > 0.0 else global_score, 0.0, 1.0))
            
            # Optional learned classifier: adds evidence, never clears a heuristic anomaly.
            learned = None
            try:
                from backend.learned import image_deepfake, registry
                learned = image_deepfake.predict(img)
                if learned is not None:
                    learned_anomaly = registry.prob_to_anomaly(learned["p_fake"], 0.28, 0.67)
                    combined_score = float(max(combined_score, learned_anomaly))
            except Exception:
                learned = None

            evidence_parts = []
            if learned is not None:
                evidence_parts.append(
                    f"Learned classifier ({learned['model_id']}) estimates {learned['p_fake']:.0%} probability the image is AI-generated."
                )
            if ela_score > 0.08:
                evidence_parts.append(f"Elevated compression error level variance ({ela_score:.3f}) suggests editing/splicing.")
            else:
                evidence_parts.append(f"Uniform compression error level ({ela_score:.3f}).")
                
            if local_score > 0.15:
                evidence_parts.append(
                    f"Localised inconsistency: one region's sensor-noise level deviates from the rest of the image "
                    f"(robust z={local_metrics['noise_block_peak_z']:.1f}), consistent with splicing, retouching or re-compression."
                )
            if jpeg_q is not None and jpeg_q < 80:
                evidence_parts.append(f"Heavily compressed input (estimated JPEG quality ~{jpeg_q:.0f}): pixel forensics are less reliable.")
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
                    "local_inconsistency": local_score,
                    **({"learned_fake_prob": learned["p_fake"]} if learned is not None else {}),
                    **({"jpeg_quality": jpeg_q} if jpeg_q is not None else {}),
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

