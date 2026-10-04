import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image, ExifTags

from backend.models.bundle import ModalityEvidence


SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "canva", "midjourney", "stable diffusion",
    "dall-e", "novelai", "automatic1111", "comfyui", "invokeai", "paint.net"
]


def parse_exif_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    v = str(value).strip()
    m = re.match(r"^(\d{4})[:\-/](\d{1,2})[:\-/](\d{1,2})[ T](\d{1,2}):(\d{1,2}):(\d{1,2})", v)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)), int(m.group(6)))
        except ValueError:
            return None
    return None


def extract_exif_dict(image_path: str) -> Dict[str, Any]:
    """
    Extracts readable EXIF metadata dictionary using Pillow.
    """
    exif_data = {}
    try:
        with Image.open(image_path) as img:
            raw_exif = img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Handle binary data or large values gracefully
                    if isinstance(value, (bytes, bytearray)):
                        continue
                    exif_data[tag_name] = str(value)
    except Exception:
        pass
    return exif_data


def analyze_metadata(
    image_path: Optional[str] = None,
    raw_metadata: Optional[Dict[str, Any]] = None
) -> ModalityEvidence:
    """
    Layer 1 metadata detector.
    Inspects EXIF and container metadata for editing software traces,
    missing camera signatures, and timestamp manipulation.
    """
    combined_meta = {}
    if raw_metadata:
        combined_meta.update(raw_metadata)
        
    exif_present = False
    if image_path and Path(image_path).exists():
        exif = extract_exif_dict(image_path)
        if exif:
            exif_present = True
            combined_meta.update(exif)
            
    if not combined_meta and not exif_present:
        return ModalityEvidence(
            modality="metadata",
            present=False,
            score=0.40,
            evidence="No metadata or EXIF headers found; metadata stripped or unavailable.",
            features={"metadata_stripped": 1.0, "software_tampered": 0.0},
            is_fallback=False
        )
        
    anomaly_score = 0.10
    evidence_parts = []
    
    # 1. Check software tag
    software_val = str(combined_meta.get("Software", "")).lower()
    tampered_software = False
    for kw in SUSPICIOUS_SOFTWARE_KEYWORDS:
        if kw in software_val:
            tampered_software = True
            anomaly_score += 0.55
            evidence_parts.append(f"Editing software signature identified: '{combined_meta.get('Software')}'.")
            break
            
    # 2. Check camera hardware signatures
    has_make = "Make" in combined_meta or "Model" in combined_meta
    if not has_make and not tampered_software:
        anomaly_score += 0.25
        evidence_parts.append("Absence of camera hardware Make/Model metadata tags.")
    elif has_make:
        evidence_parts.append(f"Camera hardware profile recorded: {combined_meta.get('Make', '')} {combined_meta.get('Model', '')}.")
        
    # 3. Check timestamps (graded by gap size: <5s clean, >1h 0.25, >24h 0.50)
    datetime_original = combined_meta.get("DateTimeOriginal")
    datetime_modified = combined_meta.get("DateTime")
    gap_seconds = None
    if datetime_original and datetime_modified:
        dt_orig = parse_exif_timestamp(datetime_original)
        dt_mod = parse_exif_timestamp(datetime_modified)
        if dt_orig and dt_mod:
            gap_seconds = abs((dt_mod - dt_orig).total_seconds())
            if gap_seconds > 86400:  # > 24 hours
                anomaly_score += 0.50
                evidence_parts.append(f"Major timestamp gap ({gap_seconds / 3600:.1f} hours) between capture ({datetime_original}) and file write ({datetime_modified}).")
            elif gap_seconds > 3600:  # > 1 hour
                anomaly_score += 0.25
                evidence_parts.append(f"Timestamp gap ({gap_seconds / 60:.1f} minutes) between capture ({datetime_original}) and file write ({datetime_modified}).")
            elif gap_seconds >= 5:  # 5s to 1 hour
                anomaly_score += 0.10
                evidence_parts.append(f"Minor timestamp gap ({gap_seconds:.0f} seconds) between capture ({datetime_original}) and file write ({datetime_modified}).")
            # If gap_seconds < 5s: clean (0 added)
        elif datetime_original != datetime_modified:
            anomaly_score += 0.20
            evidence_parts.append(f"Timestamp discrepancy between capture ({datetime_original}) and file write ({datetime_modified}).")
        
    anomaly_score = float(min(0.95, max(0.05, anomaly_score)))
    
    return ModalityEvidence(
        modality="metadata",
        present=True,
        score=anomaly_score,
        evidence=" ".join(evidence_parts),
        features={
            "software_tampered": 1.0 if tampered_software else 0.0,
            "has_camera_profile": 1.0 if has_make else 0.0,
            "timestamp_gap_seconds": gap_seconds if gap_seconds is not None else 0.0,
            "metadata_anomaly": anomaly_score
        },
        is_fallback=False
    )

