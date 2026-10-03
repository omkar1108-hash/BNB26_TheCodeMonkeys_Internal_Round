from typing import List, Dict, Any, Optional
from datetime import datetime
import re


def check_metadata_conflicts(
    metadata: Optional[Dict[str, Any]],
    extracted_dates: List[str],
    extracted_locations: List[str]
) -> List[str]:
    """
    Checks for cross-modal contradictions between EXIF metadata
    and claims found in text or audio transcripts.
    """
    conflicts = []
    if not metadata:
        return conflicts
        
    exif_date_str = metadata.get("DateTimeOriginal") or metadata.get("DateTime")
    if exif_date_str and extracted_dates:
        # Standard EXIF format: 'YYYY:MM:DD HH:MM:SS'
        try:
            exif_year = exif_date_str.split(":")[0].strip()
            for date_claim in extracted_dates:
                # Check for year divergence
                year_match = re.search(r'\b(20\d\d|19\d\d)\b', date_claim)
                if year_match:
                    claim_year = year_match.group(1)
                    if claim_year != exif_year:
                        conflicts.append(
                            f"Temporal conflict: EXIF recorded year ({exif_year}) contradicts claim text year ({claim_year})."
                        )
        except Exception:
            pass
            
    # Check for software presence when claim says 'raw unaltered live capture'
    software = metadata.get("Software")
    if software:
        conflicts.append(f"Provenance warning: File metadata contains editing tool signature '{software}'.")
        
    return conflicts

