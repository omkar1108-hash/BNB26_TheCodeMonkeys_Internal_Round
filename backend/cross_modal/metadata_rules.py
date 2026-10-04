from typing import List, Dict, Any, Optional

from backend.claims.normalize import parse_date, locations_compatible

LOCATION_KEYS = ("Location", "City", "GPSCity", "GPSLocation", "Place")


def get_metadata_date(metadata: Optional[Dict[str, Any]]):
    if not metadata:
        return None
    raw = metadata.get("DateTimeOriginal") or metadata.get("DateTime")
    return parse_date(raw) if raw else None


def get_metadata_location(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    for key in LOCATION_KEYS:
        if metadata.get(key):
            return str(metadata[key])
    return None


def check_metadata_conflicts(
    metadata: Optional[Dict[str, Any]],
    extracted_dates: List[str],
    extracted_locations: List[str],
    source_label: str = "text",
) -> List[str]:
    """
    Cross-modal contradictions between capture metadata (EXIF / sidecar) and the
    claims made in a text source: the *calendar date* and the *place*.

    Editing-software signatures are a single-artifact signal and are handled by
    the metadata detector, so they are deliberately not reported here.
    """
    conflicts: List[str] = []
    if not metadata:
        return conflicts

    exif_date = get_metadata_date(metadata)
    if exif_date and extracted_dates:
        claimed = {d for d in (parse_date(x) for x in extracted_dates) if d}
        if claimed and exif_date not in claimed:
            shown = ", ".join(sorted(d.isoformat() for d in claimed))
            conflicts.append(
                f"Temporal conflict: capture metadata date ({exif_date.isoformat()}) "
                f"contradicts the date claimed in {source_label} ({shown})."
            )

    meta_loc = get_metadata_location(metadata)
    if meta_loc and extracted_locations:
        if not any(locations_compatible(meta_loc, loc) for loc in extracted_locations):
            conflicts.append(
                f"Location conflict: capture metadata location ('{meta_loc}') "
                f"contradicts the location claimed in {source_label} ('{extracted_locations[0]}')."
            )

    return conflicts
