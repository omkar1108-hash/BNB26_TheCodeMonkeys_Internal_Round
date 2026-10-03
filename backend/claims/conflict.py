from typing import List, Dict, Any, Optional
from backend.models.bundle import ContradictionItem, ExtractedClaims


def normalize(val: Any) -> Optional[str]:
    if val is None:
        return None
    return " ".join(str(val).lower().strip().split())


def detect_field_contradictions(
    field_name: str,
    items_a: List[str],
    items_b: List[str],
    source_a: str,
    source_b: str
) -> List[ContradictionItem]:
    """
    Plain Python contradiction detection between two lists of factual values.
    """
    contradictions = []
    norm_a = [normalize(x) for x in items_a if x]
    norm_b = [normalize(x) for x in items_b if x]

    # If both sources assert at least one distinct, non-overlapping factual value
    if norm_a and norm_b:
        # Check intersection
        common = set(norm_a).intersection(set(norm_b))
        if not common:
            val_a = items_a[0]
            val_b = items_b[0]
            contradictions.append(ContradictionItem(
                field=field_name,
                value_a=val_a,
                value_b=val_b,
                source_a=source_a,
                source_b=source_b,
                description=f"{field_name.upper()} conflict between {source_a} ('{val_a}') and {source_b} ('{val_b}').",
                severity="high"
            ))
    return contradictions


def find_bundle_contradictions(
    claims_map: Dict[str, ExtractedClaims]
) -> List[ContradictionItem]:
    """
    Compares extracted claims across bundle sources (e.g. 'caption', 'document', 'metadata', 'audio').
    Uses pure deterministic Python logic over short fields.
    """
    contradictions = []
    sources = list(claims_map.keys())

    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            src_a = sources[i]
            src_b = sources[j]
            c_a = claims_map[src_a]
            c_b = claims_map[src_b]

            # 1. Dates
            contradictions.extend(detect_field_contradictions(
                "date", c_a.dates, c_b.dates, src_a, src_b
            ))

            # 2. Locations
            contradictions.extend(detect_field_contradictions(
                "location", c_a.locations, c_b.locations, src_a, src_b
            ))

            # 3. Entities
            contradictions.extend(detect_field_contradictions(
                "entity", c_a.entities, c_b.entities, src_a, src_b
            ))

            # 4. Claimed Speaker
            if c_a.claimed_speaker and c_b.claimed_speaker:
                if normalize(c_a.claimed_speaker) != normalize(c_b.claimed_speaker):
                    contradictions.append(ContradictionItem(
                        field="speaker",
                        value_a=c_a.claimed_speaker,
                        value_b=c_b.claimed_speaker,
                        source_a=src_a,
                        source_b=src_b,
                        description=f"SPEAKER conflict: '{c_a.claimed_speaker}' vs '{c_b.claimed_speaker}'.",
                        severity="high"
                    ))

    return contradictions


# Keep backwards compatibility for legacy list-of-dicts claims
def find_conflicts(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    conflicts = []
    fields = ["who", "when", "where"]
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            ca = claims[i]
            cb = claims[j]
            if ca.get("file_id") == cb.get("file_id"):
                continue
            for field in fields:
                va = normalize(ca.get(field))
                vb = normalize(cb.get(field))
                if va and vb and va != vb:
                    conflicts.append({
                        "claim_a": ca.get("claim_id", f"claim_{i}"),
                        "claim_b": cb.get("claim_id", f"claim_{j}"),
                        "field": field,
                        "value_a": ca.get(field),
                        "value_b": cb.get(field),
                        "description": f"{field.upper()} conflict: '{ca.get(field)}' vs '{cb.get(field)}'",
                        "severity": "high"
                    })
    return conflicts
    return conflicts