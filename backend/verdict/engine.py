def generate_verdict(files, claims, conflicts):
    """
    Generate a preliminary TrustLayer verdict from
    cross-file evidence.

    This is a baseline rule-based engine.
    External web evidence and provenance signals
    will be added later.
    """

    file_count = len(files)
    conflict_count = len(conflicts)

    # ---------------------------------------------------------
    # CASE 1: NOT ENOUGH EVIDENCE
    # ---------------------------------------------------------

    if file_count == 0 or len(claims) == 0:
        return {
            "label": "Not enough evidence",
            "confidence": 0.20,
            "reason": "Insufficient claims or source files were available."
        }

    # ---------------------------------------------------------
    # CASE 2: AUTHENTIC / CONSISTENT
    # ---------------------------------------------------------

    if conflict_count == 0:

        if file_count >= 2:
            return {
                "label": "Authentic",
                "confidence": 0.85,
                "reason": (
                    "The analyzed sources contain consistent structured "
                    "claims with no detected cross-file conflicts."
                )
            }

        return {
            "label": "Not enough evidence",
            "confidence": 0.40,
            "reason": (
                "Only one source is available, so cross-source "
                "consistency cannot be established."
            )
        }

    # ---------------------------------------------------------
    # COUNT DISTINCT CONFLICT TYPES
    # ---------------------------------------------------------

    conflict_fields = set()

    for conflict in conflicts:
        field = conflict.get("field")

        if field:
            conflict_fields.add(field)

    # ---------------------------------------------------------
    # CASE 3: COORDINATED FAKE
    # ---------------------------------------------------------

    if file_count >= 3 and len(conflict_fields) >= 2:
        return {
            "label": "Coordinated fake",
            "confidence": 0.90,
            "reason": (
                "Multiple sources contain significant contradictions "
                "across multiple factual fields."
            )
        }

    # ---------------------------------------------------------
    # CASE 4: MANIPULATED
    # ---------------------------------------------------------

    return {
        "label": "Manipulated",
        "confidence": 0.75,
        "reason": (
            "At least one significant cross-source contradiction "
            "was detected."
        )
    }