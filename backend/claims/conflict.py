def normalize(value):
    if value is None:
        return None

    return " ".join(str(value).lower().strip().split())


def compare_claims(claim_a, claim_b):
    """
    Compare only meaningful factual fields.

    Claims are compared only when they refer to the same topic/event.
    Missing fields are ignored.
    """

    conflicts = []

    # We currently trust structured fields more than the raw sentence.
    # "what" is intentionally not compared because simple text matching
    # incorrectly treats paraphrases as contradictions.

    fields = ["who", "when", "where"]

    for field in fields:
        value_a = normalize(claim_a.get(field))
        value_b = normalize(claim_b.get(field))

        if value_a and value_b and value_a != value_b:
            conflicts.append({
                "claim_a": claim_a["claim_id"],
                "claim_b": claim_b["claim_id"],
                "field": field,
                "value_a": claim_a.get(field),
                "value_b": claim_b.get(field),
                "description": (
                    f"{field.upper()} conflict: "
                    f"'{claim_a.get(field)}' vs '{claim_b.get(field)}'"
                ),
                "severity": "high"
            })

    return conflicts


def find_conflicts(claims):
    """
    Compare claims from different files only.

    Claims from the same file should never conflict with each other.
    """

    conflicts = []

    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):

            claim_a = claims[i]
            claim_b = claims[j]

            # Never compare claims belonging to the same file.
            if claim_a.get("file_id") == claim_b.get("file_id"):
                continue

            conflicts.extend(compare_claims(claim_a, claim_b))

    return conflicts