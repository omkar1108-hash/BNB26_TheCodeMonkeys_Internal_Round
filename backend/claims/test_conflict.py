from conflict import find_conflicts


image_claims = [
    {
        "claim_id": "image_claim_1",
        "text": "The event happened in Mumbai on 15 September 2026.",
        "who": "Government of India",
        "what": "event happened",
        "when": "15 September 2026",
        "where": "Mumbai"
    }
]


message_claims = [
    {
        "claim_id": "message_claim_1",
        "text": "The event happened in Pune on 15 September 2026.",
        "who": "Government of India",
        "what": "event happened",
        "when": "15 September 2026",
        "where": "Pune"
    }
]


document_claims = [
    {
        "claim_id": "document_claim_1",
        "text": "The event happened in Mumbai on 18 September 2026.",
        "who": "Government of India",
        "what": "event happened",
        "when": "18 September 2026",
        "where": "Mumbai"
    }
]


all_claims = [
    image_claims,
    message_claims,
    document_claims
]


conflicts = find_conflicts(all_claims)


print("=" * 60)
print("DETECTED CONFLICTS")
print("=" * 60)


for conflict in conflicts:

    print()
    print("Field:", conflict["field"])
    print("Claim A:", conflict["claim_a"])
    print("Claim B:", conflict["claim_b"])
    print("Value A:", conflict["value_a"])
    print("Value B:", conflict["value_b"])
    print("Description:", conflict["description"])
    print("Severity:", conflict["severity"])