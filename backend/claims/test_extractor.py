from extractor import extract_claims


sample_text = """
The Government of India announced a new digital education scheme
on 15 September 2026. The scheme will provide free digital learning
resources to students across Mumbai.
"""


claims = extract_claims(sample_text)


for claim in claims:

    print("=" * 60)

    print("CLAIM ID:", claim["claim_id"])
    print("TEXT:", claim["text"])
    print("WHO:", claim["who"])
    print("WHAT:", claim["what"])
    print("WHEN:", claim["when"])
    print("WHERE:", claim["where"])