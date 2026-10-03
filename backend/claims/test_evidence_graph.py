from evidence_graph import build_evidence_graph


files = [
    {
        "file_id": "file_001",
        "filename": "event_image.jpg",
        "modality": "image"
    },
    {
        "file_id": "file_002",
        "filename": "message.txt",
        "modality": "text"
    },
    {
        "file_id": "file_003",
        "filename": "event_document.pdf",
        "modality": "document"
    }
]


claims = [
    {
        "claim_id": "claim_image",
        "file_id": "file_001",
        "text": "Event happened in Mumbai.",
        "who": "Government of India",
        "what": "event happened",
        "when": "15 September 2026",
        "where": "Mumbai"
    },
    {
        "claim_id": "claim_message",
        "file_id": "file_002",
        "text": "Event happened in Pune.",
        "who": "Government of India",
        "what": "event happened",
        "when": "15 September 2026",
        "where": "Pune"
    }
]


conflicts = [
    {
        "claim_a": "claim_image",
        "claim_b": "claim_message",
        "field": "where",
        "value_a": "Mumbai",
        "value_b": "Pune",
        "description": "WHERE conflict: Mumbai vs Pune",
        "severity": "high"
    }
]


graph = build_evidence_graph(
    files,
    claims,
    conflicts
)


print("=" * 60)
print("EVIDENCE GRAPH")
print("=" * 60)

print("\nNODES:")

for node in graph["nodes"]:
    print(node)


print("\nEDGES:")

for edge in graph["edges"]:
    print(edge)
