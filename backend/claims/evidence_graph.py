from typing import List, Dict


def build_evidence_graph(
    files: List[Dict],
    claims: List[Dict],
    conflicts: List[Dict]
) -> Dict:
    """
    Build a graph representation of files, claims and conflicts.

    Nodes:
        - file
        - claim

    Edges:
        - contains
        - conflicts
    """

    nodes = []
    edges = []

    # -------------------------
    # File nodes
    # -------------------------

    for file in files:

        nodes.append({
            "id": file["file_id"],
            "type": "file",
            "label": file["filename"],
            "modality": file.get("modality")
        })

    # -------------------------
    # Claim nodes
    # -------------------------

    for claim in claims:

        nodes.append({
            "id": claim["claim_id"],
            "type": "claim",
            "label": claim["text"],
            "when": claim.get("when"),
            "where": claim.get("where"),
            "who": claim.get("who")
        })

    # -------------------------
    # File → Claim edges
    # -------------------------

    for claim in claims:

        file_id = claim.get("file_id")

        if file_id:

            edges.append({
                "source": file_id,
                "target": claim["claim_id"],
                "type": "contains"
            })

    # -------------------------
    # Conflict edges
    # -------------------------

    for conflict in conflicts:

        edges.append({
            "source": conflict["claim_a"],
            "target": conflict["claim_b"],
            "type": "conflict",
            "field": conflict.get("field"),
            "severity": conflict.get("severity", "medium"),
            "description": conflict.get("description")
        })

    return {
        "nodes": nodes,
        "edges": edges
    }