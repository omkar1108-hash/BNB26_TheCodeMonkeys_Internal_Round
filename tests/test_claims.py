import pytest
from backend.claims.extractor import extract_claims
from backend.claims.conflict import find_bundle_contradictions, find_conflicts
from backend.claims.evidence_graph import build_evidence_graph
from backend.models.bundle import ExtractedClaims


def test_claim_extraction():
    sample = "The summit was organized by Ministry of Education in Mumbai on 15 September 2026."
    claims = extract_claims(sample)
    assert isinstance(claims, ExtractedClaims)
    assert len(claims.dates) >= 1
    assert "15 September 2026" in claims.dates[0]


def test_bundle_contradictions():
    claims_a = ExtractedClaims(
        dates=["15 September 2026"],
        locations=["Mumbai"],
        entities=["Govt of India"]
    )
    claims_b = ExtractedClaims(
        dates=["18 September 2026"],
        locations=["Pune"],
        entities=["Govt of India"]
    )
    
    contradictions = find_bundle_contradictions({
        "source_a": claims_a,
        "source_b": claims_b
    })
    
    assert len(contradictions) == 2  # Date and Location conflicts
    fields = {c.field for c in contradictions}
    assert "date" in fields
    assert "location" in fields


def test_evidence_graph_building():
    files = [{"file_id": "f1", "filename": "sample.jpg", "modality": "image"}]
    claims = [{"claim_id": "c1", "file_id": "f1", "text": "Event happened"}]
    conflicts = []
    
    graph = build_evidence_graph(files, claims, conflicts)
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
