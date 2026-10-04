import pytest
from backend.models.bundle import BundleInput, VerdictResult
from backend.verdict.engine import analyze_bundle


def test_empty_bundle_abstention():
    bundle = BundleInput()
    res = analyze_bundle(bundle)
    assert isinstance(res, VerdictResult)
    assert res.abstained
    assert res.label == "insufficient_evidence"


def test_manipulated_bundle():
    bundle = BundleInput(
        text="A normal conference was held in Mumbai.",
        metadata={"Software": "Adobe Photoshop 2024 (Macintosh)"}
    )
    res = analyze_bundle(bundle)
    assert isinstance(res, VerdictResult)
    assert res.label in ["manipulated", "insufficient_evidence"]
    assert len(res.ranked_evidence) > 0
    assert res.report != ""
