import json
import numpy as np
import pytest

from backend.models.bundle import BundleInput, ExtractedClaims
from backend.verdict.engine import analyze_bundle
from backend.claims.normalize import parse_date, locations_compatible, speakers_compatible
from backend.claims.conflict import find_bundle_contradictions
from backend.claims.evidence_graph import build_bundle_graph
from backend.cross_modal.metadata_rules import check_metadata_conflicts
from backend.fusion.classifier import classifier
from backend.fusion.calibration import conformal_qhat, TemperatureScaler, fit_temperature
from backend.fusion.abstention import evaluate_abstention, conformal_prediction_set
from evaluation.run_demos import run_all


# ---------- normalisation ----------
def test_dates_compare_by_calendar_day_not_string():
    assert parse_date("15 September 2026") == parse_date("2026-09-15") == parse_date("2026:09:15 10:00:00")
    assert parse_date("September 15, 2026") == parse_date("15/09/2026")


def test_location_and_speaker_matching():
    assert locations_compatible("Mumbai", "Aditya Institute, Mumbai")
    assert not locations_compatible("Mumbai", "Pune")
    assert speakers_compatible("Dr. Rajesh Sharma", "Rajesh Sharma")
    assert not speakers_compatible("Rajesh Sharma", "Anita Desai")


# ---------- cross-modal reasoning ----------
def test_metadata_date_and_location_conflicts_detected():
    meta = {"DateTimeOriginal": "2026:09:15 10:00:00", "Location": "Mumbai"}
    assert check_metadata_conflicts(meta, ["15 September 2026"], ["Mumbai"]) == []
    out = check_metadata_conflicts(meta, ["18 September 2026"], ["Pune"])
    assert any(m.startswith("Temporal conflict") for m in out)
    assert any(m.startswith("Location conflict") for m in out)


def test_cross_source_contradictions_and_graph():
    a = ExtractedClaims(dates=["15 September 2026"], locations=["Mumbai"])
    b = ExtractedClaims(dates=["18 September 2026"], locations=["Mumbai"])
    cons = find_bundle_contradictions({"message": a, "report": b})
    assert [c.field for c in cons] == ["date"]
    g = build_bundle_graph({"message": a, "report": b}, None, {}, None, cons)
    assert any(e["type"] == "conflict" for e in g["edges"])
    # the agreed location is one shared claim node with two incoming edges
    loc_nodes = [n for n in g["nodes"] if n["type"] == "claim" and n["field"] == "location"]
    assert len(loc_nodes) == 1


# ---------- model & calibration ----------
def test_trained_model_is_loaded():
    assert classifier.model is not None, "run `python -m evaluation.train` to create the XGBoost artifact"
    assert classifier.info()["type"] == "xgboost"


def test_conformal_threshold_gives_target_coverage():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4) * 0.5, size=2000)
    y = np.array([rng.choice(4, p=p) for p in probs])  # perfectly calibrated by construction
    q = conformal_qhat(probs[:1000], y[:1000], alpha=0.1)
    cover = np.mean([1 - probs[i, y[i]] <= q for i in range(1000, 2000)])
    assert cover >= 0.86


def test_temperature_fit_reduces_overconfidence():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, size=(1500, 4))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=np.exp(np.log(r) / 2.0) / np.exp(np.log(r) / 2.0).sum()) for r in p])  # labels are softer than p
    assert fit_temperature(p, y) > 1.3


def test_abstains_on_ambiguous_conformal_set_and_single_modality_authentic():
    probs = {"authentic": 0.5, "manipulated": 0.45, "coordinated_synthetic": 0.03, "insufficient_evidence": 0.02}
    ab, reason, label, _ = evaluate_abstention(probs, 3, conformal_set=["authentic", "manipulated"])
    assert ab and label == "insufficient_evidence" and "ambiguous" in reason
    ab, _, label, _ = evaluate_abstention({"authentic": 0.95, "manipulated": 0.02, "coordinated_synthetic": 0.02, "insufficient_evidence": 0.01}, 1)
    assert ab and label == "insufficient_evidence"


# ---------- end-to-end demos ----------
@pytest.fixture(scope="module")
def demo_results():
    return run_all()


@pytest.mark.parametrize("case", ["case1_authentic", "case2_manipulated", "case3_coordinated_fake", "case4_insufficient"])
def test_demo_verdicts(demo_results, case):
    expected, res = demo_results[case]
    assert res.label == expected


def test_coordinated_demo_has_evidence_graph_conflicts_and_counterfactuals(demo_results):
    _, res = demo_results["case3_coordinated_fake"]
    fields = {c.field for c in res.cross_modal_results.contradictions}
    assert {"date", "location", "speaker"} <= fields
    assert sum(e["type"] == "conflict" for e in res.evidence_graph["edges"]) >= 3
    assert res.counterfactuals, "multi-artifact bundles should get counterfactuals"
    # individually clean artifacts: no per-modality detector should be anomalous
    assert all(ev.score < 0.45 for ev in res.modality_results.values() if ev.present)


def test_insufficient_demo_says_what_would_settle_it(demo_results):
    _, res = demo_results["case4_insufficient"]
    assert res.abstained and res.next_steps


# ---------- API ----------
def test_api_accepts_multiple_documents():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    r = c.post("/api/bundle/analyze", json={"documents": {
        "message": "TechFest 2026 was held at Aditya Institute, Pune, on 15 September 2026.",
        "report": "TechFest 2026 was held at Aditya Institute, Mumbai, on 15 September 2026."}})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "coordinated_synthetic"
    assert body["model_info"]["type"] == "xgboost"


def test_api_response_matches_schema():
    import json
    from pathlib import Path
    from jsonschema import validate
    from fastapi.testclient import TestClient
    from backend.main import app

    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "analysis_schema.json"
    schema = json.loads(schema_path.read_text())

    c = TestClient(app)
    r = c.post("/api/bundle/analyze", json={
        "text": "TechFest was held in Mumbai on 15 September 2026.",
        "documents": {"doc1": "TechFest took place in Mumbai on 15 September 2026."}
    })
    assert r.status_code == 200
    validate(instance=r.json(), schema=schema)

