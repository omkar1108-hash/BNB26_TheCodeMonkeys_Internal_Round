"""
The learned detectors need downloaded weights, so these tests exercise everything AROUND the
networks (label resolution, score mapping, windowing/resampling, fusion into detector scores,
authenticity guard, download selection) with stub predictors. Real-weight behaviour is verified
with `python -m backend.learned.download --check`.
"""
import wave
import numpy as np
import pytest
from PIL import Image

from backend.learned import registry, audio_spoof, clip_scorer
from backend.learned.download import choose_patterns
from backend.models.bundle import BundleInput


# ---------- label resolution ----------
@pytest.mark.parametrize("labels,expected", [
    ({0: "ai", 1: "hum"}, 0),
    ({0: "real", 1: "fake"}, 1),
    ({0: "Realism", 1: "Deepfake"}, 1),
    ({0: "bonafide", 1: "spoof"}, 1),
    ({0: "human", 1: "AI-generated"}, 1),
    ({0: "artificial", 1: "human"}, 0),
])
def test_fake_index_from_label_names(labels, expected):
    assert registry.fake_index(labels) == expected


def test_fake_index_refuses_to_guess():
    with pytest.raises(ValueError):
        registry.fake_index({0: "cat", 1: "dog"})
    with pytest.raises(ValueError):
        registry.fake_index({0: "fake", 1: "spoof"})  # two fake classes: ambiguous


# ---------- score mapping ----------
def test_probability_maps_to_anomaly_scale_without_clearing_evidence():
    f = registry.prob_to_anomaly
    assert f(0.1, 0.28, 0.67) == f(0.5, 0.28, 0.67) == pytest.approx(0.28)  # no evidence below chance
    assert f(1.0, 0.28, 0.67) == pytest.approx(0.95)
    assert f(0.75, 0.15, 0.80) == pytest.approx(0.55)


def test_clip_cosine_mapping():
    assert clip_scorer.cosine_to_score(0.10) == 0.0
    assert clip_scorer.cosine_to_score(0.35) == 1.0
    assert clip_scorer.cosine_to_score(0.225) == pytest.approx(0.5)


# ---------- audio preprocessing ----------
def _write_wav(path, sr, seconds, freq=220.0, ch=1):
    t = np.arange(int(sr * seconds)) / sr
    pcm = (np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16)
    if ch == 2:
        pcm = np.repeat(pcm, 2)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())


def test_wav_is_read_mixed_to_mono_and_resampled(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, 44100, 1.0, ch=2)
    x, sr = audio_spoof.read_wav_mono(str(p))
    assert sr == 44100 and x.ndim == 1 and len(x) == 44100 and abs(x).max() <= 1.0
    y = audio_spoof.resample(x, sr)
    assert len(y) == pytest.approx(16000, abs=2)


def test_windows_cover_long_clip_and_keep_short_clip_whole():
    short = np.zeros(16000 * 3, dtype=np.float32)
    assert len(audio_spoof.make_windows(short)) == 1
    long = np.arange(16000 * 30, dtype=np.float32)
    wins = audio_spoof.make_windows(long)
    assert 2 <= len(wins) <= audio_spoof.MAX_WINDOWS
    assert all(len(w) == int(audio_spoof.WINDOW_S * 16000) for w in wins)
    assert wins[0][0] == 0 and wins[-1][-1] == long[-1]  # spans start to end


# ---------- fusion into detector scores (stub predictors) ----------
def _jpeg(path, seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(90, 160, (256, 256, 3), dtype=np.uint8)).save(path, "JPEG", quality=92)


def test_learned_image_probability_raises_score_and_is_reported(tmp_path, monkeypatch):
    from backend.analyzers.image_detector import analyze_image
    from backend.learned import image_deepfake
    p = tmp_path / "i.jpg"; _jpeg(p)
    base = analyze_image(str(p))
    monkeypatch.setattr(image_deepfake, "predict", lambda img: {"p_fake": 0.97, "model_id": "stub/model"})
    ev = analyze_image(str(p))
    assert ev.score > base.score and ev.score >= 0.90
    assert "stub/model" in ev.evidence and "97%" in ev.evidence
    assert ev.features["learned_fake_prob"] == 0.97 and not ev.is_fallback


def test_learned_model_never_lowers_a_heuristic_anomaly(tmp_path, monkeypatch):
    from backend.analyzers.image_detector import analyze_image
    from backend.learned import image_deepfake
    p = tmp_path / "i.jpg"; _jpeg(p)
    base = analyze_image(str(p))
    monkeypatch.setattr(image_deepfake, "predict", lambda img: {"p_fake": 0.01, "model_id": "stub/model"})
    assert analyze_image(str(p)).score == pytest.approx(base.score)


def test_learned_audio_probability_raises_score(tmp_path, monkeypatch):
    from backend.analyzers.audio_detector import analyze_audio
    p = tmp_path / "a.wav"; _write_wav(p, 16000, 2.0)
    base = analyze_audio(str(p))
    monkeypatch.setattr(audio_spoof, "predict", lambda path: {"p_fake": 0.95, "p_fake_mean": 0.9, "n_windows": 2, "model_id": "stub/audio"})
    ev = analyze_audio(str(p))
    assert ev.score > base.score and "stub/audio" in ev.evidence and ev.features["learned_fake_prob"] == 0.95


def test_everything_still_works_with_learned_models_off(tmp_path):
    from backend.analyzers.image_detector import analyze_image
    p = tmp_path / "i.jpg"; _jpeg(p)
    ev = analyze_image(str(p))
    assert "learned_fake_prob" not in ev.features and not ev.is_fallback


# ---------- CLIP authenticity guard (stub similarity) ----------
def _authentic_bundle(tmp_path):
    from data.generator.bundle_generator import generate_synthetic_image, generate_synthetic_audio
    img, aud = tmp_path / "photo.jpg", tmp_path / "speech.wav"
    rng = np.random.default_rng(3)
    generate_synthetic_image(img, None, rng); generate_synthetic_audio(aud, None, rng)
    docs = {"message": "TechFest 2026 was held at Aditya Institute, Mumbai, on 15 September 2026. Addressed by Dr. Rajesh Sharma.",
            "report": "TechFest 2026 took place at Aditya Institute, Mumbai, on 15 September 2026."}
    meta = {"Make": "Sony", "Model": "A7", "DateTimeOriginal": "2026:09:15 10:00:00", "Location": "Mumbai"}
    return BundleInput(image_path=str(img), audio_path=str(aud), documents=docs, metadata=meta, claimed_speaker="Dr. Rajesh Sharma")


def test_low_clip_agreement_blocks_an_authentic_verdict(tmp_path, monkeypatch):
    from backend.verdict import engine
    b = _authentic_bundle(tmp_path)
    assert engine.analyze_bundle(b).label == "authentic"
    monkeypatch.setattr(engine, "compute_image_text_similarity", lambda img, texts: (0.10, False, None))
    res = engine.analyze_bundle(b)
    assert res.label == "insufficient_evidence" and res.abstained and "CLIP" in res.abstain_reason
    assert res.model_info["clip_used"] is True


def test_clip_score_does_not_change_the_fusion_features(tmp_path, monkeypatch):
    """The model is trained without a CLIP signal, so the score must not leak into its input."""
    from backend.verdict import engine
    from backend.fusion.feature_builder import build_fusion_features
    b = _authentic_bundle(tmp_path)
    sig = engine.compute_bundle_signals(b)
    a = build_fusion_features(sig["modality_results"], sig["cross_modal"])
    cm = sig["cross_modal"].model_copy(update={"image_text_similarity": 0.05})
    assert np.array_equal(a, build_fusion_features(sig["modality_results"], cm))


def test_high_clip_agreement_does_not_hurt_authentic(tmp_path, monkeypatch):
    from backend.verdict import engine
    monkeypatch.setattr(engine, "compute_image_text_similarity", lambda img, texts: (0.9, False, None))
    assert engine.analyze_bundle(_authentic_bundle(tmp_path)).label == "authentic"


# ---------- registry / download ----------
def test_registry_reports_uncached_models_and_respects_off(monkeypatch):
    registry.set_mode(None)
    monkeypatch.setattr(registry, "is_cached", lambda mid: False)
    st = registry.status()
    assert set(st) == {"image_deepfake", "audio_spoof", "clip"} and not any(v["cached"] for v in st.values())
    assert registry.get_or_load("clip", lambda mid: object()) is None  # not cached -> no load attempt
    monkeypatch.setattr(registry, "is_cached", lambda mid: True)
    assert registry.get_or_load("clip", lambda mid: "loaded") == "loaded"
    with registry.disabled():
        assert registry.get_or_load("audio_spoof", lambda mid: "x") is None


def test_failed_load_is_remembered_and_never_raises(monkeypatch):
    registry.set_mode(None)
    monkeypatch.setattr(registry, "is_cached", lambda mid: True)
    calls = []
    def boom(mid):
        calls.append(1); raise RuntimeError("bad weights")
    assert registry.get_or_load("image_deepfake", boom) is None
    assert registry.get_or_load("image_deepfake", boom) is None
    assert len(calls) == 1 and "bad weights" in registry.status()["image_deepfake"]["last_error"]


def test_download_prefers_safetensors_and_skips_other_formats():
    p = choose_patterns(["config.json", "model.safetensors", "pytorch_model.bin", "tf_model.h5"])
    assert "*.safetensors" in p["allow"] and "*.bin" in p["ignore"]
    q = choose_patterns(["config.json", "pytorch_model.bin"])
    assert "*.bin" in q["allow"] and "*.bin" not in q["ignore"]


def test_models_status_endpoint():
    from fastapi.testclient import TestClient
    from backend.main import app
    r = TestClient(app).get("/api/models/status")
    assert r.status_code == 200 and set(r.json()["models"]) == {"image_deepfake", "audio_spoof", "clip"}
