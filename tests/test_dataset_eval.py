"""
Tests for evaluation/dataset_eval.py. They use tiny generated files and stub classifiers, so they check the
loading, sampling, metric and reporting code, NOT how good any real model is (that is what the script is for).
"""
import json
import wave
import numpy as np
import pytest
from PIL import Image

from backend.learned import registry
from evaluation import dataset_eval as de


# ---------------------------------------------------------------- helpers
def _img(path, kind, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "real":
        a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    else:
        a = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None], (64, 1, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, "JPEG", quality=92)


def _wav(path, freq=220.0, seconds=1.0, sr=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sr * seconds)) / sr
    pcm = (np.sin(2 * np.pi * freq * t) * 9000).astype("int16")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())


def _image_tree(root, n=6):
    for i in range(n):
        _img(root / "real" / f"r{i}.jpg", "real", i)
        _img(root / "fake" / "gen_a" / f"f{i}.jpg", "fake", i)


# ---------------------------------------------------------------- labels & loading
@pytest.mark.parametrize("value,expected", [
    ("0", 0), ("1", 1), ("real", 0), ("FAKE", 1), ("bona-fide", 0), ("bonafide", 0), ("spoof", 1),
    ("AI-generated", 1), ("human", 0), ("cat", None), ("real_vs_fake", None),
])
def test_parse_label(value, expected):
    assert de.parse_label(value) == expected


def test_load_folder_labels_groups_and_unlabelled(tmp_path):
    _image_tree(tmp_path, 3)
    _img(tmp_path / "misc" / "x.jpg", "real")
    samples, unlabelled = de.load_folder(tmp_path, de.IMAGE_EXT)
    assert unlabelled == 1
    assert sum(s.label == de.REAL for s in samples) == 3 and sum(s.label == de.FAKE for s in samples) == 3
    assert {s.group for s in samples if s.label == de.FAKE} == {"gen_a"}


def test_load_folder_name_overrides(tmp_path):
    _img(tmp_path / "RealArt" / "a.jpg", "real")
    _img(tmp_path / "AiArtData" / "b.jpg", "fake")
    samples, unlabelled = de.load_folder(tmp_path, de.IMAGE_EXT, real_dirs=["RealArt"], fake_dirs=["AiArtData"])
    assert unlabelled == 0 and {s.label for s in samples} == {0, 1}


def test_load_manifest(tmp_path):
    _img(tmp_path / "a.jpg", "real")
    (tmp_path / "m.csv").write_text("path,label,group,caption\na.jpg,real,,a cat\nmissing.jpg,fake,genX,\n,fake,,\n")
    samples, bad = de.load_manifest(tmp_path / "m.csv", None)
    assert bad == 1 and len(samples) == 2
    assert samples[0].caption == "a cat" and samples[1].group == "genX" and samples[1].label == de.FAKE


def test_load_asvspoof_2019_and_2021_layouts(tmp_path):
    for name in ("LA_T_1", "LA_E_2", "LA_E_3"):
        _wav(tmp_path / "audio" / f"{name}.wav")
    proto = tmp_path / "p.txt"
    proto.write_text(
        "LA_0079 LA_T_1 - - bonafide\n"
        "LA_0039 LA_E_2 - A11 spoof\n"
        "LA_0009 LA_E_3 alaw ita_tx A07 spoof notrim eval\n"
        "LA_0001 LA_E_404 - A01 spoof\n")
    samples, missing = de.load_asvspoof(proto, tmp_path / "audio")
    assert missing == 1 and len(samples) == 3
    by = {s.id: s for s in samples}
    assert by["LA_T_1"].label == 0 and by["LA_T_1"].group is None
    assert by["LA_E_2"].label == 1 and by["LA_E_2"].group == "A11" and by["LA_E_2"].condition is None
    assert by["LA_E_3"].group == "A07" and by["LA_E_3"].condition == "alaw"


def test_balanced_sample_is_balanced_and_deterministic():
    samples = [de.Sample(f"s{i}", f"p{i}", i % 3 == 0) for i in range(60)]
    a, b = de.balanced_sample(samples, 5, seed=1), de.balanced_sample(samples, 5, seed=1)
    assert [s.id for s in a] == [s.id for s in b]
    assert sum(s.label == 1 for s in a) == 5 and sum(s.label == 0 for s in a) == 5
    assert [s.id for s in de.balanced_sample(samples, 5, seed=2)] != [s.id for s in a]
    assert len(de.balanced_sample(samples, 0, seed=1)) == 60


# ---------------------------------------------------------------- metrics
def test_auroc_perfect_inverted_and_ties():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert de.auroc(y, np.array([.1, .2, .3, .7, .8, .9])) == 1.0
    assert de.auroc(y, np.array([.9, .8, .7, .3, .2, .1])) == 0.0
    assert de.auroc(y, np.full(6, .5)) == pytest.approx(0.5)
    assert np.isnan(de.auroc(np.zeros(4), np.arange(4)))


def test_ranking_metrics_and_operating_point():
    y = np.array([0] * 100 + [1] * 100)
    s = np.r_[np.linspace(0.0, 0.6, 100), np.linspace(0.4, 1.0, 100)]
    m = de.ranking_metrics(y, s, n_boot=50, seed=0)
    assert 0.8 < m["auroc"] < 1.0 and m["auroc_ci95"][0] <= m["auroc"] <= m["auroc_ci95"][1]
    assert 0.0 < m["eer"] < 0.3 and m["tpr_at_5pct_fpr"] >= m["tpr_at_1pct_fpr"]
    op = de.operating_point(y, s, 0.5)
    assert op["tp"] + op["fn"] == 100 and op["fp"] + op["tn"] == 100
    assert op["accuracy"] == pytest.approx((op["tp"] + op["tn"]) / 200)
    assert "note" in de.ranking_metrics(np.zeros(5), np.arange(5.0), 0, 0)


def test_calibration_split_is_stratified_and_excluded_from_reported_metrics():
    samples = [de.Sample(f"s{i}", "p", i % 2) for i in range(100)]
    calib = de.assign_calibration_ids(samples, 0.3, seed=0)
    assert len(calib) == 30 and sum(1 for s in samples if s.id in calib and s.label == 1) == 15
    rows = [{"id": s.id, "label": s.label, "group": None, "scores": {"x": 0.9 if s.label else 0.1}} for s in samples]
    m = de.evaluate_column(rows, "x", 0.5, calib, n_boot=0, seed=0)
    assert m["n"] == 70 and m["tuned_on_n"] == 30 and "at_tuned_threshold" in m
    assert de.evaluate_column(rows, "x", 0.5, set(), 0, 0)["n"] == 100


def test_failed_scores_are_counted_not_scored():
    rows = [{"id": f"s{i}", "label": i % 2, "group": None, "scores": {"x": None if i < 4 else float(i % 2)}} for i in range(20)]
    m = de.evaluate_column(rows, "x", 0.5, set(), 0, 0)
    assert m["n_failed_or_unavailable"] == 4 and m["n"] == 16
    assert "note" in de.evaluate_column(rows, "missing_col", 0.5, set(), 0, 0)


def test_group_breakdown():
    rows = [{"id": f"r{i}", "label": 0, "group": None, "scores": {"x": 0.1}} for i in range(10)]
    rows += [{"id": f"a{i}", "label": 1, "group": "A", "scores": {"x": 0.9}} for i in range(6)]
    rows += [{"id": f"b{i}", "label": 1, "group": "B", "scores": {"x": 0.05}} for i in range(6)]
    rows += [{"id": f"c{i}", "label": 1, "group": "tiny", "scores": {"x": 0.9}} for i in range(2)]
    g = de.group_breakdown(rows, "x", 0.5, set())
    assert set(g) == {"A", "B"}
    assert g["A"]["detection_rate_at_threshold"] == 1.0 and g["B"]["detection_rate_at_threshold"] == 0.0
    assert g["A"]["auroc_vs_reals"] == 1.0 and g["B"]["auroc_vs_reals"] == 0.0


# ---------------------------------------------------------------- perturbations
def test_perturbations(tmp_path):
    from backend.analyzers.image_detector import estimate_jpeg_quality
    _img(tmp_path / "a.jpg", "real")
    assert de.perturbation_ok("jpeg70") and de.perturbation_ok("none") and not de.perturbation_ok("jpeg0") and not de.perturbation_ok("blur")
    assert de.write_perturbed(str(tmp_path / "a.jpg"), "none", tmp_path) == str(tmp_path / "a.jpg")
    out = de.write_perturbed(str(tmp_path / "a.jpg"), "jpeg50", tmp_path)
    with Image.open(out) as im:
        assert 40 <= estimate_jpeg_quality(im) <= 60
    out = de.write_perturbed(str(tmp_path / "a.jpg"), "resize50_jpeg85", tmp_path)
    with Image.open(out) as im:
        assert im.size == (32, 32)


# ---------------------------------------------------------------- end to end (stub classifiers)
def _stub_image_model(monkeypatch):
    """Stub honours registry.enabled(), like the real predict: so the 'heuristic' column must not see it."""
    from backend.learned import image_deepfake
    registry.set_mode("auto")
    monkeypatch.setattr(registry, "available", lambda key: True)
    calls = {"n": 0}

    def predict(img):
        if not registry.enabled():
            return None
        calls["n"] += 1
        return {"p_fake": 0.97 if np.abs(np.diff(np.asarray(img.convert("L"), dtype=float), axis=0)).mean() < 5 else 0.03, "model_id": "stub/image"}
    monkeypatch.setattr(image_deepfake, "predict", predict)
    return calls


def test_image_cli_end_to_end(tmp_path, monkeypatch):
    _image_tree(tmp_path / "ds", 8)
    calls = _stub_image_model(monkeypatch)
    out = tmp_path / "out"
    rc = de.main(["image", "--root", str(tmp_path / "ds"), "--perturb", "none", "jpeg70", "--bootstrap", "20",
                  "--out-dir", str(out), "--name", "t"])
    assert rc == 0
    assert calls["n"] == 8 * 2 * 2   # 16 samples x 2 conditions, hit only by the learned-ON run (the heuristic run never reaches it)
    rep = json.loads((out / "dataset_eval_image_t.json").read_text())
    assert set(rep["results"]["conditions"]) == {"none", "jpeg70"}
    none = rep["results"]["conditions"]["none"]
    assert none["learned"]["auroc"] == 1.0                      # the stub separates the classes perfectly
    assert none["learned"]["n"] == 16 and none["heuristic"]["n"] == 16
    import csv
    with open(out / "dataset_eval_image_t_scores.csv") as f:
        for r in csv.DictReader(f):   # additive-only: the combined score can never be below the heuristic one
            assert float(r["combined"]) >= float(r["heuristic"]) - 1e-9
    assert (out / "dataset_eval_image_t.md").exists() and (out / "dataset_eval_image_t_scores.csv").exists()
    assert "gen_a" in rep["results"]["groups"]["learned"]


def test_heuristic_column_never_sees_the_learned_model(tmp_path, monkeypatch):
    _image_tree(tmp_path / "ds", 3)
    _stub_image_model(monkeypatch)
    samples, _ = de.load_folder(tmp_path / "ds", de.IMAGE_EXT)
    rows = de.run_detector_task("image", samples, ["none"], learned_on=True)
    with registry.disabled():
        from backend.analyzers.image_detector import analyze_image
        for r in rows:
            assert r["scores"]["heuristic"] == pytest.approx(analyze_image(r["path"]).score)
            assert r["scores"]["learned"] is not None


def test_learned_unavailable_gives_heuristic_only_and_a_warning(tmp_path, monkeypatch):
    _image_tree(tmp_path / "ds", 4)
    registry.set_mode("auto")
    monkeypatch.setattr(registry, "available", lambda key: False)
    out = tmp_path / "out"
    assert de.main(["image", "--root", str(tmp_path / "ds"), "--bootstrap", "0", "--out-dir", str(out), "--name", "h"]) == 0
    rep = json.loads((out / "dataset_eval_image_h.json").read_text())
    none = rep["results"]["conditions"]["none"]
    assert "auroc" in none["heuristic"] and "auroc" not in none["learned"] and "auroc" not in none["combined"]
    assert any("not available" in w for w in rep["warnings"])


def test_dry_run_scores_and_writes_nothing(tmp_path, capsys):
    _image_tree(tmp_path / "ds", 3)
    out = tmp_path / "out"
    assert de.main(["image", "--root", str(tmp_path / "ds"), "--dry-run", "--out-dir", str(out)]) == 0
    assert "3 real, 3 fake" in capsys.readouterr().out and not out.exists()


def test_audio_asvspoof_end_to_end(tmp_path, monkeypatch):
    from backend.learned import audio_spoof
    registry.set_mode("auto")
    monkeypatch.setattr(registry, "available", lambda key: True)
    monkeypatch.setattr(audio_spoof, "predict",
                        lambda path: {"p_fake": 0.9 if "_E_" in path else 0.1, "p_fake_mean": 0.5, "n_windows": 1, "model_id": "stub/audio"}
                        if registry.enabled() else None)
    lines = []
    for i in range(12):   # 6 spoofs per attack id, enough for the per-group breakdown (minimum 5)
        _wav(tmp_path / "audio" / f"LA_T_{i}.wav", 200 + 20 * i); lines.append(f"LA_00{i} LA_T_{i} - - bonafide")
        _wav(tmp_path / "audio" / f"LA_E_{i}.wav", 500 + 20 * i); lines.append(f"LA_01{i} LA_E_{i} - A1{i % 2} spoof")
    (tmp_path / "p.txt").write_text("\n".join(lines))
    out = tmp_path / "out"
    rc = de.main(["audio", "--root", str(tmp_path / "audio"), "--asvspoof-protocol", str(tmp_path / "p.txt"),
                  "--bootstrap", "0", "--out-dir", str(out), "--name", "asv"])
    assert rc == 0
    rep = json.loads((out / "dataset_eval_audio_asv.json").read_text())
    assert rep["results"]["conditions"]["all"]["learned"]["auroc"] == 1.0
    assert set(rep["results"]["groups"]["learned"]) == {"A10", "A11"}


def test_clip_task_reports_guard_rates_and_suggested_anchors(tmp_path, monkeypatch):
    from backend.learned import clip_scorer
    registry.set_mode("auto")
    monkeypatch.setattr(registry, "available", lambda key: True)
    rows = ["path,caption"]
    for i in range(30):
        _img(tmp_path / "im" / f"{i}.jpg", "real", i)
        rows.append(f"im/{i}.jpg,caption number {i}")
    (tmp_path / "caps.csv").write_text("\n".join(rows))
    # score_clip_pair asks for the matching caption first, then the mismatched one
    state, rng = {"k": 0}, np.random.default_rng(0)

    def cosine(img, texts):
        state["k"] += 1
        return float((0.30 if state["k"] % 2 else 0.18) + rng.normal(0, 0.002))
    monkeypatch.setattr(clip_scorer, "image_text_cosine", cosine)
    out = tmp_path / "out"
    rc = de.main(["clip", "--root", str(tmp_path), "--manifest", str(tmp_path / "caps.csv"), "--out-dir", str(out), "--name", "c"])
    assert rc == 0
    res = json.loads((out / "dataset_eval_clip_c.json").read_text())["results"]
    assert "pairs" not in res and res["n_pairs"] == 30
    assert res["guard_false_block_rate"] == 0.0 and res["guard_mismatch_catch_rate"] == 1.0
    assert res["suggested_anchors"]["COS_MISMATCH"] < res["suggested_anchors"]["COS_MATCH"]


def test_clip_score_is_the_mapped_raw_cosine(monkeypatch):
    from backend.learned import clip_scorer
    monkeypatch.setattr(clip_scorer, "image_text_cosine", lambda img, texts: 0.225)
    assert clip_scorer.image_text_score(None, ["x"]) == pytest.approx(0.5)
    monkeypatch.setattr(clip_scorer, "image_text_cosine", lambda img, texts: None)
    assert clip_scorer.image_text_score(None, ["x"]) is None
    assert clip_scorer.image_text_cosine(None, []) is None   # no text: nothing to score (no model needed)


def test_plain_wav_is_passed_through_unchanged(tmp_path):
    _wav(tmp_path / "a.wav")
    assert de.to_pcm16_wav(str(tmp_path / "a.wav"), tmp_path) == str(tmp_path / "a.wav")


def test_flac_is_decoded_to_a_readable_wav(tmp_path):
    sf = pytest.importorskip("soundfile")
    t = np.arange(16000) / 16000.0
    sf.write(str(tmp_path / "a.flac"), (np.sin(2 * np.pi * 300 * t) * 0.3).astype("float32"), 16000)
    out = de.to_pcm16_wav(str(tmp_path / "a.flac"), tmp_path)
    assert out != str(tmp_path / "a.flac") and out.endswith(".wav")
    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getnframes() == 16000

