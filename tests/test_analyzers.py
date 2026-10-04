import pytest
from backend.analyzers.file_detector import detect_file_type
from backend.analyzers.image_detector import analyze_image
from backend.analyzers.audio_detector import analyze_audio
from backend.analyzers.text_detector import analyze_text
from backend.analyzers.metadata_detector import analyze_metadata


def test_file_detector():
    assert detect_file_type("photo.jpg") == "image"
    assert detect_file_type("audio.wav") == "audio"
    assert detect_file_type("speech.mp3") == "audio"
    assert detect_file_type("document.pdf") == "document"
    assert detect_file_type("article.txt") == "text"
    assert detect_file_type("meta.json") == "metadata"
    assert detect_file_type("unknown.xyz") == "unknown"


def test_missing_modality_analyzers():
    img_ev = analyze_image(None)
    assert not img_ev.present
    assert img_ev.score == 0.0

    aud_ev = analyze_audio(None)
    assert not aud_ev.present
    assert aud_ev.score == 0.0

    txt_ev = analyze_text(None)
    assert not txt_ev.present
    assert txt_ev.score == 0.0

    meta_ev = analyze_metadata(None, None)
    assert not meta_ev.present
    assert meta_ev.score == 0.40


def test_text_analyzer_content():
    sample_text = "TechFest 2026 was organized by the Computer Department at Mumbai on 15 September 2026."
    res = analyze_text(sample_text)
    assert res.present
    assert 0.0 <= res.score <= 1.0
    assert "type_token_ratio" in res.features
