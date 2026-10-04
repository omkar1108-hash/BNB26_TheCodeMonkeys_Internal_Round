"""Learned audio deepfake / spoof classifier (Hugging Face audio-classification, wav2vec2-style)."""
import wave
from math import gcd
from typing import Any, Dict, List, Optional
import numpy as np

from backend.learned import registry
from backend.config import DEVICE

TARGET_SR = 16000
WINDOW_S = 6.0
MAX_WINDOWS = 4


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    if sw == 1:
        x = (x - 128.0) / 128.0
    else:
        x = x / float(np.iinfo(dtype).max)
    return x, sr


def resample(x: np.ndarray, sr: int, target: int = TARGET_SR) -> np.ndarray:
    if sr == target or len(x) == 0:
        return x
    from scipy.signal import resample_poly
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def make_windows(x: np.ndarray, sr: int = TARGET_SR, window_s: float = WINDOW_S, max_windows: int = MAX_WINDOWS) -> List[np.ndarray]:
    """Up to ``max_windows`` evenly spaced windows. Short clips are returned whole."""
    n = int(window_s * sr)
    if len(x) <= n:
        return [x]
    starts = np.linspace(0, len(x) - n, num=min(max_windows, max(2, int(np.ceil(len(x) / n))))).astype(int)
    return [x[s:s + n] for s in starts]


def _load(model_id: str):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(model_id, local_files_only=True).to(DEVICE).eval()
    return {"fe": fe, "model": model, "fake_idx": registry.fake_index(model.config.id2label), "id": model_id}


def predict(wav_path: str) -> Optional[Dict[str, Any]]:
    """Returns {'p_fake', 'p_fake_mean', 'n_windows', 'model_id'} or None if unavailable."""
    bundle = registry.get_or_load("audio_spoof", _load)
    if bundle is None:
        return None
    try:
        import torch
        x, sr = read_wav_mono(wav_path)
        x = resample(x, sr)
        if len(x) < TARGET_SR // 2:
            return None  # <0.5 s: not enough signal to judge
        ps = []
        for w in make_windows(x):
            inputs = bundle["fe"](w, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                logits = bundle["model"](**inputs).logits
            ps.append(float(torch.softmax(logits, dim=-1)[0, bundle["fake_idx"]].cpu()))
        return {"p_fake": float(max(ps)), "p_fake_mean": float(np.mean(ps)), "n_windows": len(ps),
                "model_id": bundle["id"]}
    except Exception as e:
        registry._FAILED["audio_spoof"] = f"{type(e).__name__}: {e}"
        return None
