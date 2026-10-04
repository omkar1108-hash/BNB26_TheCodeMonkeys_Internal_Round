"""
Registry for the optional *learned* models.

Design rules
  * Heuristic detectors always run. Learned models ADD evidence when their weights are already
    cached locally; they never replace the heuristics and never download anything at request time.
  * Download once with ``python -m backend.learned.download``; runtime loads with local_files_only.
  * ``TRUSTLAYER_LEARNED=off`` disables them entirely. ``disabled()`` does so for a block of code
    (used for the synthetic demo bundles and the benchmark, which are not real photos / voices).
  * Model ids can be overridden with TRUSTLAYER_IMAGE_MODEL / TRUSTLAYER_AUDIO_MODEL / TRUSTLAYER_CLIP_MODEL
    (a Hugging Face repo id or a local directory).
"""
import contextlib
import contextvars
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

MODELS: Dict[str, Dict[str, str]] = {
    "image_deepfake": {
        "env": "TRUSTLAYER_IMAGE_MODEL",
        "default": "Ateeqq/ai-vs-human-image-detector",
        "what": "AI-generated vs. human image classifier",
    },
    "audio_spoof": {
        "env": "TRUSTLAYER_AUDIO_MODEL",
        "default": "MelodyMachine/Deepfake-audio-detection-V2",
        "what": "wav2vec2 audio deepfake classifier",
    },
    "clip": {
        "env": "TRUSTLAYER_CLIP_MODEL",
        "default": "openai/clip-vit-base-patch32",
        "what": "CLIP image-text similarity",
    },
}

FAKE_TOKENS = {"fake", "ai", "artificial", "synthetic", "generated", "deepfake", "spoof", "spoofed", "aigenerated", "gan"}
REAL_TOKENS = {"real", "human", "hum", "authentic", "bonafide", "genuine", "realism", "natural", "original", "live"}

_FORCED: contextvars.ContextVar = contextvars.ContextVar("trustlayer_learned_forced", default=None)
_DEFAULT_OVERRIDE: Optional[str] = None
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_FAILED: Dict[str, str] = {}


# ---------------------------------------------------------------- mode
def mode() -> str:
    forced = _FORCED.get()
    if forced:
        return forced
    if _DEFAULT_OVERRIDE:
        return _DEFAULT_OVERRIDE
    return os.getenv("TRUSTLAYER_LEARNED", "auto").strip().lower()


def enabled() -> bool:
    return mode() not in ("off", "0", "false", "no")


def set_mode(value: Optional[str]) -> None:
    """Process-wide override (used by tests). None restores the environment setting."""
    global _DEFAULT_OVERRIDE
    _DEFAULT_OVERRIDE = value


@contextlib.contextmanager
def disabled():
    """Turn learned models off inside the block (benchmark / synthetic demos)."""
    token = _FORCED.set("off")
    try:
        yield
    finally:
        _FORCED.reset(token)


# ---------------------------------------------------------------- ids & cache
def model_id(key: str) -> str:
    spec = MODELS[key]
    return os.getenv(spec["env"], spec["default"])


def is_cached(repo_or_path: str) -> bool:
    """True if the model's config is available locally (HF cache or a local directory)."""
    p = Path(repo_or_path)
    if p.exists() and (p / "config.json").exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo_or_path, "config.json")
        return isinstance(hit, (str, os.PathLike))
    except Exception:
        return False


def status() -> Dict[str, Dict[str, Any]]:
    out = {}
    for key, spec in MODELS.items():
        mid = model_id(key)
        out[key] = {
            "model_id": mid,
            "what": spec["what"],
            "cached": is_cached(mid),
            "loaded": key in _CACHE,
            "last_error": _FAILED.get(key),
        }
    return out


def available(key: str) -> bool:
    """Learned model is switched on AND its weights are cached."""
    return enabled() and is_cached(model_id(key)) and key not in _FAILED


def get_or_load(key: str, factory: Callable[[str], Any]) -> Optional[Any]:
    """Load once (thread-safe) and cache. Returns None if unavailable or if loading failed."""
    if not available(key):
        return None
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        try:
            _CACHE[key] = factory(model_id(key))
        except Exception as e:  # never break an analysis because a model failed to load
            _FAILED[key] = f"{type(e).__name__}: {e}"
            return None
        return _CACHE[key]


def reset() -> None:
    with _LOCK:
        _CACHE.clear()
        _FAILED.clear()


# ---------------------------------------------------------------- label mapping (pure)
def _tokens(label: str) -> set:
    flat = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).split()
    toks = set(flat)
    toks.add("".join(flat))  # 'ai-generated' -> 'aigenerated'
    return toks


def fake_index(id2label: Dict[Any, str]) -> int:
    """
    Index of the 'fake / AI-generated / spoof' class, resolved from the model's own label names.
    Raises ValueError rather than guess when the labels are not recognisable.
    """
    fake, real = [], []
    for idx, label in id2label.items():
        t = _tokens(label)
        is_fake = bool(t & FAKE_TOKENS)
        is_real = bool(t & REAL_TOKENS)
        if is_fake and not is_real:
            fake.append(int(idx))
        elif is_real and not is_fake:
            real.append(int(idx))
    if len(fake) == 1:
        return fake[0]
    if not fake and len(real) == 1 and len(id2label) == 2:
        return next(int(i) for i in id2label if int(i) != real[0])
    raise ValueError(f"Cannot tell which class means fake/spoof from labels {dict(id2label)}")


def prob_to_anomaly(p_fake: float, clean_baseline: float, span: float) -> float:
    """
    Map a classifier's P(fake) onto the detector's anomaly scale.
    p <= 0.5 carries no evidence of manipulation, so it maps to the clean baseline (a learned
    model never *clears* a heuristic anomaly); p -> 1 maps to baseline + span.
    """
    x = (float(p_fake) - 0.5) / 0.5
    x = min(max(x, 0.0), 1.0)
    return clean_baseline + span * x
