from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image

from backend.learned import registry, clip_scorer


def compute_image_text_similarity(
    image_path: Optional[str],
    texts: Union[str, List[str], None],
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    CLIP agreement between an image and its text source(s), on a 0-1 scale (None = not computed).
    Returns (score, is_fallback, reason). The score is evidence for a human reviewer and a guard
    against calling a bundle authentic; it is NOT a feature of the trained fusion model.
    A missing CLIP model is reported as a fallback, never replaced by a guess.
    """
    if isinstance(texts, str):
        texts = [texts]
    texts = [t for t in (texts or []) if t and t.strip()]
    if not image_path or not texts or not Path(image_path).exists():
        return None, False, None
    if not registry.enabled():
        return None, False, None
    if not registry.is_cached(registry.model_id("clip")):
        return None, True, (
            f"CLIP weights ({registry.model_id('clip')}) are not cached; image-text check skipped. "
            "Run `python -m backend.learned.download`."
        )
    try:
        with Image.open(image_path) as img:
            score = clip_scorer.image_text_score(img, texts)
    except Exception as e:
        return None, True, f"Image-text check failed: {e}"
    if score is None:
        return None, True, f"CLIP unavailable: {registry.status()['clip'].get('last_error') or 'load failed'}"
    return score, False, None


def compute_image_caption_similarity(image_path, caption):
    """Backward-compatible wrapper."""
    return compute_image_text_similarity(image_path, caption)
