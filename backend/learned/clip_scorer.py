"""CLIP image-text agreement, expressed on a 0-1 scale."""
from typing import List, Optional
from PIL import Image

from backend.learned import registry
from backend.config import DEVICE

# Typical CLIP ViT-B/32 cosine similarity: ~0.28-0.34 for a caption that matches the image,
# ~0.10-0.20 for an unrelated one. These anchors are a sensible default, NOT calibrated on this
# project's data; treat the score as a screening signal.
COS_MISMATCH = 0.15
COS_MATCH = 0.30


def cosine_to_score(cos: float) -> float:
    s = (float(cos) - COS_MISMATCH) / (COS_MATCH - COS_MISMATCH)
    return float(min(max(s, 0.0), 1.0))


def _load(model_id: str):
    from transformers import CLIPModel, CLIPProcessor
    processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE).eval()
    return {"processor": processor, "model": model}


def image_text_cosine(image: Image.Image, texts: List[str]) -> Optional[float]:
    """
    Raw CLIP cosine similarity between the image and the best-matching text (max over texts).
    Unclipped, so it keeps resolution where ``image_text_score`` saturates; used by the evaluation.
    Text is truncated by the tokenizer (77 tokens), not by characters.
    """
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return None
    bundle = registry.get_or_load("clip", _load)
    if bundle is None:
        return None
    try:
        import torch
        proc, model = bundle["processor"], bundle["model"]
        inputs = proc(text=texts, images=image.convert("RGB"), return_tensors="pt",
                      padding=True, truncation=True, max_length=77).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        # CLIPModel returns L2-normalised embeddings, so the dot product is the cosine similarity.
        cos = (out.image_embeds @ out.text_embeds.T).squeeze(0).cpu().numpy()
        return float(cos.max())
    except Exception as e:
        registry._FAILED["clip"] = f"{type(e).__name__}: {e}"
        return None


def image_text_score(image: Image.Image, texts: List[str]) -> Optional[float]:
    """Best (max) agreement between the image and any of the text sources: 0 = unrelated, 1 = matching."""
    cos = image_text_cosine(image, texts)
    return None if cos is None else cosine_to_score(cos)
