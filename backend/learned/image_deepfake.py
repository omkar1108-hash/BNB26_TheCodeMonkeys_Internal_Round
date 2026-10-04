"""Learned AI-generated-image classifier (Hugging Face image-classification model)."""
from typing import Any, Dict, Optional
from PIL import Image

from backend.learned import registry
from backend.config import DEVICE


def _load(model_id: str):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True).to(DEVICE).eval()
    return {"processor": processor, "model": model, "fake_idx": registry.fake_index(model.config.id2label),
            "id": model_id}


def predict(image: Image.Image) -> Optional[Dict[str, Any]]:
    """Returns {'p_fake', 'model_id'} or None when the learned model is off / not cached / failed."""
    bundle = registry.get_or_load("image_deepfake", _load)
    if bundle is None:
        return None
    try:
        import torch
        inputs = bundle["processor"](images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = bundle["model"](**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        return {"p_fake": float(probs[bundle["fake_idx"]]), "model_id": bundle["id"]}
    except Exception as e:
        registry._FAILED["image_deepfake"] = f"{type(e).__name__}: {e}"
        return None
