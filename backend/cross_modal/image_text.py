from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from backend.config import DEVICE


def compute_image_caption_similarity(
    image_path: Optional[str],
    caption: Optional[str]
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    Computes cross-modal similarity between an image and text caption.
    Returns: (similarity_score [0, 1], is_fallback, fallback_reason)
    """
    if not image_path or not caption or not Path(image_path).exists() or not caption.strip():
        return None, False, None
        
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        
        model_id = "openai/clip-vit-base-patch32"
        # Attempt lightweight zero-shot similarity with local cache
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(DEVICE)
        
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[caption[:77]], images=image, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            similarity = float(torch.sigmoid(logits_per_image / 10.0).cpu().numpy()[0][0])
            
        return float(np.clip(similarity, 0.0, 1.0)), False, None
        
    except Exception as e:
        # Graceful fallback: lexical & color correlation heuristic
        # If words in caption mention visual terms (e.g. night, dark, red, bright), check image brightness
        try:
            with Image.open(image_path) as img:
                gray = img.convert("L").resize((32, 32))
                mean_bright = float(np.mean(np.array(gray))) / 255.0
                
            caption_lower = caption.lower()
            sim = 0.65  # baseline concordance
            
            if "night" in caption_lower or "dark" in caption_lower:
                sim = 0.85 if mean_bright < 0.40 else 0.25
            elif "sunny" in caption_lower or "day" in caption_lower or "bright" in caption_lower:
                sim = 0.85 if mean_bright > 0.50 else 0.30
                
            return sim, True, f"CLIP model weights not cached locally; lightweight photometric fallback used ({str(e)})"
        except Exception as inner_e:
            return 0.50, True, f"Image-text alignment error: {str(inner_e)}"

