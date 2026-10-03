import os
import json
import wave
import random
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw
import numpy as np

from backend.config import DATASETS_DIR


def generate_synthetic_image(output_path: Path, manipulate: bool = False):
    """Generates an image artifact with optional localized manipulation."""
    img = Image.new("RGB", (256, 256), color=(random.randint(60, 200), random.randint(60, 200), random.randint(60, 200)))
    draw = ImageDraw.Draw(img)
    
    # Natural shapes
    for _ in range(5):
        x1, y1 = random.randint(10, 200), random.randint(10, 200)
        draw.rectangle([x1, y1, x1 + 40, y1 + 40], fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        
    if manipulate:
        # Inject artificial high-frequency noise block (splicing artifact)
        noise_box = [50, 50, 110, 110]
        noise_data = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_data)
        img.paste(noise_img, (50, 50))
        
    img.save(output_path, "JPEG", quality=95)


def generate_synthetic_audio(output_path: Path, manipulate: bool = False):
    """Generates a WAV audio artifact with optional high-frequency vocoder distortion."""
    framerate = 16000
    duration = 2.0  # seconds
    n_samples = int(framerate * duration)
    
    t = np.linspace(0, duration, n_samples, False)
    # Speech fundamental frequency simulation (~150-250 Hz harmonic tone)
    tone = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.25 * np.sin(2 * np.pi * 400 * t)
    
    if manipulate:
        # Add high-frequency noise simulating vocoder buzz / artifacts
        tone += 0.4 * np.random.uniform(-1, 1, n_samples)
        
    audio_data = (tone * 32767).astype(np.int16)
    
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(audio_data.tobytes())


def generate_bundle_dataset(
    output_dir: Path = DATASETS_DIR / "synthetic_bundles",
    num_per_class: int = 50
) -> Dict[str, int]:
    """
    Generates synthetic multimodal bundles across the 4 target classes:
    authentic, manipulated, coordinated_synthetic, insufficient_evidence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    
    classes = ["authentic", "manipulated", "coordinated_synthetic", "insufficient_evidence"]
    counts = {c: 0 for c in classes}
    
    dates = ["15 September 2026", "18 September 2026", "22 October 2026"]
    locations = ["Mumbai", "Pune", "Bengaluru", "Delhi"]
    speakers = ["Dr. Rajesh Sharma", "Prof. Anita Desai", "Spokesperson Khan"]
    
    for cls in classes:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_per_class):
            bundle_id = f"{cls}_{i:04d}"
            bundle_dir = cls_dir / bundle_id
            bundle_dir.mkdir(parents=True, exist_ok=True)
            
            img_path = bundle_dir / "image.jpg"
            aud_path = bundle_dir / "audio.wav"
            meta_path = bundle_dir / "metadata.json"
            txt_path = bundle_dir / "text.txt"
            
            chosen_date = random.choice(dates)
            chosen_loc = random.choice(locations)
            chosen_spk = random.choice(speakers)
            
            if cls == "authentic":
                generate_synthetic_image(img_path, manipulate=False)
                generate_synthetic_audio(aud_path, manipulate=False)
                meta = {
                    "Make": "Sony", "Model": "Alpha A7 IV",
                    "DateTimeOriginal": f"2026:09:15 10:00:00"
                }
                text = f"Annual Innovation Summit took place at Convention Center, {chosen_loc}, on {chosen_date}. Addressed by {chosen_spk}."
                
            elif cls == "manipulated":
                manip_target = random.choice(["image", "audio", "metadata"])
                generate_synthetic_image(img_path, manipulate=(manip_target == "image"))
                generate_synthetic_audio(aud_path, manipulate=(manip_target == "audio"))
                meta = {
                    "Software": "Adobe Photoshop 2024" if manip_target == "metadata" else "Camera Firmware 1.0",
                    "DateTimeOriginal": "2026:09:15 10:00:00"
                }
                text = f"Annual Innovation Summit took place at Convention Center, {chosen_loc}, on {chosen_date}. Addressed by {chosen_spk}."
                
            elif cls == "coordinated_synthetic":
                # Individual artifacts look clean (unmanipulated), but content contradicts!
                generate_synthetic_image(img_path, manipulate=False)
                generate_synthetic_audio(aud_path, manipulate=False)
                # Contradicting location and date
                other_loc = "Pune" if chosen_loc != "Pune" else "Mumbai"
                other_date = "18 September 2026" if chosen_date != "18 September 2026" else "22 October 2026"
                
                meta = {
                    "Make": "Nikon", "Model": "Z9",
                    "DateTimeOriginal": "2026:09:15 10:00:00"
                }
                # Text asserts other location and date
                text = f"Annual Innovation Summit took place at Convention Center, {other_loc}, on {other_date}. Addressed by {chosen_spk}."
                
            else:  # insufficient_evidence
                # Stripped or partial bundle (only caption without supporting signals)
                img_path = None
                aud_path = None
                meta = {}
                text = f"A brief unverified message about an event at {chosen_loc}."
                
            # Write metadata & text
            if meta_path:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            if txt_path and text:
                with open(txt_path, "w") as f:
                    f.write(text)
                    
            manifest.append({
                "bundle_id": bundle_id,
                "label": cls,
                "image_path": str(img_path) if img_path else None,
                "audio_path": str(aud_path) if aud_path else None,
                "text": text,
                "metadata_path": str(meta_path),
                "claimed_speaker": chosen_spk
            })
            counts[cls] += 1
            
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    return counts


if __name__ == "__main__":
    generated = generate_bundle_dataset(num_per_class=10)
    print("Synthetic Bundle Dataset Generated:", generated)
