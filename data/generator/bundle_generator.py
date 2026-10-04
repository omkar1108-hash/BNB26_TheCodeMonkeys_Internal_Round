"""
Synthetic, technique-tagged multimodal bundle generator.

Every bundle carries a ``technique`` tag so evaluation can hold out whole
manipulation techniques (Leave-One-Technique-Out) and measure generalisation to
manipulations the fusion model never saw during training.

IMPORTANT: this is a *controlled synthetic benchmark*. Images and audio are
procedurally generated, not real photographs or recordings, so the numbers it
produces measure the fusion/cross-modal reasoning, not real-world forensics.
"""
import io
import json
import wave
import random
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from backend.config import DATASETS_DIR
from backend.models.bundle import BundleInput

CITIES = [
    "Mumbai", "Pune", "Bengaluru", "Delhi", "Chennai", "Hyderabad", "Kolkata", "Ahmedabad",
    "Jaipur", "Lucknow", "Chandigarh", "Kochi", "Indore", "Bhopal", "Nagpur", "Surat",
    "London", "Singapore", "Dubai", "New York", "San Francisco"
]
VENUES = [
    "Convention Center", "City Auditorium", "Town Hall", "Institute of Technology",
    "Exhibition Grounds", "Science City", "International Center", "University Campus",
    "Community Hall", "Grand Pavilion", "Business Park"
]
EVENTS = [
    "Annual Innovation Summit", "TechFest 2026", "Regional Flood Relief Briefing",
    "Clean Energy Expo", "Public Health Awareness Drive", "Startup Founders Meetup",
    "Global AI Conclave", "Cybersecurity Forum", "National Science Congress",
    "Economic Development Summit", "Disaster Resilience Workshop"
]
SPEAKERS = [
    "Rajesh Sharma", "Anita Desai", "Imran Khan", "Meera Nair", "Vikram Patel",
    "Sunita Rao", "Arun Varma", "Pooja Hegde", "Siddharth Roy", "Deepa Malik",
    "Kavita Krishnan", "Rohan Joshi", "Amitav Sen", "Priyanka Das"
]
HONORIFICS = ["Dr. ", "Prof. ", "Mr. ", "Ms. ", ""]
ORGS = [
    "the Computer Department", "the City Council", "the Health Ministry",
    "the Industry Forum", "the National Science Foundation", "the Tech Advisory Board"
]
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"
]
CAMERAS = [
    ("Sony", "Alpha A7 IV"), ("Nikon", "Z9"), ("Canon", "EOS R6"), ("Apple", "iPhone 15"),
    ("Samsung", "Galaxy S24"), ("Google", "Pixel 8"), ("Fujifilm", "X-T5"), ("Xiaomi", "14 Ultra")
]

# technique -> (label, family)
TECHNIQUES: Dict[str, Tuple[str, str]] = {
    # authentic
    "authentic_full": ("authentic", "authentic"),
    "authentic_partial": ("authentic", "authentic"),
    "authentic_two_sources": ("authentic", "authentic"),
    "authentic_hard_negative": ("authentic", "authentic"),
    # single-artifact manipulation
    "image_noise_patch": ("manipulated", "image"),
    "image_double_jpeg": ("manipulated", "image"),
    "image_blur_patch": ("manipulated", "image"),
    "audio_vocoder_noise": ("manipulated", "audio"),
    "audio_bandlimit": ("manipulated", "audio"),
    "audio_splice": ("manipulated", "audio"),
    "metadata_editor_tag": ("manipulated", "metadata"),
    "metadata_ai_tag": ("manipulated", "metadata"),
    "metadata_timestamp_gap": ("manipulated", "metadata"),
    # cross-modal contradictions between individually clean artifacts
    "coord_date_shift": ("coordinated_synthetic", "cross_modal"),
    "coord_location_swap": ("coordinated_synthetic", "cross_modal"),
    "coord_speaker_swap": ("coordinated_synthetic", "cross_modal"),
    "coord_cross_doc_date": ("coordinated_synthetic", "cross_modal"),
    "coord_cross_doc_location": ("coordinated_synthetic", "cross_modal"),
    "coord_date_and_location": ("coordinated_synthetic", "cross_modal"),
    # too little evidence
    "lone_text": ("insufficient_evidence", "sparse"),
    "lone_image": ("insufficient_evidence", "sparse"),
    "lone_audio": ("insufficient_evidence", "sparse"),
    "lone_metadata": ("insufficient_evidence", "sparse"),
}


# ---------------------------------------------------------------------------
# Artifact synthesis
# ---------------------------------------------------------------------------
def generate_synthetic_image(output_path: Path, technique: Optional[str] = None,
                             rng: Optional[np.random.Generator] = None, launder_prob: float = 0.0):
    """Photo-like image: gradient + shapes + sensor noise, optionally manipulated with randomized patch location/size."""
    rng = rng or np.random.default_rng()
    h = w = 256
    c0 = rng.integers(40, 200, 3)
    c1 = rng.integers(40, 220, 3)
    ramp = np.linspace(0, 1, w)[None, :, None]
    base = (c0[None, None, :] * (1 - ramp) + c1[None, None, :] * ramp) * np.ones((h, 1, 1))
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(4, 9))):
        x1, y1 = int(rng.integers(10, 190)), int(rng.integers(10, 190))
        draw.rectangle([x1, y1, x1 + int(rng.integers(25, 65)), y1 + int(rng.integers(25, 65))],
                       fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    noise_sigma = float(rng.uniform(2.0, 4.5))
    arr = np.array(img).astype(np.float32) + rng.normal(0, noise_sigma, (h, w, 3))  # sensor noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Randomize patch size and position across the frame
    pw = int(rng.integers(35, 110))
    ph = int(rng.integers(35, 110))
    px = int(rng.integers(10, max(11, w - pw - 10)))
    py = int(rng.integers(10, max(11, h - ph - 10)))
    box = (px, py, px + pw, py + ph)

    if technique == "image_noise_patch":
        noise = Image.fromarray(rng.integers(0, 256, (ph, pw, 3), dtype=np.uint8))
        img.paste(noise, box[:2])
    elif technique == "image_double_jpeg":
        patch = img.crop(box)
        buf = io.BytesIO()
        recomp_q = int(rng.integers(15, 45))
        patch.save(buf, "JPEG", quality=recomp_q)
        buf.seek(0)
        img.paste(Image.open(buf).convert("RGB"), box[:2])
    elif technique == "image_blur_patch":
        blur_rad = float(rng.uniform(2.5, 6.0))
        img.paste(img.crop(box).filter(ImageFilter.GaussianBlur(blur_rad)), box[:2])

    base_q = int(rng.integers(88, 96))
    img.save(output_path, "JPEG", quality=base_q)
    if launder_prob and rng.random() < launder_prob:
        # Simulate re-sharing: lossy re-compression and (sometimes) rescaling.
        im = Image.open(output_path).convert("RGB")
        if rng.random() < 0.5:
            f = float(rng.uniform(0.70, 0.95))
            im = im.resize((int(256 * f), int(256 * f))).resize((256, 256))
        im.save(output_path, "JPEG", quality=int(rng.integers(60, 88)))


def generate_synthetic_audio(output_path: Path, technique: Optional[str] = None,
                             rng: Optional[np.random.Generator] = None,
                             low_bitrate: bool = False):
    """Speech-like 16 kHz WAV: harmonic series + breath noise, optionally manipulated with varied duration and cutoff."""
    rng = rng or np.random.default_rng()
    sr = 16000
    dur = float(rng.uniform(1.8, 2.5))
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0 = float(rng.uniform(105, 250))
    sig = np.zeros(n)
    for k in range(1, int(7000 / f0)):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
    mod_f = float(rng.uniform(2.5, 4.0))
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * mod_f * t)  # syllable-rate amplitude modulation
    noise_lvl = float(rng.uniform(0.015, 0.055))
    sig += rng.normal(0, noise_lvl, n)

    if technique == "audio_vocoder_noise":
        sig += float(rng.uniform(0.55, 0.95)) * rng.uniform(-1, 1, n)
    elif technique == "audio_bandlimit":
        spec = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(n, 1 / sr)
        cutoff = float(rng.uniform(1800, 2400))
        spec[freqs > cutoff] = 0
        sig = np.fft.irfft(spec, n)
    elif technique == "audio_splice":
        cut_frac = float(rng.uniform(0.35, 0.65))
        cut = int(n * cut_frac)
        other = np.zeros(n - cut)
        f1 = f0 * float(rng.uniform(1.3, 1.9))
        for k in range(1, int(7000 / f1)):
            other += (1.0 / k) * np.sin(2 * np.pi * f1 * k * t[: n - cut])
        sig = np.concatenate([sig[:cut], other * (np.max(np.abs(sig)) / (np.max(np.abs(other)) + 1e-9))])

    if low_bitrate:
        # Simulate quantization / ambient noise
        sig += rng.normal(0, 0.06, n)

    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Text / metadata synthesis
# ---------------------------------------------------------------------------
def _fmt_date(d: Tuple[int, int, int], style: int) -> str:
    y, m, day = d
    if style == 0:
        return f"{day} {MONTHS[m - 1]} {y}"
    if style == 1:
        return f"{MONTHS[m - 1]} {day}, {y}"
    if style == 2:
        return f"{y:04d}-{m:02d}-{day:02d}"
    if style == 3:
        return f"{day:02d}/{m:02d}/{y:04d}"
    # Ordinal suffix format: e.g. "15th September 2026"
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix} {MONTHS[m - 1]} {y}"


def _rand_date(rnd: random.Random, avoid: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    while True:
        d = (rnd.choice([2025, 2026]), rnd.randint(1, 12), rnd.randint(1, 28))
        if d != avoid:
            return d


def _make_text(rnd: random.Random, event: str, venue: str, city: str, date: Tuple[int, int, int],
               speaker: Optional[str], org: str, template: int, date_style: int) -> str:
    ds = _fmt_date(date, date_style)
    spk = f" Addressed by {speaker}." if speaker else ""
    if template == 0:
        return f"{event} took place at {venue}, {city}, on {ds}.{spk}"
    if template == 1:
        return f"{event} was held at {venue}, {city}, on {ds}. The event was organized by {org}.{spk}"
    if template == 2:
        return f"Report: {event} in {city} on {ds}.{spk}"
    return f"{event} was hosted at {venue}, {city}, on {ds}. Attendance was high.{spk}"


def _make_meta(rnd: random.Random, date: Tuple[int, int, int], city: str,
               camera: bool = True) -> Dict[str, Any]:
    y, m, d = date
    ts = f"{y:04d}:{m:02d}:{d:02d} {rnd.randint(8, 18):02d}:{rnd.randint(0, 59):02d}:{rnd.randint(0, 59):02d}"
    meta: Dict[str, Any] = {"DateTimeOriginal": ts, "DateTime": ts, "Location": city}
    if camera:
        make, model = rnd.choice(CAMERAS)
        meta.update({"Make": make, "Model": model})
    return meta


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------
def build_spec(technique: str, rnd: random.Random) -> Dict[str, Any]:
    """Decide which modalities a bundle contains and what each one asserts."""
    label, family = TECHNIQUES[technique]
    event = rnd.choice(EVENTS)
    venue = rnd.choice(VENUES)
    city = rnd.choice(CITIES)
    date = _rand_date(rnd)
    speaker = rnd.choice(SPEAKERS)
    org = rnd.choice(ORGS)
    template = rnd.randint(0, 3)
    date_style = rnd.randint(0, 4)
    honorific = rnd.choice(HONORIFICS)

    spec: Dict[str, Any] = dict(
        technique=technique, label=label, family=family, image=None, audio=None, meta=None,
        docs={}, claimed_speaker=None,
    )
    true_text = lambda spk=None: _make_text(rnd, event, venue, city, date, spk, org, template, date_style)
    other_city = rnd.choice([c for c in CITIES if c != city])
    other_date = _rand_date(rnd, avoid=date)
    spk_name = f"{honorific}{speaker}"

    def add_extras(layouts):
        return rnd.choice(layouts)

    if technique == "authentic_full":
        spec["image"], spec["audio"] = "clean", "clean"
        spec["meta"] = _make_meta(rnd, date, city, camera=rnd.random() > 0.15)
        spec["docs"] = {"caption": true_text(spk_name)}
        spec["claimed_speaker"] = spk_name if rnd.random() > 0.3 else speaker
    elif technique == "authentic_partial":
        combo = add_extras(["img_meta", "audio", "meta", "img"])
        has_img, has_aud, has_meta = {"img_meta": (1, 0, 1), "audio": (0, 1, 0), "meta": (0, 0, 1), "img": (1, 0, 0)}[combo]
        spec["image"] = "clean" if has_img else None
        spec["audio"] = "clean" if has_aud else None
        spec["meta"] = _make_meta(rnd, date, city, camera=rnd.random() > 0.15) if has_meta else None
        spec["docs"] = {"caption": true_text(spk_name if has_aud else None)}
        if has_aud:
            spec["claimed_speaker"] = spk_name
    elif technique == "authentic_two_sources":
        spec["docs"] = {"message": true_text(spk_name), "report": _make_text(rnd, event, venue, city, date, None, org, (template + 1) % 4, (date_style + 1) % 5)}
        if rnd.random() > 0.5:
            spec["meta"] = _make_meta(rnd, date, city)
    elif technique == "authentic_hard_negative":
        # Hard negative authentic: re-compressed image, missing EXIF or camera stripped,
        # noisy/ambient audio, or short caption. Label is strictly authentic.
        spec["image"] = "clean"
        spec["audio"] = "clean"
        if rnd.random() > 0.5:
            spec["meta"] = _make_meta(rnd, date, city, camera=False)
        else:
            spec["meta"] = None
        if rnd.random() > 0.5:
            spec["docs"] = {"caption": f"At {venue}, {city} on {_fmt_date(date, date_style)}. {spk_name} addressed the audience."}
        else:
            spec["docs"] = {"caption": true_text(spk_name)}
        spec["claimed_speaker"] = spk_name
        spec["hard_negative"] = True
    elif family == "image":
        spec["image"] = technique
        spec["docs"] = {"caption": true_text()}
        if rnd.random() > 0.4:
            spec["meta"] = _make_meta(rnd, date, city)
        if rnd.random() > 0.6:
            spec["audio"] = "clean"
            spec["docs"] = {"caption": true_text(spk_name)}
            spec["claimed_speaker"] = spk_name
    elif family == "audio":
        spec["audio"] = technique
        spec["docs"] = {"caption": true_text(spk_name)}
        spec["claimed_speaker"] = spk_name
        if rnd.random() > 0.5:
            spec["image"] = "clean"
        if rnd.random() > 0.5:
            spec["meta"] = _make_meta(rnd, date, city)
    elif family == "metadata":
        meta = _make_meta(rnd, date, city, camera=True)
        if technique == "metadata_editor_tag":
            meta["Software"] = rnd.choice(["Adobe Photoshop 2024", "GIMP 2.10", "Canva", "Paint.NET"])
        elif technique == "metadata_ai_tag":
            meta["Software"] = rnd.choice(["Midjourney v6", "Stable Diffusion 1.5", "DALL-E 3", "ComfyUI"])
            meta.pop("Make", None)
            meta.pop("Model", None)
        else:  # timestamp gap between capture and write
            y, mo, d = date
            meta["DateTime"] = f"{y:04d}:{mo:02d}:{min(d + 3, 28):02d} 23:59:59" if d < 25 else f"{y:04d}:{mo:02d}:01 00:00:01"
        spec["meta"] = meta
        spec["docs"] = {"caption": true_text()}
        if rnd.random() > 0.5:
            spec["image"] = "clean"
    elif technique == "coord_date_shift":
        spec["meta"] = _make_meta(rnd, date, city)
        spec["docs"] = {"caption": _make_text(rnd, event, venue, city, other_date, None, org, template, date_style)}
        if rnd.random() > 0.5:
            spec["image"] = "clean"
    elif technique == "coord_location_swap":
        spec["meta"] = _make_meta(rnd, date, city)
        spec["docs"] = {"caption": _make_text(rnd, event, venue, other_city, date, None, org, template, date_style)}
        if rnd.random() > 0.5:
            spec["image"] = "clean"
    elif technique == "coord_speaker_swap":
        other_speaker = rnd.choice([s for s in SPEAKERS if s != speaker])
        spec["audio"] = "clean"
        spec["claimed_speaker"] = f"{rnd.choice(HONORIFICS)}{other_speaker}"
        spec["docs"] = {"caption": true_text(spk_name)}
        if rnd.random() > 0.5:
            spec["meta"] = _make_meta(rnd, date, city)
    elif technique == "coord_cross_doc_date":
        spec["docs"] = {"message": true_text(), "report": _make_text(rnd, event, venue, city, other_date, None, org, (template + 1) % 4, date_style)}
        if rnd.random() > 0.6:
            spec["image"] = "clean"
    elif technique == "coord_cross_doc_location":
        spec["docs"] = {"message": true_text(), "report": _make_text(rnd, event, venue, other_city, date, None, org, (template + 1) % 4, date_style)}
        if rnd.random() > 0.6:
            spec["image"] = "clean"
    elif technique == "coord_date_and_location":
        spec["meta"] = _make_meta(rnd, date, city)
        spec["docs"] = {"message": _make_text(rnd, event, venue, other_city, other_date, None, org, template, date_style),
                        "caption": true_text()}
        spec["image"] = "clean"
    elif technique == "lone_text":
        spec["docs"] = {"message": f"A brief unverified message about an event in {city}."}
    elif technique == "lone_image":
        spec["image"] = "clean"
    elif technique == "lone_audio":
        spec["audio"] = "clean"
    elif technique == "lone_metadata":
        spec["meta"] = _make_meta(rnd, date, city)
    return spec


def _write_bundle(spec: Dict[str, Any], bundle_dir: Path, base_dir: Path,
                  rng: np.random.Generator, launder_prob: float = 0.0) -> Dict[str, Any]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(p.relative_to(base_dir)).replace("\\", "/")
    rec: Dict[str, Any] = {
        "bundle_id": bundle_dir.name,
        "label": spec["label"],
        "technique": spec["technique"],
        "family": spec["family"],
        "image_path": None, "audio_path": None, "metadata_path": None,
        "documents": {}, "claimed_speaker": spec["claimed_speaker"],
    }
    is_hard_neg = spec.get("hard_negative", False)
    img_launder = 1.0 if is_hard_neg else launder_prob
    if spec["image"]:
        p = bundle_dir / "image.jpg"
        generate_synthetic_image(p, None if spec["image"] == "clean" else spec["image"], rng, img_launder)
        rec["image_path"] = rel(p)
    if spec["audio"]:
        p = bundle_dir / "audio.wav"
        generate_synthetic_audio(p, None if spec["audio"] == "clean" else spec["audio"], rng, low_bitrate=is_hard_neg)
        rec["audio_path"] = rel(p)
    if spec["meta"]:
        p = bundle_dir / "metadata.json"
        p.write_text(json.dumps(spec["meta"], indent=2))
        rec["metadata_path"] = rel(p)
    for name, text in spec["docs"].items():
        (bundle_dir / f"{name}.txt").write_text(text)
        rec["documents"][name] = text
    return rec


def generate_bundle_dataset(
    output_dir: Path = DATASETS_DIR / "synthetic_bundles",
    n_per_technique: int = 40,
    seed: int = 7,
    launder_prob: float = 0.35,
) -> Dict[str, int]:
    """Generate ``n_per_technique`` bundles for every technique; returns counts per label."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    manifest: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for technique, (label, _family) in TECHNIQUES.items():
        for i in range(n_per_technique):
            spec = build_spec(technique, rnd)
            rec = _write_bundle(spec, output_dir / label / f"{technique}_{i:04d}", output_dir, rng, launder_prob)
            manifest.append(rec)
            counts[label] = counts.get(label, 0) + 1
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return counts


def record_to_bundle(rec: Dict[str, Any], base_dir: Path) -> BundleInput:
    """Turn a manifest record into a BundleInput (resolving relative paths)."""
    resolve = lambda p: str((base_dir / p)) if p else None
    meta = None
    if rec.get("metadata_path"):
        meta = json.loads((base_dir / rec["metadata_path"]).read_text())
    return BundleInput(
        bundle_id=rec["bundle_id"],
        image_path=resolve(rec.get("image_path")),
        audio_path=resolve(rec.get("audio_path")),
        documents=rec.get("documents") or None,
        metadata=meta,
        claimed_speaker=rec.get("claimed_speaker"),
    )


if __name__ == "__main__":
    generated = generate_bundle_dataset()
    print("Synthetic bundle dataset generated:", generated)
