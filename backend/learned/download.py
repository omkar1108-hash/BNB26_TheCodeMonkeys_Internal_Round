"""
Download (once) and verify the optional learned detectors.

    python -m backend.learned.download              # download all three
    python -m backend.learned.download --only audio_spoof clip
    python -m backend.learned.download --check      # no download: load each model, print label mapping, run a sanity pass

Weights go to the normal Hugging Face cache (~/.cache/huggingface). At runtime TrustLayer only
loads from that cache and never downloads on its own.
"""
import argparse
import sys
from typing import Dict, List

from backend.learned import registry

COMMON_ALLOW = ["*.json", "*.txt", "*.model", "*.py"]


def choose_patterns(repo_files: List[str]) -> Dict[str, List[str]]:
    """Pick the weight format to fetch: safetensors if the repo has it, else PyTorch .bin. Skip TF/Flax/ONNX."""
    has_safe = any(f.endswith(".safetensors") for f in repo_files)
    allow = list(COMMON_ALLOW) + (["*.safetensors"] if has_safe else ["*.bin"])
    ignore = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite", "*.pth", "*.ckpt", "training_args.bin", "optimizer*"]
    if has_safe:
        ignore.append("*.bin")
    return {"allow": allow, "ignore": ignore}


def download(key: str) -> None:
    from huggingface_hub import list_repo_files, snapshot_download
    mid = registry.model_id(key)
    print(f"[{key}] {mid}")
    files = list_repo_files(mid)
    pats = choose_patterns(files)
    path = snapshot_download(mid, allow_patterns=pats["allow"], ignore_patterns=pats["ignore"])
    print(f"    cached at {path}")


def check(key: str) -> bool:
    """Load the model and run a trivial input through it. Prints how labels were resolved."""
    import numpy as np
    from PIL import Image
    mid = registry.model_id(key)
    if not registry.is_cached(mid):
        print(f"[{key}] NOT CACHED  ({mid})")
        return False
    registry.reset()
    try:
        if key == "image_deepfake":
            from backend.learned import image_deepfake as m
            res = m.predict(Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8")))
            b = registry._CACHE.get(key)
            print(f"[{key}] OK  labels={dict(b['model'].config.id2label)}  fake_class={b['fake_idx']}  p_fake(noise image)={res and round(res['p_fake'], 3)}")
        elif key == "audio_spoof":
            import tempfile, wave
            from backend.learned import audio_spoof as m
            t = np.arange(32000) / 16000.0
            pcm = (np.sin(2 * np.pi * 220 * t) * 12000).astype("int16")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                with wave.open(f.name, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000); wf.writeframes(pcm.tobytes())
                res = m.predict(f.name)
            b = registry._CACHE.get(key)
            print(f"[{key}] OK  labels={dict(b['model'].config.id2label)}  fake_class={b['fake_idx']}  p_fake(sine tone)={res and round(res['p_fake'], 3)}")
        elif key == "clip":
            from backend.learned import clip_scorer as m
            img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
            sc = m.image_text_score(img, ["a photo of a crowded conference hall"])
            print(f"[{key}] OK  score(noise image vs caption)={sc and round(sc, 3)}")
        ok = key in registry._CACHE
        if not ok:
            print(f"    could not load: {registry._FAILED.get(key)}")
        return ok
    except Exception as e:
        print(f"[{key}] FAILED: {type(e).__name__}: {e}")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(registry.MODELS), help="subset of models")
    ap.add_argument("--check", action="store_true", help="verify cached models instead of downloading")
    args = ap.parse_args(argv)
    keys = args.only or list(registry.MODELS)
    if args.check:
        results = [check(k) for k in keys]
        return 0 if all(results) else 1
    for k in keys:
        try:
            download(k)
        except Exception as e:
            print(f"    download failed: {type(e).__name__}: {e}")
            return 1
    print("\nVerifying...")
    results = [check(k) for k in keys]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
