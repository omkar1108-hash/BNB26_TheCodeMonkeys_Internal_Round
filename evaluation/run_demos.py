"""Run the demo folders through the live pipeline and print label, confidence and key evidence."""
import json
from pathlib import Path
from backend.models.bundle import BundleInput
from backend.verdict.engine import analyze_bundle
from backend.learned import registry

DEMOS = Path(__file__).resolve().parents[1] / "data" / "demos"
EXPECTED = {"case1_authentic": "authentic", "case2_manipulated": "manipulated",
            "case3_coordinated_fake": "coordinated_synthetic", "case4_insufficient": "insufficient_evidence"}


def load_case(d: Path) -> BundleInput:
    docs = {p.stem: p.read_text() for p in sorted(d.glob("*.txt")) if p.name != "claimed_speaker.txt"}
    meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else None
    spk = (d / "claimed_speaker.txt").read_text().strip() if (d / "claimed_speaker.txt").exists() else None
    img = d / "photo.jpg"
    aud = d / "speech.wav"
    return BundleInput(bundle_id=d.name, image_path=str(img) if img.exists() else None,
                       audio_path=str(aud) if aud.exists() else None, documents=docs or None,
                       metadata=meta, claimed_speaker=spk)


def run_all() -> dict:
    out = {}
    for name, expected in EXPECTED.items():
        # Demo bundles are synthetic stand-ins, not real photos/voices: learned detectors stay off.
        with registry.disabled():
            r = analyze_bundle(load_case(DEMOS / name))
        out[name] = (expected, r)
    return out


if __name__ == "__main__":
    ok = 0
    for name, (expected, r) in run_all().items():
        hit = r.label == expected
        ok += hit
        print(f"{'PASS' if hit else 'FAIL'}  {name:26s} expected={expected:22s} got={r.label:22s} conf={r.confidence:.2f} conformal={r.conformal_set}")
        for e in r.ranked_evidence[:3]:
            print(f"        - {e.description}")
        for s in r.next_steps[:2]:
            print(f"        > {s}")
    print(f"{ok}/4 demo cases match")
