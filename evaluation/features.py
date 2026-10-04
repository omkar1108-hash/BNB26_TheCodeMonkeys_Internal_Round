"""Turn a generated dataset into a fusion feature table (cached as .npz)."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

from backend.config import TARGET_CLASSES
from backend.fusion.feature_builder import build_fusion_features
from backend.verdict.engine import compute_bundle_signals
from data.generator.bundle_generator import record_to_bundle, generate_bundle_dataset
from backend.learned import registry


@dataclass
class Table:
    X: np.ndarray
    y: np.ndarray            # class indices into TARGET_CLASSES
    techniques: np.ndarray
    families: np.ndarray
    ids: np.ndarray
    fallback: np.ndarray     # True where a *signal* fallback was used (excluded from metrics)

    def subset(self, mask: np.ndarray) -> "Table":
        return Table(self.X[mask], self.y[mask], self.techniques[mask], self.families[mask],
                     self.ids[mask], self.fallback[mask])


def build_table(dataset_dir: Path, use_cache: bool = True) -> Table:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    cache = dataset_dir / "features.npz"
    if use_cache and cache.exists() and cache.stat().st_mtime >= manifest_path.stat().st_mtime:
        z = np.load(cache, allow_pickle=False)
        return Table(z["X"], z["y"], z["techniques"], z["families"], z["ids"], z["fallback"])

    records = json.loads(manifest_path.read_text())
    X: List[np.ndarray] = []
    y, techs, fams, ids, fb = [], [], [], [], []
    for rec in records:
        # Synthetic benchmark images/audio are not real photos/voices: learned detectors stay off
        # so results are reproducible on any machine regardless of which weights are cached.
        with registry.disabled():
            sig = compute_bundle_signals(record_to_bundle(rec, dataset_dir))
        X.append(build_fusion_features(sig["modality_results"], sig["cross_modal"]))
        y.append(TARGET_CLASSES.index(rec["label"]))
        techs.append(rec["technique"])
        fams.append(rec["family"])
        ids.append(rec["bundle_id"])
        fb.append(any(ev.is_fallback for ev in sig["modality_results"].values()) or sig["cross_modal_fallback"])
    t = Table(np.vstack(X), np.array(y), np.array(techs), np.array(fams), np.array(ids), np.array(fb))
    np.savez(cache, X=t.X, y=t.y, techniques=t.techniques, families=t.families, ids=t.ids, fallback=t.fallback)
    return t


DATASETS = {
    # name: (subdir, n_per_technique, seed)
    "train": ("synthetic_bundles", 40, 7),
    "calib": ("synthetic_calib", 15, 99),
    "test": ("synthetic_test", 15, 123),
}


def ensure_datasets(root: Path, regenerate: bool = False) -> dict:
    """Generate the three independent datasets (train / calibration / test) if missing."""
    tables = {}
    for name, (sub, n, seed) in DATASETS.items():
        d = Path(root) / sub
        if regenerate or not (d / "manifest.json").exists():
            generate_bundle_dataset(d, n_per_technique=n, seed=seed)
            cache = d / "features.npz"
            if cache.exists():
                cache.unlink()
        tables[name] = build_table(d)
    return tables
