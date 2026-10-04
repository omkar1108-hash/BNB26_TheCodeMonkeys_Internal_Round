"""
Evaluate the TrustLayer detectors on REAL labelled datasets (not the synthetic benchmark).

    # image: folders named real/fake (or human/ai, REAL/FAKE, ...) anywhere under --root
    python -m evaluation.dataset_eval image --root data/datasets/my_images --limit 500 \
        --perturb none jpeg70 resize50_jpeg85

    # audio: ASVspoof-style protocol file, or a CSV manifest (path,label[,group,condition])
    python -m evaluation.dataset_eval audio --root /data/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac \
        --asvspoof-protocol /data/ASVspoof2019/LA/ASVspoof2019_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --limit 500
    python -m evaluation.dataset_eval audio --root /data/in_the_wild --manifest meta.csv --limit 500

    # CLIP image-caption check: CSV manifest with path,caption
    python -m evaluation.dataset_eval clip --root /data/coco_val --manifest captions.csv --limit 500

    # look at what would be evaluated without scoring anything
    python -m evaluation.dataset_eval image --root ... --dry-run

Each sample is scored three ways, so you can see what the learned model adds:
    heuristic : the shipped detector with learned models OFF
    learned   : the raw classifier probability P(fake) (read from the detector's own features)
    combined  : the shipped detector with learned models ON (what a user actually gets)
Label convention everywhere: 1 = fake / spoof / manipulated, 0 = real / bona fide.

Outputs (evaluation/results/): dataset_eval_<task>_<name>.json / .md / _scores.csv
"""
import argparse
import csv
import json
import random
import re
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.learned import registry

REAL, FAKE = 0, 1
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
COLUMNS = ("heuristic", "learned", "combined")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ============================================================ samples & loading
@dataclass
class Sample:
    id: str
    path: str
    label: int
    group: Optional[str] = None       # generator / attack id (used for the per-group breakdown)
    condition: Optional[str] = None   # e.g. codec; image perturbations are added at scoring time
    caption: Optional[str] = None


def parse_label(value: Any) -> Optional[int]:
    """'0'/'1', or any real/fake word the learned-model registry already understands (bona-fide, spoof, AI, human...)."""
    s = str(value).strip().lower()
    if s in ("0", "1"):
        return int(s)
    toks = registry._tokens(s)
    fake, real = bool(toks & registry.FAKE_TOKENS), bool(toks & registry.REAL_TOKENS)
    if fake and not real:
        return FAKE
    if real and not fake:
        return REAL
    return None


def load_folder(root: Path, exts: set, real_dirs: Sequence[str] = (), fake_dirs: Sequence[str] = ()) -> Tuple[List[Sample], int]:
    """Label = the nearest ancestor directory (below root) whose name says real or fake. Returns (samples, n_unlabelled)."""
    real_set, fake_set = {d.lower() for d in real_dirs}, {d.lower() for d in fake_dirs}

    def dir_label(name: str) -> Optional[int]:
        if name.lower() in real_set:
            return REAL
        if name.lower() in fake_set:
            return FAKE
        return parse_label(name)

    samples, unlabelled = [], 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        parts = p.relative_to(root).parent.parts
        label, idx = None, -1
        for i in range(len(parts) - 1, -1, -1):
            label = dir_label(parts[i])
            if label is not None:
                idx = i
                break
        if label is None:
            unlabelled += 1
            continue
        group = parts[idx + 1] if idx + 1 < len(parts) else None
        samples.append(Sample(id=str(p.relative_to(root)), path=str(p), label=label, group=group))
    return samples, unlabelled


def load_manifest(csv_path: Path, root: Optional[Path]) -> Tuple[List[Sample], int]:
    """CSV with header. Required: path, label (or caption for the clip task). Optional: group, condition, caption."""
    base = root or csv_path.parent
    samples, bad = [], 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
            rel = row.get("path") or row.get("file") or row.get("filename")
            if not rel:
                bad += 1
                continue
            label = parse_label(row["label"]) if row.get("label") not in (None, "") else REAL
            if label is None:
                bad += 1
                continue
            p = Path(rel) if Path(rel).is_absolute() else base / rel
            samples.append(Sample(id=rel, path=str(p), label=label, group=row.get("group") or None,
                                  condition=row.get("condition") or None, caption=row.get("caption") or None))
    return samples, bad


def load_asvspoof(protocol: Path, audio_dir: Path, exts: Sequence[str] = (".flac", ".wav")) -> Tuple[List[Sample], int]:
    """
    ASVspoof protocol lines, e.g.
      2019: LA_0039 LA_E_2834763 - A11 spoof            LA_0079 LA_T_1138215 - - bonafide
      2021: LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval
    The key (bonafide/spoof) and the attack id (A01..A99) are located by value, so both layouts parse.
    Returns (samples, n_missing_files).
    """
    samples, missing = [], 0
    with open(protocol, encoding="utf-8") as f:
        for line in f:
            fld = line.split()
            key = next((x.lower() for x in fld if x.lower() in ("bonafide", "spoof")), None)
            if len(fld) < 5 or key is None:
                continue
            label = REAL if key == "bonafide" else FAKE
            attack = next((x for x in fld[2:] if re.fullmatch(r"A\d{2}", x)), None)
            cond = fld[2] if len(fld) >= 8 and not re.fullmatch(r"A\d{2}|-", fld[2]) else None
            path = next((audio_dir / (fld[1] + e) for e in exts if (audio_dir / (fld[1] + e)).exists()), None)
            if path is None:
                missing += 1
                continue
            samples.append(Sample(id=fld[1], path=str(path), label=label, group=attack if label == FAKE else None, condition=cond))
    return samples, missing


def balanced_sample(samples: List[Sample], per_class: Optional[int], seed: int) -> List[Sample]:
    """Up to ``per_class`` samples from each class, deterministic for a given seed."""
    if not per_class:
        return list(samples)
    rng = random.Random(seed)
    out: List[Sample] = []
    for lab in (REAL, FAKE):
        pool = sorted((s for s in samples if s.label == lab), key=lambda s: s.id)
        out += rng.sample(pool, min(per_class, len(pool)))
    return sorted(out, key=lambda s: s.id)


def assign_calibration_ids(samples: Sequence[Sample], frac: float, seed: int) -> set:
    """Stratified set of sample ids reserved for choosing a threshold (never used for reported metrics)."""
    if frac <= 0:
        return set()
    rng, ids = random.Random(seed + 1), set()
    for lab in (REAL, FAKE):
        pool = sorted(s.id for s in samples if s.label == lab)
        rng.shuffle(pool)
        ids.update(pool[: int(round(frac * len(pool)))])
    return ids


# ============================================================ metrics (pure numpy / sklearn)
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auroc_ci(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> Optional[Tuple[float, float]]:
    """Stratified bootstrap 95% CI."""
    if n_boot <= 0:
        return None
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        p = rng.choice(pos, len(pos), replace=True)
        n = rng.choice(neg, len(neg), replace=True)
        vals.append(auroc(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def operating_point(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {"threshold": float(thr), "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": float(np.nanmean([tpr, 1 - fpr])),
            "tpr": tpr, "fpr": fpr, "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def youden_threshold(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_curve
    if len(set(y.tolist())) < 2:
        return None
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return t if np.isfinite(t) else float(np.max(s))


def ranking_metrics(y: np.ndarray, s: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_curve
    out: Dict[str, Any] = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    if out["n_real"] == 0 or out["n_fake"] == 0:
        out["note"] = "needs both classes"
        return out
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    k = int(np.argmin(np.abs(fnr - fpr)))
    ci = bootstrap_auroc_ci(y, s, n_boot, seed)
    out.update({
        "auroc": auroc(y, s), "auroc_ci95": ci,
        "average_precision": float(average_precision_score(y, s)),
        "eer": float((fpr[k] + fnr[k]) / 2),
        "tpr_at_1pct_fpr": float(tpr[fpr <= 0.01].max()), "tpr_at_5pct_fpr": float(tpr[fpr <= 0.05].max()),
    })
    return out


def evaluate_column(rows: List[Dict[str, Any]], col: str, thr: float, calib_ids: set, n_boot: int, seed: int) -> Dict[str, Any]:
    """Metrics for one score column over the given rows. Failed samples (score None) are counted, not scored."""
    scored = [r for r in rows if r["scores"].get(col) is not None]
    out: Dict[str, Any] = {"n_failed_or_unavailable": len(rows) - len(scored)}
    if not scored:
        out["note"] = "no scores (model unavailable or every sample failed)"
        return out
    if calib_ids:
        cal = [r for r in scored if r["id"] in calib_ids]
        scored = [r for r in scored if r["id"] not in calib_ids]
        yc = np.array([r["label"] for r in cal]); sc = np.array([r["scores"][col] for r in cal], dtype=float)
        tuned = youden_threshold(yc, sc) if len(cal) else None
    else:
        tuned = None
    y = np.array([r["label"] for r in scored]); s = np.array([r["scores"][col] for r in scored], dtype=float)
    out.update(ranking_metrics(y, s, n_boot, seed))
    if "auroc" in out:
        out["at_fixed_threshold"] = operating_point(y, s, thr)
        if tuned is not None:
            out["at_tuned_threshold"] = operating_point(y, s, tuned)
            out["tuned_on_n"] = int(len(cal))
    return out


def group_breakdown(rows: List[Dict[str, Any]], col: str, thr: float, calib_ids: set) -> Dict[str, Any]:
    """Per generator / attack: detection rate at the fixed threshold and AUROC against all reals."""
    scored = [r for r in rows if r["scores"].get(col) is not None and r["id"] not in calib_ids]
    reals = [r["scores"][col] for r in scored if r["label"] == REAL]
    out = {}
    for g in sorted({r["group"] for r in scored if r["label"] == FAKE and r["group"]}):
        fk = [r["scores"][col] for r in scored if r["label"] == FAKE and r["group"] == g]
        if len(fk) < 5 or not reals:
            continue
        y = np.r_[np.ones(len(fk)), np.zeros(len(reals))]
        s = np.r_[fk, reals].astype(float)
        out[g] = {"n_fake": len(fk), "detection_rate_at_threshold": float(np.mean(np.array(fk) >= thr)), "auroc_vs_reals": auroc(y, s)}
    return out


# ============================================================ perturbations (image robustness)
def perturbation_ok(name: str) -> bool:
    return name in ("none", "resize50_jpeg85") or bool(re.fullmatch(r"jpeg([1-9]\d?|100)", name))


def write_perturbed(src: str, name: str, tmpdir: Path) -> str:
    """Return a path to the (possibly re-encoded) image. 'none' returns the original path untouched."""
    from PIL import Image
    if name == "none":
        return src
    out = tmpdir / f"perturbed_{abs(hash((src, name)))}.jpg"
    with Image.open(src) as im:
        im = im.convert("RGB")
        if name == "resize50_jpeg85":
            im = im.resize((max(8, im.width // 2), max(8, im.height // 2)), Image.LANCZOS)
            im.save(out, "JPEG", quality=85)
        else:
            im.save(out, "JPEG", quality=int(name[4:]))
    return str(out)


# ============================================================ per-sample scoring (uses the REAL analyzers)
def _ev_score(ev) -> Optional[float]:
    return None if (not ev.present or ev.is_fallback) else float(ev.score)


def score_image_path(path: str, learned_on: bool) -> Dict[str, Optional[float]]:
    from backend.analyzers.image_detector import analyze_image
    with registry.disabled():
        heur = analyze_image(path)
    out = {"heuristic": _ev_score(heur), "learned": None, "combined": None}
    if learned_on:
        comb = analyze_image(path)
        out["combined"] = _ev_score(comb)
        p = comb.features.get("learned_fake_prob")
        out["learned"] = None if p is None else float(p)
    return out


def score_audio_path(path: str, learned_on: bool) -> Dict[str, Optional[float]]:
    from backend.analyzers.audio_detector import analyze_audio
    with registry.disabled():
        heur = analyze_audio(path)
    out = {"heuristic": _ev_score(heur), "learned": None, "combined": None}
    if learned_on:
        comb = analyze_audio(path)
        out["combined"] = _ev_score(comb)
        p = comb.features.get("learned_fake_prob")
        out["learned"] = None if p is None else float(p)
    return out


def to_pcm16_wav(src: str, tmpdir: Path) -> str:
    """
    Return a WAV path the analyzers can read. Ordinary 8/16/32-bit PCM WAVs are passed through UNTOUCHED, so the
    shipped detector sees exactly what a user upload would. Anything else (FLAC, MP3, float or 24-bit WAV) is
    decoded with soundfile to 16-bit mono PCM.
    """
    if Path(src).suffix.lower() == ".wav":
        try:
            with wave.open(src, "rb") as wf:
                if wf.getsampwidth() in (1, 2, 4):
                    return src
        except (wave.Error, EOFError):
            pass  # float / 24-bit / odd header: fall through to soundfile
    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("Non-WAV audio (e.g. ASVspoof FLAC) needs soundfile:  py -m pip install soundfile")
    data, sr = sf.read(src, dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype(np.int16) if data.shape[1] > 1 else data[:, 0]
    out = tmpdir / f"audio_{abs(hash(src))}.wav"
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(int(sr)); wf.writeframes(mono.tobytes())
    return str(out)


def score_clip_pair(image_path: str, caption: str, other_caption: str) -> Optional[Dict[str, float]]:
    from PIL import Image
    from backend.learned import clip_scorer
    with Image.open(image_path) as im:
        m = clip_scorer.image_text_cosine(im, [caption])
        x = clip_scorer.image_text_cosine(im, [other_caption])
    if m is None or x is None:
        return None
    return {"matched_cos": m, "mismatched_cos": x}


# ============================================================ run drivers
def _progress(i: int, n: int, t0: float) -> None:
    if i == n or i % 25 == 0:
        el = time.time() - t0
        print(f"  {i}/{n}  {el:.0f}s elapsed, ~{el / i * (n - i):.0f}s left", file=sys.stderr, flush=True)


def run_detector_task(task: str, samples: List[Sample], perturbs: List[str], learned_on: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    t0, total = time.time(), len(samples)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, s in enumerate(samples, 1):
            try:
                if task == "image":
                    for pert in perturbs:
                        path = write_perturbed(s.path, pert, tmp)
                        rows.append({**_meta(s), "condition": pert, "scores": score_image_path(path, learned_on)})
                else:
                    path = to_pcm16_wav(s.path, tmp)
                    rows.append({**_meta(s), "condition": s.condition or "all", "scores": score_audio_path(path, learned_on)})
            except SystemExit:
                raise
            except Exception as e:  # one bad file must not kill a long run
                print(f"  skipped {s.id}: {type(e).__name__}: {e}", file=sys.stderr)
                for pert in (perturbs if task == "image" else [s.condition or "all"]):
                    rows.append({**_meta(s), "condition": pert, "scores": {c: None for c in COLUMNS}})
            _progress(i, total, t0)
    return rows


def _meta(s: Sample) -> Dict[str, Any]:
    return {"id": s.id, "path": s.path, "label": s.label, "group": s.group}


def summarize_detector_rows(rows, thr, calib_ids, n_boot, seed, baseline_condition) -> Dict[str, Any]:
    conditions = sorted({r["condition"] for r in rows}, key=lambda c: (c != baseline_condition, c))
    res: Dict[str, Any] = {"conditions": {}, "groups": {}}
    for c in conditions:
        crow = [r for r in rows if r["condition"] == c]
        res["conditions"][c] = {col: evaluate_column(crow, col, thr, calib_ids, n_boot, seed) for col in COLUMNS}
    base = [r for r in rows if r["condition"] == baseline_condition] or rows
    for col in COLUMNS:
        g = group_breakdown(base, col, thr, calib_ids)
        if g:
            res["groups"][col] = g
    return res


def run_clip_task(samples: List[Sample], seed: int) -> Dict[str, Any]:
    samples = [s for s in samples if s.caption and Path(s.path).exists()]
    if len(samples) < 10:
        raise SystemExit("CLIP evaluation needs at least 10 images with captions (manifest columns: path,caption).")
    rng = random.Random(seed)
    n, shift = len(samples), rng.randint(1, len(samples) - 1)
    t0, pairs = time.time(), []
    for i, s in enumerate(samples, 1):
        other = samples[(i - 1 + shift) % n]
        if other.caption.strip().lower() == s.caption.strip().lower():
            continue
        r = score_clip_pair(s.path, s.caption, other.caption)
        if r is not None:
            pairs.append({"id": s.id, **r})
        _progress(i, n, t0)
    if not pairs:
        raise SystemExit("CLIP produced no scores (not cached, or load failed): " + str(registry.status()["clip"]))
    from backend.learned.clip_scorer import COS_MATCH, COS_MISMATCH, cosine_to_score
    try:
        from backend.verdict.engine import CLIP_MISMATCH_GUARD as guard
    except Exception:
        guard = 0.30
    m = np.array([p["matched_cos"] for p in pairs]); x = np.array([p["mismatched_cos"] for p in pairs])
    y = np.r_[np.ones(len(m)), np.zeros(len(x))]
    sc = lambda a: np.array([cosine_to_score(v) for v in a])
    q = lambda a: {k: float(np.percentile(a, p)) for k, p in (("p5", 5), ("p50", 50), ("p95", 95))}
    return {
        "n_pairs": len(pairs), "pairing": "each image vs its own caption (positive) and vs another image's caption (negative)",
        "auroc_matched_vs_mismatched": auroc(y, np.r_[m, x]),
        "cosine_quantiles": {"matched": q(m), "mismatched": q(x)},
        "current_anchors": {"COS_MISMATCH": COS_MISMATCH, "COS_MATCH": COS_MATCH, "guard_on_0_1_scale": guard},
        "guard_false_block_rate": float(np.mean(sc(m) < guard)),
        "guard_mismatch_catch_rate": float(np.mean(sc(x) < guard)),
        "suggested_anchors": {"COS_MATCH": float(np.percentile(m, 50)), "COS_MISMATCH": float(np.percentile(x, 95)),
                              "note": "starting points from this sample only; re-check on a second dataset before adopting"},
        "pairs": pairs,
    }


# ============================================================ reporting
def fmt(v: Any, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def render_markdown(report: Dict[str, Any]) -> str:
    ds = report["dataset"]
    size = (f"{ds['n_real']} images with captions" if report["task"] == "clip"
            else f"{ds['n_real']} real + {ds['n_fake']} fake samples (label 1 = fake/spoof)")
    L = [f"# Dataset evaluation: {report['task']} / {report['name']}", "",
         f"Run {report['run']['timestamp']} · seed {report['run']['seed']} · {size}", ""]
    for w in report["warnings"]:
        L.append(f"> **Note:** {w}")
    L.append("")
    L.append("## Models")
    for k, v in report["models"].items():
        L.append(f"- `{k}`: {v['model_id']} — {'cached' if v['cached'] else 'NOT cached'}"
                 + (f" (last error: {v['last_error']})" if v.get("last_error") else ""))
    L.append("")
    res = report["results"]
    if report["task"] == "clip":
        L += ["## CLIP image-caption agreement", "",
              f"- Pairs: {res['n_pairs']} ({res['pairing']})",
              f"- AUROC, matched vs mismatched: **{fmt(res['auroc_matched_vs_mismatched'])}**",
              f"- Cosine quantiles (p5 / p50 / p95): matched {res['cosine_quantiles']['matched']}, mismatched {res['cosine_quantiles']['mismatched']}",
              f"- Guard (score < {res['current_anchors']['guard_on_0_1_scale']}): wrongly blocks **{fmt(res['guard_false_block_rate'], True)}** of matching pairs, "
              f"catches **{fmt(res['guard_mismatch_catch_rate'], True)}** of mismatched pairs",
              f"- Current anchors {res['current_anchors']['COS_MISMATCH']}/{res['current_anchors']['COS_MATCH']}; "
              f"data-driven starting point: COS_MISMATCH={res['suggested_anchors']['COS_MISMATCH']:.3f}, COS_MATCH={res['suggested_anchors']['COS_MATCH']:.3f} "
              f"({res['suggested_anchors']['note']})", ""]
        return "\n".join(L)
    thr = report["run"]["threshold"]
    L += [f"## Results by condition (fixed threshold {thr} for accuracy / FPR / FNR)", "",
          "heuristic = shipped detector, learned OFF · learned = raw classifier P(fake) · combined = shipped detector, learned ON", ""]
    for cond, cols in res["conditions"].items():
        L += [f"### {cond}", "", "| score | AUROC [95% CI] | AP | EER | TPR@1%FPR | TPR@5%FPR | accuracy | FPR | FNR | scored / failed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for col, m in cols.items():
            if "auroc" not in m:
                L.append(f"| {col} | {m.get('note', '-')} | | | | | | | | 0 / {m['n_failed_or_unavailable']} |")
                continue
            ci = m.get("auroc_ci95")
            op = m["at_fixed_threshold"]
            L.append(f"| {col} | {fmt(m['auroc'])} [{fmt(ci[0]) if ci else '-'}, {fmt(ci[1]) if ci else '-'}] | {fmt(m['average_precision'])} | "
                     f"{fmt(m['eer'], True)} | {fmt(m['tpr_at_1pct_fpr'], True)} | {fmt(m['tpr_at_5pct_fpr'], True)} | "
                     f"{fmt(op['accuracy'], True)} | {fmt(op['fpr'], True)} | {fmt(1 - op['tpr'], True)} | {m['n']} / {m['n_failed_or_unavailable']} |")
            if "at_tuned_threshold" in m:
                t = m["at_tuned_threshold"]
                L.append(f"| {col} (threshold {t['threshold']:.3f} tuned on {m['tuned_on_n']} held-out calib samples) | | | | | | "
                         f"{fmt(t['accuracy'], True)} | {fmt(t['fpr'], True)} | {fmt(1 - t['tpr'], True)} | |")
        L.append("")
    if res["groups"]:
        L += ["## Per generator / attack (baseline condition)", ""]
        for col, groups in res["groups"].items():
            L += [f"**{col}**", "", "| group | n fake | detected @ threshold | AUROC vs reals |", "|---|---|---|---|"]
            for g, m in groups.items():
                L.append(f"| {g} | {m['n_fake']} | {fmt(m['detection_rate_at_threshold'], True)} | {fmt(m['auroc_vs_reals'])} |")
            L.append("")
    return "\n".join(L)


def write_outputs(report: Dict[str, Any], rows: Optional[List[Dict[str, Any]]], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"dataset_eval_{report['task']}_{report['name']}"
    paths = [out_dir / f"{stem}.json", out_dir / f"{stem}.md"]
    slim = json.loads(json.dumps(report, default=float))
    if report["task"] == "clip":
        slim["results"].pop("pairs", None)
    paths[0].write_text(json.dumps(slim, indent=2))
    paths[1].write_text(render_markdown(report))
    if rows:
        p = out_dir / f"{stem}_scores.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label", "group", "condition", *COLUMNS])
            for r in rows:
                w.writerow([r["id"], r["label"], r["group"] or "", r["condition"], *[r["scores"].get(c) for c in COLUMNS]])
        paths.append(p)
    elif report["task"] == "clip":
        p = out_dir / f"{stem}_scores.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "matched_cos", "mismatched_cos"])
            for r in report["results"]["pairs"]:
                w.writerow([r["id"], r["matched_cos"], r["mismatched_cos"]])
        paths.append(p)
    return paths


def build_warnings(task: str, learned_requested: bool, learned_on: bool, n_real: int, n_fake: int, calib_frac: float) -> List[str]:
    w = ["Check each model card for its training data and avoid evaluating on the same dataset: overlap inflates every number here."]
    if learned_requested and not learned_on:
        w.append("Learned model not available (not cached, or TRUSTLAYER_LEARNED=off): only the heuristic column is meaningful. "
                 "Run `python -m backend.learned.download` first.")
    if min(n_real, n_fake) < 100:
        w.append(f"Small sample ({n_real} real / {n_fake} fake): confidence intervals are wide; do not quote a single number.")
    if not calib_frac and task != "clip":
        w.append("Accuracy/FPR/FNR use a fixed threshold, not one tuned on this data. Threshold-free metrics (AUROC, EER) do not depend on it.")
    if task == "audio":
        w.append("The learned audio model scores the worst of up to four 6 s windows; ASVspoof-style short clips are usually one window.")
    return w


# ============================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["image", "audio", "clip"])
    ap.add_argument("--root", type=Path, help="dataset root (folders, or base for manifest/protocol paths)")
    ap.add_argument("--manifest", type=Path, help="CSV: path,label[,group,condition,caption]")
    ap.add_argument("--asvspoof-protocol", type=Path, help="ASVspoof protocol file (audio task); --root is the audio folder")
    ap.add_argument("--real-dirs", nargs="*", default=[], help="folder names that mean real (overrides auto-detection)")
    ap.add_argument("--fake-dirs", nargs="*", default=[], help="folder names that mean fake (overrides auto-detection)")
    ap.add_argument("--limit", type=int, default=0, help="max samples PER CLASS (0 = all); keeps the classes balanced")
    ap.add_argument("--perturb", nargs="*", default=["none"], help="image only: none, jpegNN, resize50_jpeg85")
    ap.add_argument("--threshold", type=float, default=0.5, help="fixed decision threshold for accuracy/FPR/FNR")
    ap.add_argument("--calib-frac", type=float, default=0.0, help="hold out this fraction to tune a threshold; metrics use the rest")
    ap.add_argument("--bootstrap", type=int, default=500, help="bootstrap resamples for the AUROC CI (0 = off)")
    ap.add_argument("--heuristic-only", action="store_true", help="skip the learned models")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", help="label for the output files (default: dataset folder name)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="list what would be evaluated, score nothing")
    return ap


def load_samples(args) -> Tuple[List[Sample], str]:
    exts = IMAGE_EXT if args.task in ("image", "clip") else AUDIO_EXT
    note = ""
    if args.task == "audio" and args.asvspoof_protocol:
        if not args.root:
            raise SystemExit("--asvspoof-protocol needs --root pointing at the audio folder")
        samples, missing = load_asvspoof(args.asvspoof_protocol, args.root)
        note = f"{missing} protocol entries had no matching audio file" if missing else ""
    elif args.manifest:
        samples, bad = load_manifest(args.manifest, args.root)
        have = [s for s in samples if Path(s.path).exists()]
        note = f"{len(samples) - len(have)} manifest rows point to missing files; {bad} rows unreadable"
        samples = have
    elif args.root:
        if args.task == "clip":
            raise SystemExit("clip task needs --manifest with path,caption")
        samples, unl = load_folder(args.root, exts, args.real_dirs, args.fake_dirs)
        note = f"{unl} files had no real/fake folder name (use --real-dirs/--fake-dirs or a manifest)" if unl else ""
    else:
        raise SystemExit("give --root (folders), --manifest, or --asvspoof-protocol")
    return samples, note


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    bad = [p for p in args.perturb if not perturbation_ok(p)]
    if bad:
        raise SystemExit(f"unknown --perturb value(s): {bad}")
    samples, note = load_samples(args)
    if note:
        print(f"note: {note}", file=sys.stderr)
    samples = balanced_sample(samples, args.limit, args.seed)
    n_real = sum(s.label == REAL for s in samples); n_fake = sum(s.label == FAKE for s in samples)
    groups: Dict[str, int] = {}
    for s in samples:
        if s.group:
            groups[s.group] = groups.get(s.group, 0) + 1
    print(f"{args.task}: {len(samples)} samples ({n_real} real, {n_fake} fake); groups: {groups or 'none'}")
    if args.task != "clip" and (n_real == 0 or n_fake == 0):
        raise SystemExit("need samples of BOTH classes to evaluate")
    if args.dry_run:
        return 0

    key = {"image": "image_deepfake", "audio": "audio_spoof", "clip": "clip"}[args.task]
    learned_on = (not args.heuristic_only) and registry.available(key)
    if args.task == "clip" and not learned_on:
        raise SystemExit(f"CLIP model not available: {registry.status()['clip']}. Run `python -m backend.learned.download --only clip`.")
    name = args.name or (args.root.name if args.root else args.manifest.stem)
    report: Dict[str, Any] = {
        "task": args.task, "name": name,
        "run": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": args.seed, "threshold": args.threshold,
                "calib_frac": args.calib_frac, "limit_per_class": args.limit, "perturbations": args.perturb if args.task == "image" else None,
                "learned_on": learned_on},
        "dataset": {"n_real": n_real, "n_fake": n_fake, "groups": groups, "root": str(args.root) if args.root else None},
        "models": {k: v for k, v in registry.status().items() if k == key},
        "warnings": build_warnings(args.task, not args.heuristic_only, learned_on, n_real, n_fake, args.calib_frac),
    }
    rows = None
    if args.task == "clip":
        report["results"] = run_clip_task(samples, args.seed)
    else:
        rows = run_detector_task(args.task, samples, args.perturb if args.task == "image" else [], learned_on)
        calib = assign_calibration_ids(samples, args.calib_frac, args.seed)
        baseline = "none" if args.task == "image" and "none" in args.perturb else None  # audio: pool every condition
        report["results"] = summarize_detector_rows(rows, args.threshold, calib, args.bootstrap, args.seed, baseline)
        report["models"] = {k: v for k, v in registry.status().items() if k == key}  # refresh: reflects load errors
    paths = write_outputs(report, rows, args.out_dir)
    print("\n" + render_markdown(report))
    print("\nwrote:\n  " + "\n  ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
