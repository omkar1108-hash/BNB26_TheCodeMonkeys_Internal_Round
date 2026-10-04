"""
Generalisation evaluation.

    python -m evaluation.run_eval

Experiments (all on the synthetic benchmark, with the fusion model retrained per fold):
  1. LOTO  - Leave-One-Technique-Out: each manipulation technique is removed from training
             and calibration, then evaluated on fresh bundles of that technique.
  2. LOFO  - Leave-One-Family-Out: all image / audio / metadata / cross-modal techniques held out.
  3. Ablation - single modality vs. detectors-only fusion vs. fusion + cross-modal checks.
  4. Robustness - re-run after 'laundering' images (re-compression, resize, screenshot-like).
Results are written to evaluation/results/eval_results.json and EVAL_REPORT.md.
"""
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from backend.config import TARGET_CLASSES, DATASETS_DIR
from backend.fusion.calibration import TemperatureScaler
from backend.fusion.abstention import evaluate_abstention, conformal_prediction_set
from backend.fusion.decision import decide, reliability_guard, ood_guard
from backend.fusion.feature_builder import FEATURE_INDEX, MODALITY_COLUMNS, build_fusion_features
from backend.learned import registry
from backend.verdict.engine import compute_bundle_signals
from data.generator.bundle_generator import TECHNIQUES, record_to_bundle
from evaluation.features import ensure_datasets, Table
from evaluation.metrics import compute_ece, compute_coverage_vs_accuracy
from evaluation.train import fit_xgb, proba_matrix, calibrate

RESULTS_DIR = Path(__file__).resolve().parent / "results"
INSUFF = TARGET_CLASSES.index("insufficient_evidence")
COORD = TARGET_CLASSES.index("coordinated_synthetic")


def decisions(P_cal: np.ndarray, X: np.ndarray, qhat) -> List[Tuple[str, float]]:
    """Apply the same abstention policy as the live system. Returns (final_label, confidence)."""
    out = []
    for p, x in zip(P_cal, X):
        probs = {c: float(p[i]) for i, c in enumerate(TARGET_CLASSES)}
        total = float(x[FEATURE_INDEX["total_modalities_present"]]) + max(0.0, float(x[FEATURE_INDEX["n_text_sources"]]) - 1.0)
        ab, _, label, conf = evaluate_abstention(probs, total, conformal_set=conformal_prediction_set(probs, qhat))
        if not ab and reliability_guard(x, label):
            label = "insufficient_evidence"
        elif not ab and ood_guard(x):
            label = "insufficient_evidence"
        out.append((label, conf))
    return out


def outcome(true_idx: int, label: str) -> str:
    true = TARGET_CLASSES[true_idx]
    if label == true:
        return "correct"
    if label == "insufficient_evidence":
        return "abstained"
    return "wrong"


def run_fold(tr: Table, ca: Table, te: Table, train_mask, calib_mask, test_mask):
    model = fit_xgb(tr.X[train_mask], tr.y[train_mask])
    T, qhat = calibrate(model, ca.X[calib_mask], ca.y[calib_mask])
    Xt, yt = te.X[test_mask], te.y[test_mask]
    P = TemperatureScaler(T).calibrate_matrix(proba_matrix(model, Xt))
    dec = decisions(P, Xt, qhat)
    return yt, P, dec


def summarise(yt, dec) -> Dict[str, float]:
    oc = [outcome(int(y), d[0]) for y, d in zip(yt, dec)]
    n = max(len(oc), 1)
    return {
        "n": len(oc),
        "correct": round(oc.count("correct") / n, 3),
        "abstained": round(oc.count("abstained") / n, 3),
        "wrong_confident": round(oc.count("wrong") / n, 3),
    }


MANIP_TECHNIQUES = [t for t, (lab, fam) in TECHNIQUES.items() if fam in ("image", "audio", "metadata", "cross_modal")]


def fold_test_masks(te: Table):
    """(technique, mask of test bundles scored by that fold's model). Held-out technique bundles
    plus a round-robin share of the benign (authentic/sparse) bundles, each scored exactly once."""
    benign_idx = np.where(np.isin(te.families, ["authentic", "sparse"]))[0]
    out = []
    for k, tech in enumerate(MANIP_TECHNIQUES):
        m = te.techniques == tech
        rr = np.zeros(len(te.y), bool)
        rr[benign_idx[k::len(MANIP_TECHNIQUES)]] = True
        out.append((tech, m, m | rr))
    return out


def loto(tr: Table, ca: Table, te: Table) -> Dict:
    """
    Leave-One-Technique-Out over every manipulation technique. Authentic and sparse bundles are
    never held out (they are not manipulation techniques). Each authentic / insufficient test
    bundle is scored exactly once, round-robin, by one of the fold models, so the pooled metrics
    cover the whole test set including false alarms on authentic content.
    """
    rows = {}
    pooled_y, pooled_P, pooled_pred, pooled_conf = [], [], [], []
    benign_idx = np.where(np.isin(te.families, ["authentic", "sparse"]))[0]
    for k, tech in enumerate(MANIP_TECHNIQUES):
        seen_y, _, seen_dec = run_fold(tr, ca, te, np.ones(len(tr.y), bool), np.ones(len(ca.y), bool), te.techniques == tech)
        train_mask, calib_mask = tr.techniques != tech, ca.techniques != tech
        test_mask = (te.techniques == tech)
        rr = np.zeros(len(te.y), bool)
        rr[benign_idx[k::len(MANIP_TECHNIQUES)]] = True
        un_y, un_P, un_dec = run_fold(tr, ca, te, train_mask, calib_mask, test_mask | rr)
        sel = np.where(test_mask[(test_mask | rr)])[0]
        rows[tech] = {
            "label": TECHNIQUES[tech][0], "family": TECHNIQUES[tech][1],
            "seen": summarise(seen_y, seen_dec),
            "unseen": summarise(un_y[sel], [un_dec[i] for i in sel]),
        }
        pooled_y += list(un_y)
        pooled_P += list(un_P)
        pooled_pred += [TARGET_CLASSES.index(d[0]) for d in un_dec]
        pooled_conf += [d[1] for d in un_dec]
    y, P = np.array(pooled_y), np.array(pooled_P)
    pred, conf = np.array(pooled_pred), np.array(pooled_conf)
    correct = (pred == y).astype(float)
    try:
        auroc = float(roc_auc_score(label_binarize(y, classes=range(4)), P, multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")
    auth = y == TARGET_CLASSES.index("authentic")
    return {
        "per_technique": rows,
        "pooled_unseen": {
            "n": int(len(y)),
            "accuracy": round(float(correct.mean()), 4),
            "macro_f1": round(float(f1_score(y, pred, labels=range(4), average="macro", zero_division=0)), 4),
            "auroc_ovr": round(auroc, 4),
            "ece": round(compute_ece(conf, correct), 4),
            "authentic_false_alarm_rate": round(float(np.mean(np.isin(pred[auth], [1, 2]))), 4),
            "coverage_vs_accuracy": compute_coverage_vs_accuracy(conf, correct, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            "per_class_recall": {
                TARGET_CLASSES[i]: round(float(recall_score(y == i, pred == i, zero_division=0)), 3) for i in range(4)
            },
        },
    }


def lofo(tr: Table, ca: Table, te: Table) -> Dict:
    out = {}
    for fam in ["image", "audio", "metadata"]:
        y, _, dec = run_fold(tr, ca, te, tr.families != fam, ca.families != fam, te.families == fam)
        out[fam] = summarise(y, dec)
    return out


def ablation(tr: Table, te: Table) -> Dict:
    cols = lambda names: [FEATURE_INDEX[n] for n in names]
    det_cols = cols(sum(MODALITY_COLUMNS.values(), [])) + cols(["total_modalities_present"])
    sets = {
        "image only": cols(MODALITY_COLUMNS["image"]),
        "audio only": cols(MODALITY_COLUMNS["audio"]),
        "text only": cols(MODALITY_COLUMNS["text"]),
        "metadata only": cols(MODALITY_COLUMNS["metadata"]),
        "all detectors (no cross-modal)": det_cols,
        "detectors + cross-modal, no aggregate features": [
            i for n, i in FEATURE_INDEX.items()
            if n not in (
                "max_modality_anomaly", "n_anomalous_modalities", "total_conflicts", "image_reliability",
                "mean_modality_anomaly", "anomaly_spread", "top1_top2_margin",
                "n_detectors_above_30", "n_detectors_above_60", "authentic_centroid_distance"
            )
        ],
        "detectors + cross-modal + aggregates (full)": list(range(tr.X.shape[1])),
    }
    out = {}
    for name, columns in sets.items():
        # seen split
        m = fit_xgb(tr.X, tr.y, columns)
        pred = proba_matrix(m, te.X, columns).argmax(1)
        # pooled LOTO (argmax)
        lp, ly = [], []
        for tech, _held, mask in fold_test_masks(te):
            mm = fit_xgb(tr.X[tr.techniques != tech], tr.y[tr.techniques != tech], columns)
            lp += list(proba_matrix(mm, te.X[mask], columns).argmax(1))
            ly += list(te.y[mask])
        lp, ly = np.array(lp), np.array(ly)
        out[name] = {
            "seen_macro_f1": round(float(f1_score(te.y, pred, average="macro", zero_division=0)), 3),
            "seen_coordinated_recall": round(float(recall_score(te.y == COORD, pred == COORD, zero_division=0)), 3),
            "loto_macro_f1": round(float(f1_score(ly, lp, average="macro", zero_division=0)), 3),
            "loto_coordinated_recall": round(float(recall_score(ly == COORD, lp == COORD, zero_division=0)), 3),
        }
    return out


def launder(src: Path, dst: Path, mode: str):
    img = Image.open(src).convert("RGB")
    if mode == "jpeg_q50":
        img.save(dst, "JPEG", quality=50)
    elif mode == "resize_half":
        img.resize((128, 128)).resize((256, 256)).save(dst, "JPEG", quality=92)
    elif mode == "screenshot_like":
        img.resize((205, 205)).filter(ImageFilter.GaussianBlur(0.6)).resize((256, 256)).save(dst, "JPEG", quality=70)


def robustness() -> Dict:
    d = DATASETS_DIR / "synthetic_test"
    recs = [r for r in json.loads((d / "manifest.json").read_text()) if r["image_path"]]
    out = {}
    tmp = Path(tempfile.mkdtemp())
    try:
        for mode in ["none", "jpeg_q50", "resize_half", "screenshot_like"]:
            ok, n, img_manip_ok, img_manip_n, auth_ok, auth_n, auth_abst, auth_acc = 0, 0, 0, 0, 0, 0, 0, 0
            img_manip_abst = 0
            for r in recs:
                b = record_to_bundle(r, d)
                if mode != "none":
                    dst = tmp / f"{r['bundle_id']}_{mode}.jpg"
                    launder(Path(b.image_path), dst, mode)
                    b.image_path = str(dst)
                with registry.disabled():
                    sig = compute_bundle_signals(b)
                dec = decide(build_fusion_features(sig["modality_results"], sig["cross_modal"]))
                good = dec.label == r["label"]
                ok += good
                n += 1
                if r["family"] == "image":
                    img_manip_ok += good
                    img_manip_abst += dec.label == "insufficient_evidence"
                    img_manip_n += 1
                if r["label"] == "authentic":
                    auth_ok += good
                    auth_abst += dec.label == "insufficient_evidence"
                    auth_acc += dec.label in ("manipulated", "coordinated_synthetic")
                    auth_n += 1
            out[mode] = {
                "accuracy_on_image_bundles": round(ok / n, 3),
                "image_manipulation_recall": round(img_manip_ok / max(img_manip_n, 1), 3),
                "image_manipulation_abstained": round(img_manip_abst / max(img_manip_n, 1), 3),
                "authentic_accepted": round(auth_ok / max(auth_n, 1), 3),
                "authentic_abstained": round(auth_abst / max(auth_n, 1), 3),
                "authentic_falsely_accused": round(auth_acc / max(auth_n, 1), 3),
                "n": n,
            }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def markdown(res: Dict) -> str:
    L = ["# TrustLayer evaluation report", "",
         "> Synthetic, technique-tagged benchmark (procedurally generated artifacts). "
         "Numbers measure fusion and cross-modal reasoning, not real-world forensic accuracy.", ""]
    p = res["loto"]["pooled_unseen"]
    L += ["## 1. Leave-One-Technique-Out (unseen manipulation techniques)", "",
          f"Pooled over {p['n']} held-out bundles: accuracy **{p['accuracy']}**, macro-F1 **{p['macro_f1']}**, "
          f"AUROC (OvR) **{p['auroc_ovr']}**, ECE **{p['ece']}**, authentic false-alarm rate **{p['authentic_false_alarm_rate']}**.", "",
          "| Technique | Class | Seen: correct | Unseen: correct | Unseen: abstained | Unseen: wrong-confident |",
          "|---|---|---|---|---|---|"]
    for t, r in res["loto"]["per_technique"].items():
        L.append(f"| {t} | {r['label']} | {r['seen']['correct']:.0%} | {r['unseen']['correct']:.0%} | "
                 f"{r['unseen']['abstained']:.0%} | {r['unseen']['wrong_confident']:.0%} |")
    L += ["", "## 2. Leave-One-Family-Out (stress test)", "",
          "| Held-out family | Correct | Abstained | Wrong-confident |", "|---|---|---|---|"]
    for f, r in res["lofo"].items():
        L.append(f"| {f} | {r['correct']:.0%} | {r['abstained']:.0%} | {r['wrong_confident']:.0%} |")
    L += ["", "## 3. Ablation: does reasoning across modalities help?", "",
          "| Model | Seen macro-F1 | Seen coordinated recall | LOTO macro-F1 | LOTO coordinated recall |", "|---|---|---|---|---|"]
    for n, r in res["ablation"].items():
        L.append(f"| {n} | {r['seen_macro_f1']} | {r['seen_coordinated_recall']} | {r['loto_macro_f1']} | {r['loto_coordinated_recall']} |")
    L += ["", "## 4. Robustness to image laundering", "",
          "| Transform | Accuracy (image bundles) | Image-manipulation recall | Image-manipulation abstained | Authentic accepted | Authentic abstained | Authentic falsely accused |",
          "|---|---|---|---|---|---|---|"]
    for m, r in res["robustness"].items():
        L.append(f"| {m} | {r['accuracy_on_image_bundles']:.0%} | {r['image_manipulation_recall']:.0%} | {r['image_manipulation_abstained']:.0%} | "
                 f"{r['authentic_accepted']:.0%} | {r['authentic_abstained']:.0%} | {r['authentic_falsely_accused']:.0%} |")
    return "\n".join(L) + "\n"


def main():
    tables = ensure_datasets(DATASETS_DIR)
    tr, ca, te = tables["train"], tables["calib"], tables["test"]
    res = {
        "loto": loto(tr, ca, te),
        "lofo": lofo(tr, ca, te),
        "ablation": ablation(tr, te),
        "robustness": robustness(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eval_results.json").write_text(json.dumps(res, indent=2))
    md = markdown(res)
    (RESULTS_DIR / "EVAL_REPORT.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
