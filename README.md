# TrustLayer: Multimodal Digital Content Verification

TrustLayer analyses a **bundle** of artifacts (image, audio, one or more text sources, metadata) and decides whether it is
**`authentic`**, **`manipulated`**, **`coordinated_synthetic`** or **`insufficient_evidence`**. It reasons *across* artifacts,
shows the evidence behind every verdict, abstains when evidence is thin, and reports how well it generalises to manipulation
techniques it never saw in training.

FastAPI backend + Streamlit frontend. Runs fully local (optional local LLM via Ollama).

## What makes it different

| Capability | How it works | Where |
|---|---|---|
| **Cross-source reasoning** | Every text source is claim-extracted separately; dates are compared as calendar days, places and speakers with tolerant matching; EXIF date/location is checked against every source. Pure Python, no LLM in the decision. | `backend/claims/`, `backend/cross_modal/metadata_rules.py` |
| **Evidence graph** | Artifacts and the claims they assert as a graph; a claim shared by several artifacts is one node, contradictions are red edges. | `backend/claims/evidence_graph.py`, UI |
| **Counterfactuals** | "Without the audio the verdict would be ..." for each artifact, by re-running fusion with it removed. | `backend/verdict/counterfactual.py` |
| **Calibrated abstention** | XGBoost probabilities → temperature scaling (fitted by NLL) → split-conformal prediction set (90% target) → abstain if the set is ambiguous, entropy is high, or the only evidence is weak/fragile. | `backend/fusion/` |
| **"What would settle it?"** | On abstention (or thin evidence) the system lists the exact artifacts that would resolve it. | `backend/verdict/guidance.py` |
| **Generalisation evidence** | Technique-tagged synthetic benchmark; Leave-One-Technique-Out, Leave-One-Family-Out, ablations and laundering robustness. | `evaluation/` |

## Quick start

```bash
python -m venv venv
source venv/bin/activate            # Windows: .\venv\Scripts\activate
pip install -r backend/requirements.txt

uvicorn backend.main:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                      # terminal 2  -> http://localhost:8501
```

Optional but recommended for real content: download the learned detectors once (about 1 GB, needs internet):

```bash
python -m backend.learned.download          # image deepfake classifier, audio spoof classifier, CLIP
python -m backend.learned.download --check  # load each model, print how its labels were resolved, run a sanity pass
```

A trained model ships in `backend/fusion/artifacts/` (`xgb_bundle.json`, `calibration.json`). To retrain from scratch:

```bash
python -m evaluation.train --regen     # generates train/calib/test sets, trains XGBoost, fits temperature + conformal threshold
python -m evaluation.run_eval          # LOTO / LOFO / ablation / robustness -> evaluation/results/
```

Optional: `ollama pull qwen2.5:7b-instruct && ollama serve` for LLM claim extraction and a written report.
Without it the system uses a deterministic regex extractor and a template report, and says so (`model_info`).

## Learned detectors (optional, for real uploads)

| Key | Default model | Used for |
|---|---|---|
| `image_deepfake` | `Ateeqq/ai-vs-human-image-detector` | P(image is AI-generated), added to the image anomaly score |
| `audio_spoof` | `MelodyMachine/Deepfake-audio-detection-V2` (wav2vec2) | P(speech is synthetic), worst of up to 4 six-second windows |
| `clip` | `openai/clip-vit-base-patch32` | image-vs-text agreement (0-1) |

Override with `TRUSTLAYER_IMAGE_MODEL`, `TRUSTLAYER_AUDIO_MODEL`, `TRUSTLAYER_CLIP_MODEL` (repo id or local folder).
`TRUSTLAYER_LEARNED=off` disables all of them. Status: sidebar of the UI or `GET /api/models/status`.

How they are used, and why:

* **Additive only.** The heuristic detectors always run. A learned probability can raise an anomaly score; it can never clear one
  (probabilities at or below 0.5 count as "no evidence").
* **Labels are resolved from the model's own `id2label`** (e.g. `ai`/`hum`, `real`/`fake`, `bonafide`/`spoof`); if they are not
  recognisable the model is not used rather than guessed.
* **Never downloaded at request time**, and a model that fails to load is skipped, not fatal.
* **CLIP is evidence and a guard, not a model feature.** The fusion model was trained on synthetic bundles with no real CLIP
  signal, so the score is not fed to it. Instead a very low image-text agreement (< 0.30) stops a bundle from being called
  `authentic` (it abstains), and appears in the ranked evidence.
* **Synthetic demos and the benchmark run with learned models off** (they are procedurally generated, not real photos or voices),
  so results are reproducible on any machine. The UI says so.
* The learned scores have **not** been calibrated or evaluated on real data in this repo (see *Evaluating on real data*). Treat them as strong hints, not verdicts.

## Evaluating on real data

`evaluation/dataset_eval.py` scores the shipped detectors on real, labelled files and writes JSON, a Markdown report and a per-sample
CSV to `evaluation/results/`. Every sample is scored three ways: **heuristic** (learned models off), **learned** (the raw classifier
probability) and **combined** (what a user actually gets), so you can see what the learned model adds. Label 1 = fake/spoof.

```bash
pip install -r backend/requirements.txt          # includes soundfile, needed for FLAC audio
python -m backend.learned.download               # the learned models must be cached first

# images: folders named real/fake (or human/ai, REAL/FAKE...) anywhere under --root; --perturb tests re-compressed copies
python -m evaluation.dataset_eval image --root path/to/images --limit 500 --perturb none jpeg70 resize50_jpeg85

# audio: ASVspoof protocol file (2019 and 2021 layouts), or a CSV manifest with path,label[,group,condition]
python -m evaluation.dataset_eval audio --root path/to/flac_dir --asvspoof-protocol path/to/protocol.txt --limit 500
python -m evaluation.dataset_eval audio --root path/to/audio --manifest meta.csv --limit 500

# CLIP image-caption check: CSV with path,caption (own caption = match, another image's caption = mismatch)
python -m evaluation.dataset_eval clip --root path/to/images --manifest captions.csv --limit 500

# add --dry-run first to check the labels/groups it found without scoring anything
```

It reports AUROC with a bootstrap 95% CI, average precision, EER, TPR at 1% / 5% FPR, accuracy/FPR/FNR at a fixed threshold, and a
per-generator / per-attack breakdown. `--calib-frac 0.3` holds out part of the data to tune a threshold and reports on the rest.
For CLIP it reports how often the 0.30 guard wrongly blocks a matching caption and how many mismatches it catches, plus data-driven
suggestions for the `COS_MISMATCH` / `COS_MATCH` anchors. Check each model card's training data and do not evaluate on it.
No results from real data are committed yet: run it and paste the report into this section.

## Try it

* UI → sidebar **Demo Case Loader**: four ready-made bundles in `data/demos/` (see its README), or upload your own.
* CLI: `python -m evaluation.run_demos`
* API:

```bash
curl -X POST localhost:8000/api/bundle/analyze -H "Content-Type: application/json" -d '{
  "documents": {"message": "TechFest 2026 was held at Aditya Institute, Pune, on 15 September 2026.",
                "report":  "TechFest 2026 was held at Aditya Institute, Mumbai, on 15 September 2026."}}'
```

`/api/bundle/analyze-upload` (multipart) accepts `image`, `audio`, `document`, `extra_documents` (repeat), `caption`,
`metadata_json`, `claimed_speaker`.

## Tests

```bash
python -m pytest tests -q        # 45 tests: normalisation, cross-modal rules, conformal/temperature maths, abstention,
                                 # the four demo verdicts, API, learned-model plumbing (stubbed, no weights needed)
```

## Architecture

```
 bundle ─▶ Layer 1 per-modality detectors ─┐
                                           ├▶ Layer 3 fusion (XGBoost) ▶ temperature ▶ conformal ▶ abstention ▶ verdict
 bundle ─▶ Layer 2 cross-modal checks ─────┘                                                              │
                                                                       Layer 4: ranked evidence, evidence graph,
                                                                       counterfactuals, next steps, report
```

**Layer 1 (implemented).** Image: global ELA + FFT, a *local* sensor-noise consistency test (per-block robust outliers) whose
evidence bar rises as JPEG quality falls, plus the optional learned AI-image classifier. Audio: spectral centroid/rolloff/ZCR,
a splice detector (spectral discontinuity), plus the optional learned wav2vec2 spoof classifier. Text: stylometry heuristics.
Metadata: editing/AI-tool signatures, missing camera profile, timestamp gap.

**Layer 2 (implemented).** Cross-source contradictions (date/location/entity/speaker), metadata-vs-claims, claimed-speaker
vs. other sources, and image-text agreement via CLIP *when its weights are cached locally* (otherwise reported as skipped,
never guessed).

## Evaluation (synthetic benchmark)

Full tables: `evaluation/results/EVAL_REPORT.md`. Headline (each manipulation technique held out of training and calibration):

* Unseen-technique accuracy **82.7%**, macro-F1 **0.81**, AUROC **0.98**, **0%** false accusations of authentic bundles.
* All six cross-modal (coordinated) techniques: **100%** correct when held out.
* Why cross-modal reasoning matters: coordinated-class recall on unseen techniques is **0.38** with detectors alone and **1.00** with cross-modal features.
* Conformal prediction sets reach ~90% empirical coverage on the test split.

## Honest limitations

* **Synthetic data.** Images/audio in the benchmark are procedurally generated, so its numbers measure the fusion and
  cross-modal reasoning, not real-world deepfake detection. Evaluating the detectors needs real datasets.
* **Learned detectors are optional and not yet evaluated.** The image and audio classifiers are public Hugging Face models; they
  were integrated and unit-tested with stubs, and `evaluation/dataset_eval.py` is the harness for benchmarking them, but no
  real-data results are recorded in this repo yet. Public deepfake classifiers are
  known to generalise poorly to new generators and to re-compressed media. Without them the detectors are heuristics.
  They detect *AI-generated* media, not splicing or retouching of a real photo. There is no LM perplexity; the text detector is
  stylometry only.
* **Some unseen techniques fail.** Held out, `audio_splice` is 33% correct (60% wrong-confident) and `metadata_timestamp_gap`
  0% (93% wrong-confident). Holding out an entire *family* (audio or metadata) collapses the model — it cannot flag
  manipulation types for which no detector signal ever fired in training.
* **Conservative on weak image evidence.** To avoid accusing authentic content, a lone moderate image anomaly or a heavily
  re-compressed image leads to abstention, so image-manipulation recall drops sharply on laundered images.
* Speaker verification is a placeholder (no speaker embedding model yet), so a speaker mismatch is detected only through
  conflicting *claims*, not by listening to the audio.
