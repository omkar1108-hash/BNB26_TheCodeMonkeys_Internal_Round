# TrustLayer: Multimodal Digital Content Verification System

TrustLayer is an end-to-end multimodal verification framework and web application (FastAPI backend + Streamlit frontend) designed to analyze content **bundles** consisting of **Images**, **Captions/Text**, **Audio**, and **Metadata**.

The system detects single-artifact manipulations as well as complex **coordinated synthetic disinformation campaigns** where individual artifacts appear clean in isolation but are mutually contradictory across modalities.

---

## 🎯 Target Taxonomy

TrustLayer classifies incoming multimodal bundles into one of four calibrated verdicts:

1. **`authentic`**: All submitted artifacts pass individual authenticity checks and exhibit mutual cross-modal consistency.
2. **`manipulated`**: At least one individual artifact has been altered, spliced, or synthetically generated (e.g., audio spoof or edited image).
3. **`coordinated_synthetic`**: Individual artifacts pass standalone quality checks, but cross-modal verification reveals factual, temporal, spatial, or contextual contradictions.
4. **`insufficient_evidence`**: Evidence is incomplete, ambiguous, or the system abstains due to low prediction confidence or high prediction entropy.

---

## 🏛️ Four-Layer Architecture

```
                  ┌────────────────────────────────────────┐
                  │    Incoming Multimodal Bundle          │
                  │ (Image + Text/Caption + Audio + Meta)  │
                  └───────────────────┬────────────────────┘
                                      │
   ┌──────────────────────────────────┴──────────────────────────────────┐
   ▼                                                                     ▼
[ Layer 1: Per-Modality Detectors ]                  [ Layer 2: Cross-Modal Verification ]
 • Image: CLIP + ELA + 2D FFT Frequencies             • Image-Caption: CLIP Cosine Similarity
 • Audio: Wav2Vec2 Spoof Head + Spectral Features     • Audio-Speaker: Embedding vs. Claimed Identity
 • Text: Perplexity + Stylometry (TTR, Entropy)       • Metadata Rules: EXIF Timestamps vs Text Dates
 • Metadata: EXIF Tampering & Software Signatures     • Claims: Ollama JSON Extractor + Python Conflict Engine
   └──────────────────────────────────┬──────────────────────────────────┘
                                      │
                                      ▼
                      [ Layer 3: Fusion & Calibration ]
                       • Missing-Modality Indicator Flags (never crash)
                       • XGBoost 4-Class Probabilistic Classifier
                       • Temperature / Platt Scaling Calibration
                       • Conformal & Threshold-Based Abstention
                                      │
                                      ▼
                    [ Layer 4: Explanations & Reporting ]
                       • Ranked Evidence List (by anomaly contribution)
                       • Grounded Ollama LLM Analytical Report
                       • Automated Fact Verification & Labeled Fallback
                                      │
                                      ▼
                    [ Web UI: FastAPI + Streamlit ]
```

### Layer 1: Per-Modality Detectors
Each detector extracts dense features and returns a normalized anomaly score $[0, 1]$, evidence string, and fallback tracking flags:
- **Image**: Error Level Analysis (ELA) compression differentials, 2D Fast Fourier Transform (FFT) high-frequency spectrum artifacts, and CLIP vision embeddings.
- **Audio**: Wav2Vec2 spoof classifier head combined with spectral centroid, spectral rolloff, and zero-crossing rate analysis.
- **Text**: Causal LM perplexity scoring along with stylometric markers (Type-Token Ratio, sentence length variance, punctuation entropy).
- **Metadata**: EXIF parsing via `ExifRead`/`Pillow`, flagging editing software tags (e.g., Photoshop, GIMP), timestamp anomalies, and missing camera hardware signatures.

### Layer 2: Cross-Modal Verifiers
- **CLIP Cross-Modal Similarity**: Evaluates semantic alignment between visual elements and descriptive text/captions.
- **Speaker vs. Claimed Identity**: Verifies acoustic speaker embeddings against claimed speaker identities or verified reference profiles.
- **Metadata Consistency Rules**: Flags discrepancies between EXIF capture timestamps/locations and dates/locations claimed in text.
- **Ollama Claim & Contradiction Engine**: Uses local Ollama (`qwen2.5:7b-instruct`) with strict JSON schema enforcement (`format`) to extract short factual fields (`entities`, `dates`, `locations`, `claimed_speaker`). Cross-claim contradictions are then evaluated deterministically in pure Python to eliminate LLM hallucination risk.

### Layer 3: Multimodal Fusion, Calibration & Abstention
- **Missing Modality Handling**: Missing artifacts are encoded via binary presence indicator flags (`has_image`, `has_audio`, etc.) and zero-imputed feature vectors. Missing modalities lower decision confidence gracefully rather than causing system failure.
- **XGBoost 4-Class Classifier**: Trained on concatenated per-modality scores, cross-modal distances, missing modality flags, and contradiction metrics.
- **Probabilistic Calibration**: Temperature scaling / Platt scaling ensures output probabilities correspond to true empirical accuracy (minimizing Expected Calibration Error).
- **Abstention Engine**: Triggers abstention to `"insufficient_evidence"` when maximum class confidence falls below a tuned threshold or when prediction entropy exceeds safety bounds.

### Layer 4: Explanations & Grounded Reporting
- **Evidence Ranking**: Orders all observed anomalies and contradictions by feature importance and deviation severity.
- **Grounded LLM Report**: Synthesizes structured findings into an executive intelligence report using local Ollama. The report is verified against input findings; if unsupported assertions are detected or Ollama is unavailable, the system safely falls back to a deterministic, labeled template.

---

## ⚙️ Hardware Policy & Fallback Integrity

- **CPU-First with Auto-CUDA Detection**: Seamlessly detects and leverages NVIDIA CUDA GPUs when available; defaults smoothly to CPU execution without requiring code changes.
- **Labeled Fallbacks**: Any heuristic or mock fallback triggered due to missing optional dependencies or unavailable model weights is explicitly tagged with `is_fallback: true` and a detailed `fallback_reason`.
- **Evaluation Integrity**: All benchmark evaluation routines strictly exclude records flagged with `is_fallback: true` to guarantee scientific rigor and honest metrics.

---

## 📂 Repository Layout

```
.
├── backend/
│   ├── main.py                  # FastAPI application entrypoint
│   ├── config.py                # Environment, paths, and hardware detection
│   ├── requirements.txt         # Core dependencies
│   ├── analyzers/               # Layer 1 per-modality detectors
│   │   ├── file_detector.py     # Modality extension resolver
│   │   ├── document_analyzer.py # PDF / DOCX / TXT extractors
│   │   ├── image_detector.py    # ELA, FFT, CLIP analyzer (Layer 1)
│   │   ├── audio_detector.py    # Wav2Vec2 spoof & spectral analyzer (Layer 1)
│   │   ├── text_detector.py     # Perplexity & stylometry analyzer (Layer 1)
│   │   └── metadata_detector.py # EXIF & timestamp validator (Layer 1)
│   ├── claims/                  # Layer 2 claim extraction & conflict analysis
│   │   ├── extractor.py         # Structured claim extraction
│   │   ├── conflict.py          # Deterministic cross-claim conflict detection
│   │   └── evidence_graph.py    # Evidence graph construction
│   ├── cross_modal/             # Layer 2 cross-modal verification engines
│   │   ├── image_text.py        # CLIP image-caption consistency
│   │   ├── audio_speaker.py     # Speaker embedding verification
│   │   └── metadata_rules.py    # Cross-modal metadata conflict rules
│   ├── fusion/                  # Layer 3 feature builder, classifier & calibration
│   │   ├── feature_builder.py   # Modality vectorization & missing-flag handling
│   │   ├── classifier.py        # 4-class XGBoost classifier
│   │   ├── calibration.py       # Temperature scaling
│   │   └── abstention.py        # Conformal & threshold abstention
│   ├── models/                  # Pydantic data models & contracts
│   │   └── bundle.py            # Bundle, ModalityEvidence, and VerdictResult
│   └── verdict/                 # Layer 4 explainer & reporting
│       ├── engine.py            # Verdict aggregation
│       ├── explainer.py         # Evidence ranking engine
│       └── llm_report.py        # Verified Ollama report generator
├── frontend/
│   └── app.py                   # Streamlit web interface
├── data/
│   ├── demos/                   # Demo cases (authentic, manipulated, coordinated)
│   ├── datasets/                # Benchmark dataset storage root
│   └── generator/               # Synthetic coordinated bundle generator
├── evaluation/
│   └── eval_suite.py            # LOGO/LOTO splits, F1, AUROC, ECE, Ablation
└── schemas/
    └── analysis_schema.json     # JSON Schema validation contract
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/omkar1108-hash/BNB26_TheCodeMonkeys_Internal_Round.git
cd BNB26_TheCodeMonkeys_Internal_Round

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 2. Configure Local Ollama (Optional for LLM Claims & Reports)

TrustLayer uses a local Ollama instance for structured claim extraction and grounded reporting:

```bash
# Pull the recommended model
ollama pull qwen2.5:7b-instruct

# Start Ollama (defaults to http://localhost:11434)
ollama serve
```

Configure environment variables if using custom endpoints or models:

```bash
export OLLAMA_HOST="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5:7b-instruct"
export DATASET_ROOT="./data/datasets"
```

### 3. Running the FastAPI Backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit the interactive OpenAPI documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### 4. Running the Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501) to interact with the multimodal upload interface.

---

## 🧪 Testing & Verification

Run the test suite across claims, analyzers, and evidence graph modules:

```bash
python backend/claims/test_extractor.py
python backend/claims/test_conflict.py
python backend/claims/test_evidence_graph.py
python backend/analyzers/test_detector.py
```

---

## 📊 Evaluation & Benchmarking

The evaluation framework supports rigorous out-of-distribution and ablation experiments:
- **Evaluation Splits**:
  - **LOGO (Leave-One-Generator-Out)**: Tests generalization across unseen generative models (e.g., Midjourney vs. Stable Diffusion vs. DALL-E).
  - **LOTO (Leave-One-Technique-Out)**: Tests robustness when entire manipulation classes (e.g., voice cloning or face morphing) are held out during training.
- **Evaluation Metrics**:
  - Multiclass Macro-F1 & Confusion Matrix
  - Area Under ROC (One-vs-Rest AUROC)
  - Expected Calibration Error (ECE)
  - Coverage vs. Accuracy curves under varying abstention thresholds
  - Full Ablation Matrix: Single Modality vs. Multimodal Fusion vs. Fusion + Cross-Modal Checks.

---

## 📄 License & Attribution

Developed by **The Code Monkeys** for the **BNB26 Internal Round**. Released under the MIT License.
