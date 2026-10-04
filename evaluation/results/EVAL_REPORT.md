# TrustLayer evaluation report

> Synthetic, technique-tagged benchmark (procedurally generated artifacts). Numbers measure fusion and cross-modal reasoning, not real-world forensic accuracy.

## 1. Leave-One-Technique-Out (unseen manipulation techniques)

Pooled over 330 held-out bundles: accuracy **0.8273**, macro-F1 **0.8145**, AUROC (OvR) **0.9772**, ECE **0.1468**, authentic false-alarm rate **0.0**.

| Technique | Class | Seen: correct | Unseen: correct | Unseen: abstained | Unseen: wrong-confident |
|---|---|---|---|---|---|
| image_noise_patch | manipulated | 73% | 73% | 27% | 0% |
| image_double_jpeg | manipulated | 60% | 60% | 40% | 0% |
| image_blur_patch | manipulated | 47% | 40% | 40% | 20% |
| audio_vocoder_noise | manipulated | 67% | 67% | 33% | 0% |
| audio_bandlimit | manipulated | 100% | 100% | 0% | 0% |
| audio_splice | manipulated | 47% | 33% | 7% | 60% |
| metadata_editor_tag | manipulated | 100% | 100% | 0% | 0% |
| metadata_ai_tag | manipulated | 100% | 100% | 0% | 0% |
| metadata_timestamp_gap | manipulated | 73% | 0% | 7% | 93% |
| coord_date_shift | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_location_swap | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_speaker_swap | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_cross_doc_date | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_cross_doc_location | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_date_and_location | coordinated_synthetic | 100% | 100% | 0% | 0% |

## 2. Leave-One-Family-Out (stress test)

| Held-out family | Correct | Abstained | Wrong-confident |
|---|---|---|---|
| image | 38% | 29% | 33% |
| audio | 2% | 18% | 80% |
| metadata | 0% | 11% | 89% |

## 3. Ablation: does reasoning across modalities help?

| Model | Seen macro-F1 | Seen coordinated recall | LOTO macro-F1 | LOTO coordinated recall |
|---|---|---|---|---|
| image only | 0.303 | 0.1 | 0.299 | 0.1 |
| audio only | 0.294 | 0.833 | 0.248 | 0.833 |
| text only | 0.42 | 0.122 | 0.416 | 0.111 |
| metadata only | 0.423 | 0.556 | 0.277 | 0.056 |
| all detectors (no cross-modal) | 0.749 | 0.6 | 0.614 | 0.378 |
| detectors + cross-modal, no aggregate features | 0.95 | 1.0 | 0.83 | 0.833 |
| detectors + cross-modal + aggregates (full) | 0.96 | 1.0 | 0.896 | 1.0 |

## 4. Robustness to image laundering

| Transform | Accuracy (image bundles) | Image-manipulation recall | Image-manipulation abstained | Authentic accepted | Authentic abstained | Authentic falsely accused |
|---|---|---|---|---|---|---|
| none | 76% | 60% | 38% | 68% | 32% | 0% |
| jpeg_q50 | 57% | 11% | 84% | 28% | 64% | 8% |
| resize_half | 71% | 40% | 44% | 68% | 28% | 4% |
| screenshot_like | 57% | 11% | 82% | 28% | 72% | 0% |
