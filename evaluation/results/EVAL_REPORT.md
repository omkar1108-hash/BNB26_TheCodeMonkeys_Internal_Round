# TrustLayer evaluation report

> Synthetic, technique-tagged benchmark (procedurally generated artifacts). Numbers measure fusion and cross-modal reasoning, not real-world forensic accuracy.

## 1. Leave-One-Technique-Out (unseen manipulation techniques)

Pooled over 345 held-out bundles: accuracy **0.8348**, macro-F1 **0.828**, AUROC (OvR) **0.9769**, ECE **0.0842**, authentic false-alarm rate **0.0**.

| Technique | Class | Seen: correct | Unseen: correct | Unseen: abstained | Unseen: wrong-confident |
|---|---|---|---|---|---|
| image_noise_patch | manipulated | 80% | 73% | 20% | 7% |
| image_double_jpeg | manipulated | 27% | 27% | 40% | 33% |
| image_blur_patch | manipulated | 27% | 27% | 47% | 27% |
| audio_vocoder_noise | manipulated | 27% | 13% | 67% | 20% |
| audio_bandlimit | manipulated | 100% | 100% | 0% | 0% |
| audio_splice | manipulated | 100% | 100% | 0% | 0% |
| metadata_editor_tag | manipulated | 100% | 100% | 0% | 0% |
| metadata_ai_tag | manipulated | 100% | 100% | 0% | 0% |
| metadata_timestamp_gap | manipulated | 100% | 100% | 0% | 0% |
| coord_date_shift | coordinated_synthetic | 73% | 73% | 13% | 13% |
| coord_location_swap | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_speaker_swap | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_cross_doc_date | coordinated_synthetic | 93% | 93% | 0% | 7% |
| coord_cross_doc_location | coordinated_synthetic | 100% | 100% | 0% | 0% |
| coord_date_and_location | coordinated_synthetic | 100% | 100% | 0% | 0% |

## 2. Leave-One-Family-Out (stress test)

| Held-out family | Correct | Abstained | Wrong-confident |
|---|---|---|---|
| image | 0% | 38% | 62% |
| audio | 31% | 22% | 47% |
| metadata | 9% | 47% | 44% |

## 3. Ablation: does reasoning across modalities help?

| Model | Seen macro-F1 | Seen coordinated recall | LOTO macro-F1 | LOTO coordinated recall |
|---|---|---|---|---|
| image only | 0.316 | 0.211 | 0.286 | 0.178 |
| audio only | 0.349 | 0.833 | 0.314 | 0.833 |
| text only | 0.449 | 0.222 | 0.449 | 0.222 |
| metadata only | 0.394 | 0.567 | 0.35 | 0.567 |
| all detectors (no cross-modal) | 0.784 | 0.678 | 0.649 | 0.433 |
| detectors + cross-modal, no aggregate features | 0.931 | 0.956 | 0.822 | 0.778 |
| detectors + cross-modal + aggregates (full) | 0.929 | 0.956 | 0.903 | 0.944 |

## 4. Robustness to image laundering

| Transform | Accuracy (image bundles) | Image-manipulation recall | Image-manipulation abstained | Authentic accepted | Authentic abstained | Authentic falsely accused |
|---|---|---|---|---|---|---|
| none | 69% | 44% | 49% | 57% | 40% | 2% |
| jpeg_q50 | 57% | 13% | 67% | 38% | 60% | 2% |
| resize_half | 66% | 44% | 44% | 50% | 42% | 8% |
| screenshot_like | 56% | 11% | 76% | 38% | 60% | 2% |
