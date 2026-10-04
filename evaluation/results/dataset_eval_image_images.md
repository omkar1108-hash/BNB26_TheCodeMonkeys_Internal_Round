# Dataset evaluation: image / images

Run 2026-10-04T01:27:59+00:00 · seed 0 · 250 real + 250 fake samples (label 1 = fake/spoof)

> **Note:** Check each model card for its training data and avoid evaluating on the same dataset: overlap inflates every number here.
> **Note:** Accuracy/FPR/FNR use a fixed threshold, not one tuned on this data. Threshold-free metrics (AUROC, EER) do not depend on it.

## Models
- `image_deepfake`: Ateeqq/ai-vs-human-image-detector — cached

## Results by condition (fixed threshold 0.5 for accuracy / FPR / FNR)

heuristic = shipped detector, learned OFF · learned = raw classifier P(fake) · combined = shipped detector, learned ON

### none

| score | AUROC [95% CI] | AP | EER | TPR@1%FPR | TPR@5%FPR | accuracy | FPR | FNR | scored / failed |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 0.664 [0.615, 0.714] | 0.590 | 37.2% | 0.8% | 3.6% | 49.4% | 4.0% | 97.2% | 500 / 0 |
| learned | 0.374 [0.329, 0.425] | 0.437 | 59.0% | 1.2% | 3.2% | 47.6% | 16.4% | 88.4% | 500 / 0 |
| combined | 0.543 [0.496, 0.591] | 0.494 | 42.8% | 1.2% | 3.2% | 47.0% | 19.2% | 86.8% | 500 / 0 |

### jpeg70

| score | AUROC [95% CI] | AP | EER | TPR@1%FPR | TPR@5%FPR | accuracy | FPR | FNR | scored / failed |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 0.675 [0.625, 0.725] | 0.603 | 36.4% | 0.4% | 4.4% | 49.4% | 1.6% | 99.6% | 500 / 0 |
| learned | 0.338 [0.293, 0.388] | 0.424 | 63.2% | 1.2% | 4.4% | 47.0% | 15.6% | 90.4% | 500 / 0 |
| combined | 0.560 [0.518, 0.608] | 0.507 | 41.6% | 1.2% | 3.6% | 46.6% | 16.4% | 90.4% | 500 / 0 |

### resize50_jpeg85

| score | AUROC [95% CI] | AP | EER | TPR@1%FPR | TPR@5%FPR | accuracy | FPR | FNR | scored / failed |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 0.578 [0.527, 0.630] | 0.521 | 43.0% | 0.0% | 3.2% | 49.2% | 4.8% | 96.8% | 500 / 0 |
| learned | 0.413 [0.367, 0.464] | 0.451 | 55.8% | 1.2% | 4.0% | 48.8% | 9.6% | 92.8% | 500 / 0 |
| combined | 0.468 [0.437, 0.500] | 0.498 | 53.4% | 1.2% | 4.0% | 48.4% | 12.8% | 90.4% | 500 / 0 |
