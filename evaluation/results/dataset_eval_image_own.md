# Dataset evaluation: image / own

Run 2026-10-04T02:24:06+00:00 · seed 0 · 60 real + 60 fake samples (label 1 = fake/spoof)

> **Note:** Check each model card for its training data and avoid evaluating on the same dataset: overlap inflates every number here.
> **Note:** Small sample (60 real / 60 fake): confidence intervals are wide; do not quote a single number.
> **Note:** Accuracy/FPR/FNR use a fixed threshold, not one tuned on this data. Threshold-free metrics (AUROC, EER) do not depend on it.

## Models
- `image_deepfake`: Ateeqq/ai-vs-human-image-detector — cached

## Results by condition (fixed threshold 0.5 for accuracy / FPR / FNR)

heuristic = shipped detector, learned OFF · learned = raw classifier P(fake) · combined = shipped detector, learned ON

### none

| score | AUROC [95% CI] | AP | EER | TPR@1%FPR | TPR@5%FPR | accuracy | FPR | FNR | scored / failed |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 0.336 [0.240, 0.438] | 0.474 | 65.0% | 1.7% | 18.3% | 56.7% | 5.0% | 81.7% | 120 / 0 |
| learned | 0.897 [0.830, 0.947] | 0.917 | 17.5% | 35.0% | 73.3% | 80.0% | 28.3% | 11.7% | 120 / 0 |
| combined | 0.891 [0.820, 0.947] | 0.917 | 17.5% | 36.7% | 73.3% | 79.2% | 30.0% | 11.7% | 120 / 0 |

### jpeg70

| score | AUROC [95% CI] | AP | EER | TPR@1%FPR | TPR@5%FPR | accuracy | FPR | FNR | scored / failed |
|---|---|---|---|---|---|---|---|---|---|
| heuristic | 0.275 [0.182, 0.371] | 0.417 | 68.3% | 0.0% | 10.0% | 52.5% | 3.3% | 91.7% | 120 / 0 |
| learned | 0.892 [0.827, 0.946] | 0.912 | 16.7% | 35.0% | 70.0% | 78.3% | 30.0% | 13.3% | 120 / 0 |
| combined | 0.870 [0.796, 0.937] | 0.903 | 16.7% | 35.0% | 70.0% | 80.0% | 26.7% | 13.3% | 120 / 0 |
