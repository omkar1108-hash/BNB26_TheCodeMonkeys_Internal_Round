from datasets import load_dataset
from pathlib import Path

N, MIN_SIDE = 250, 256

ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
names = ds.features["label"].names
print("class names:", names)
fake_ids = {i for i, n in enumerate(names) if "ai" in n.lower()}
print("treated as FAKE:", [names[i] for i in fake_ids])

root = Path("data/datasets/images")
(root / "real").mkdir(parents=True, exist_ok=True)
(root / "fake").mkdir(parents=True, exist_ok=True)

count = {"real": 0, "fake": 0}
ds = ds.shuffle(seed=0)
for i in range(len(ds)):
    if min(count.values()) >= N:
        break
    row = ds[i]
    kind = "fake" if row["label"] in fake_ids else "real"
    if count[kind] >= N:
        continue
    img = row["image"].convert("RGB")
    if min(img.size) < MIN_SIDE:
        continue
    img.save(root / kind / f"{kind}_{count[kind]:04d}.png")
    count[kind] += 1
print(count)