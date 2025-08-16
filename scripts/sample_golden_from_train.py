import os, glob, random, shutil
SRC="data/images/train"
DST="data/golden/positives"
os.makedirs(DST, exist_ok=True)
imgs=[p for p in glob.glob(os.path.join(SRC,"**","*.*"), recursive=True)
      if p.lower().endswith((".jpg",".jpeg",".png"))]
random.seed(42)
sample = imgs if len(imgs) <= 150 else random.sample(imgs, 150)
for i, src in enumerate(sample, 1):
    ext = os.path.splitext(src)[1].lower()
    shutil.copy(src, os.path.join(DST, f"pos_{i:03d}{ext}"))
print(f"Copied {len(sample)} to {DST}")
