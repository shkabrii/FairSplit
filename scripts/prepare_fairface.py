import argparse
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fairface")
    p.add_argument("--cache_dir", default=None)
    a = p.parse_args()

    ds = load_dataset("HuggingFaceM4/FairFace", "0.25", cache_dir=a.cache_dir)
    splits = ["train", "validation"]
    n = sum(len(ds[s]) for s in splits)
    os.makedirs(a.out, exist_ok=True)
    pix = np.lib.format.open_memmap(os.path.join(a.out, "pixels_224.npy"), mode="w+",
                                    dtype=np.uint8, shape=(n, 3, 224, 224))
    rows, i = [], 0
    for split in splits:
        for r in ds[split]:
            im = r["image"].convert("RGB")
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.BILINEAR)
            pix[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
            rows.append(dict(split=split, age=r["age"], gender=r["gender"], race=r["race"],
                             service_test=r["service_test"]))
            i += 1
    pix.flush()
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "labels.csv"), index=False)


if __name__ == "__main__":
    main()
