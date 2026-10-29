import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ATTRS = ["Male", "High_Cheekbones", "Young"]


def load_image(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    t = (h - w) // 2
    im = im.crop((0, t, w, t + w)).resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--attr_file", required=True)
    p.add_argument("--partition_file", required=True)
    p.add_argument("--out", default="data/celeba-fm/celeba_dinov2L_full.npz")
    p.add_argument("--bs", type=int, default=256)
    a = p.parse_args()

    attr = pd.read_csv(a.attr_file, sep=r"\s+", skiprows=1)
    attr.index.name = "image_id"
    attr = attr.reset_index()
    part = pd.read_csv(a.partition_file, sep=r"\s+", header=None, names=["image_id", "partition"])
    df = attr.merge(part, on="image_id")
    df = pd.concat([df[df["partition"] == 0], df[df["partition"] == 2]]).reset_index(drop=True)
    split = np.where(df["partition"].values == 0, "train", "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    X = np.empty((len(df), enc.config.hidden_size), dtype=np.float32)
    ids = df["image_id"].values
    with torch.no_grad():
        for i in range(0, len(df), a.bs):
            xb = np.stack([load_image(os.path.join(a.img_dir, f)) for f in ids[i:i + a.bs]])
            x = torch.from_numpy(xb).to(device).float() / 255.0
            x = (x - MEAN.to(device)) / STD.to(device)
            X[i:i + len(xb)] = enc(pixel_values=x).pooler_output.float().cpu().numpy()

    out = {"X": X, "split": split.astype("U16"), "image_id": ids.astype("U32")}
    for c in ATTRS:
        out[c] = (df[c].values > 0).astype(np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
