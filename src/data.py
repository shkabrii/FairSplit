import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs",
    "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
    "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
    "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]


def tensors(*arrays):
    return TensorDataset(*[torch.as_tensor(np.asarray(a, dtype=np.float32)) for a in arrays])


def column(a):
    return np.asarray(a, dtype=np.float32).reshape(-1, 1)


def load_adult(root):
    names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital-status", "occupation",
             "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week",
             "native-country", "target"]
    read = dict(names=names, na_values="?", sep=r"\s*,\s*", engine="python")
    df = pd.concat([pd.read_csv(os.path.join(root, "adult.data"), **read),
                    pd.read_csv(os.path.join(root, "adult.test"), skiprows=1, **read)])
    df = df.drop(columns=["fnlwgt"]).dropna()
    s = (df["sex"] == "Male").astype(int).to_frame()
    y = df["target"].replace({"<=50K.": 0, ">50K.": 1, ">50K": 1, "<=50K": 0})
    X = df.drop(columns=["target", "sex"])
    categorical = ["workclass", "marital-status", "occupation", "relationship", "native-country", "education", "race"]
    X[categorical] = X[categorical].astype("string")
    numeric = X.select_dtypes(exclude="string").columns
    X[numeric] = X[numeric].astype("float32")
    return X, y, s


def load_compas(root):
    df = pd.read_csv(os.path.join(root, "compas-scores-two-years.csv"))
    df = df[(df["days_b_screening_arrest"] >= -30) & (df["days_b_screening_arrest"] <= 30)
            & (df["is_recid"] != -1) & (df["c_charge_degree"] != "O") & (df["score_text"] != "N/A")]
    df = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)
    y = df["two_year_recid"].astype(int)
    s = ((df["sex"] == "Female").astype(int) + 2 * (df["age"] <= 25).astype(int)).to_frame()
    X = df[["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "c_charge_degree"]].copy()
    X["c_charge_degree"] = X["c_charge_degree"].astype("string")
    numeric = X.select_dtypes(exclude="string").columns
    X[numeric] = X[numeric].astype("float32")
    return X, y, s


def frame_to_tensor(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("int64")
    return torch.from_numpy(df.values).float()


def tabular(name, root, seed):
    X, y, s = (load_adult if name == "adult" else load_compas)(os.path.join(root, name, "raw"))
    X = pd.get_dummies(X, columns=X.select_dtypes("string").columns)
    X_tr, X_rest, y_tr, y_rest, s_tr, s_rest = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=seed)
    X_te, X_va, y_te, y_va, s_te, s_va = train_test_split(X_rest, y_rest, s_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    numeric = X.select_dtypes("float32").columns
    scaler = StandardScaler().fit(X_tr[numeric])
    splits = []
    for Xi, yi, si in ((X_tr, y_tr, s_tr), (X_va, y_va, s_va), (X_te, y_te, s_te)):
        Xi = Xi.copy()
        Xi[numeric] = pd.DataFrame(scaler.transform(Xi[numeric]), columns=numeric, index=Xi.index)
        splits.append(TensorDataset(frame_to_tensor(Xi), frame_to_tensor(yi), frame_to_tensor(si)))
    return splits


def load_pixels(path, channels):
    df = pd.read_csv(path, na_values="NA")
    X = np.stack([np.array(p.split(), dtype="float32").reshape(channels, 48, 48) / 255 for p in df["pixels"]])
    return (np.repeat(X, 3, axis=1) if channels == 1 else X), df


def image(name, root, seed):
    if name == "celeba-a":
        X, df = load_pixels(os.path.join(root, "celeba", "raw", "celeba.csv"), 3)
        attrs = df[["Attractive", "Male"]].to_numpy()
    else:
        X, df = load_pixels(os.path.join(root, "utkface", "raw", "age_gender.csv"), 1)
        age = pd.cut(df["age"], bins=[0, 25, 45, 120], labels=[0, 1, 2], right=True).astype(int)
        attrs = np.column_stack([age, df["ethnicity"].map({0: 0, 1: 1, 2: 2, 3: 2, 4: 2})])
    key = np.array([f"{a}_{b}" for a, b in attrs])
    X_tr, X_rest, a_tr, a_rest, _, key_rest = train_test_split(X, attrs, key, test_size=0.2, stratify=key, random_state=seed)
    X_te, X_va, a_te, a_va = train_test_split(X_rest, a_rest, test_size=0.5, stratify=key_rest, random_state=seed)
    return [tensors(Xi, column(a[:, 0]), column(a[:, 1])) for Xi, a in ((X_tr, a_tr), (X_va, a_va), (X_te, a_te))]


class FairFace(Dataset):
    def __init__(self, pixels, y, s):
        self.pixels, self.y, self.s = pixels, y, s

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, i):
        return torch.from_numpy(self.pixels[i]), torch.tensor([float(self.y[i])]), torch.tensor([float(self.s[i])])


def fairface(root, seed, n_train, n_test):
    root = os.path.join(root, "fairface")
    pixels = np.load(os.path.join(root, "pixels_224.npy"), mmap_mode="r")
    labels = pd.read_csv(os.path.join(root, "labels.csv"))
    y = (labels["age"].values >= 4).astype(int)
    s = labels["gender"].values.astype(int)
    rng = np.random.RandomState(seed)
    train = np.where(labels["split"].values == "train")[0]
    test = np.where(labels["split"].values == "validation")[0]
    train = rng.choice(train, min(n_train, len(train)), replace=False)
    test = rng.choice(test, min(n_test, len(test)), replace=False)

    def rows(idx):
        order = np.argsort(idx)
        out = np.empty_like(order)
        out[order] = np.arange(len(order))
        return np.asarray(pixels[idx[order]])[out]

    return [FairFace(rows(i), y[i], s[i]) for i in (train, test)]


def load_features(root, backbone):
    root = os.path.join(root, "celeba-fm")
    if backbone.startswith("clip_"):
        out = []
        for split in ("train", "test"):
            X = torch.load(os.path.join(root, "fairerclip", f"d=celebA-s={split}-m={backbone}.pt"),
                           map_location="cpu", weights_only=False).float().numpy()
            c = pd.read_csv(os.path.join(root, "fairerclip", f"celeba_{split}.csv"), header=None)
            out += [X, pd.DataFrame((c.iloc[:, 1:].values > 0).astype(int), columns=CELEBA_ATTRS)]
        return out
    z = np.load(os.path.join(root, f"celeba_{backbone}.npz"))
    X, split = z["X"].astype(np.float32), z["split"].astype(str)
    labels = pd.DataFrame({k: (z[k] > 0).astype(int) for k in z.files
                           if k not in ("X", "split") and np.issubdtype(z[k].dtype, np.number)})
    out = []
    for idx in (np.where(split == "train")[0], np.where(split != "train")[0]):
        out += [X[idx], labels.iloc[idx].reset_index(drop=True)]
    return out


def features(root, backbone, target, sensitive, transfer):
    X_all, lab_tr, X_te, lab_te = load_features(root, backbone)
    y_all, s_all = lab_tr[target].values, lab_tr[sensitive].values
    rng = np.random.RandomState(12345)
    tr, va = [], []
    for cell in np.unique(y_all * 2 + s_all):
        idx = np.where(y_all * 2 + s_all == cell)[0]
        idx = idx[rng.permutation(len(idx))]
        cut = int(round(0.1 * len(idx)))
        va.append(idx[:cut])
        tr.append(idx[cut:])
    tr, va = np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))
    mu, sd = X_all[tr].mean(0), X_all[tr].std(0) + 1e-8
    norm = lambda X: ((X - mu) / sd).astype(np.float32)
    splits = [TensorDataset(torch.from_numpy(norm(X_all[i])), torch.from_numpy(y_all[i]).float().unsqueeze(1),
                            torch.from_numpy(s_all[i]).float().unsqueeze(1)) for i in (tr, va)]
    splits.append(TensorDataset(torch.from_numpy(norm(X_te)), torch.from_numpy(lab_te[target].values).float().unsqueeze(1),
                                torch.from_numpy(lab_te[sensitive].values).float().unsqueeze(1)))
    transfer_labels = [lab_tr[transfer].values[tr], lab_tr[transfer].values[va], lab_te[transfer].values]
    return splits, transfer_labels


def load(cfg, seed):
    if cfg.kind == "tabular":
        return dict(zip(("train", "val", "test"), tabular(cfg.dataset, cfg.data, seed)))
    if cfg.kind == "image":
        return dict(zip(("train", "val", "test"), image(cfg.dataset, cfg.data, seed)))
    if cfg.kind == "vit":
        return dict(zip(("train", "test"), fairface(cfg.data, seed, cfg.n_train, cfg.n_test)))
    splits, transfer = features(cfg.data, cfg.backbone, cfg.target, cfg.sensitive, cfg.transfer)
    return dict(zip(("train", "val", "test"), splits), transfer=transfer)
