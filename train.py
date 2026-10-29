import argparse
import json
import multiprocessing
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.data import load
from src.methods import METHODS
from src.metrics import metric_evaluation
from src.models import build

DEFAULTS = dict(seeds=[0, 1, 2, 3, 4], steps=150, lr=1e-3, lr_step=50, warmup=0, init_fraction=1.0, dropout=0.5,
                sigma=1.0, lambda_f=0.0, lambda_s=0.0, rho_f=0.0, rho_s=0.0)
DATASETS = {
    "adult": dict(kind="tabular", batch_size=1024, layers=[512, 256, 64], log=["train", "val", "test"]),
    "compas": dict(kind="tabular", batch_size=1024, layers=[512, 256, 64], log=["train", "val", "test"]),
    "celeba-a": dict(kind="image", batch_size=128, log=["val", "test"]),
    "utkface-ar": dict(kind="image", batch_size=128, log=["val", "test"]),
    "fairface": dict(kind="vit", batch_size=128, steps=300, lr=1e-4, lr_step=100, init_fraction=0.3, log=[],
                     n_train=60000, n_test=10000),
    "celeba-fm": dict(kind="features", batch_size=1024, eval_batch_size=8192, steps=600, lr=3e-4, lr_step=200,
                      init_fraction=0.3, log=[], layers=[1024, 512, 256], dropout=0.0, backbone="clip_ViTL14",
                      target="High_Cheekbones", sensitive="Male", transfer="Young"),
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def stream(dataset, batch_size):
    while True:
        yield from DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def labels(dataset):
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].numpy().ravel(), dataset.tensors[2].numpy().ravel()
    return np.asarray(dataset.y), np.asarray(dataset.s)


@torch.no_grad()
def predict(method, loader, device):
    method.model.eval()
    outs, hs, ys, ss = [], [], [], []
    for x, y, s in loader:
        h, out = method.predict(x.to(device))
        if out.shape[1] > 1:
            out = torch.softmax(out, dim=1)
        outs.append(out.cpu().numpy())
        hs.append(h.cpu().numpy())
        ys.append(y.numpy())
        ss.append(s.numpy())
    return np.concatenate(outs), np.concatenate(hs), np.concatenate(ys), np.concatenate(ss)


def evaluate(method, loader, device):
    out, _, y, s = predict(method, loader, device)
    return {k: float(v) for k, v in metric_evaluation(y, out, s).items()}


def run(cfg, seed, device):
    seed_everything(seed)
    data = load(cfg, seed)
    train = data["train"]
    y_train, s_train = labels(train)
    num_classes, num_groups = len(np.unique(y_train)), len(np.unique(s_train))
    info = dict(num_classes=num_classes, num_groups=num_groups, y_train=y_train, s_train=s_train)

    Method = METHODS[cfg.method]
    method = Method(cfg, build(cfg, train[0][0].numel(), Method.outputs(num_classes)), info, device)
    model = method.model.to(device)

    eval_bs = getattr(cfg, "eval_batch_size", cfg.batch_size)
    loaders = {k: DataLoader(v, batch_size=eval_bs, shuffle=False) for k, v in data.items() if k in ("val", "test")}
    loaders["train"] = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    if method.paired:
        groups = [TensorDataset(*[t[train.tensors[2][:, 0] == g] for t in train.tensors]) for g in (0, 1)]
        batches = zip(*[stream(g, cfg.batch_size // 2) for g in groups])
    else:
        batches = stream(train, cfg.batch_size)

    def optimizer():
        opt = optim.Adam(model.parameters(), lr=cfg.lr)
        return opt, StepLR(opt, step_size=cfg.lr_step, gamma=0.1)

    opt, schedule = optimizer()
    for step, batch in enumerate(batches):
        if step >= cfg.steps:
            break
        if cfg.warmup and step == cfg.warmup:
            method.start(loaders["train"], device)
            opt, schedule = optimizer()
        model.train()
        opt.zero_grad()
        warmup = step < cfg.warmup
        if method.paired:
            loss = method.loss(*[[t.to(device) for t in b] for b in batch], warmup)
        else:
            loss = method.loss(*[t.to(device) for t in batch], warmup)
        loss.backward()
        opt.step()
        schedule.step()
        if step <= 1:
            for split in cfg.log:
                print(f"step {step} {split}", json.dumps(evaluate(method, loaders[split], device)), flush=True)

    result = {split: evaluate(method, loaders[split], device) for split in ("val", "test") if split in loaders}
    result.update(method.summary())
    if cfg.kind == "features":
        mask = method.mask()
        h_train = predict(method, DataLoader(train, batch_size=eval_bs), device)[1]
        h_test = predict(method, loaders["test"], device)[1]
        probe = LogisticRegression(max_iter=3000, C=1.0).fit(h_train[:, mask], data["transfer"][0])
        s_test = labels(data["test"])[1]
        result["transfer"] = {k: float(v) for k, v in metric_evaluation(
            data["transfer"][2], probe.predict_proba(h_test[:, mask])[:, 1], s_test).items()}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="key=value")
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    cfg = dict(DEFAULTS)
    raw = yaml.safe_load(open(a.config))
    cfg.update(DATASETS[raw["dataset"]])
    cfg.update(raw)
    cfg.update({k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in a.overrides)})
    cfg = SimpleNamespace(**cfg, data=a.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg.method + (f"-{cfg.backbone}" if cfg.kind == "features" else "")
    out_dir = os.path.join(a.out, cfg.dataset, name)
    os.makedirs(out_dir, exist_ok=True)
    for seed in a.seeds if a.seeds is not None else cfg.seeds:
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            result = pool.apply(run, (cfg, seed, device))
        json.dump(dict(result, seed=seed, config=vars(cfg)), open(os.path.join(out_dir, f"seed{seed}.json"), "w"), indent=2)
        print(cfg.dataset, name, f"seed {seed}", json.dumps(result["test"]))


if __name__ == "__main__":
    main()
