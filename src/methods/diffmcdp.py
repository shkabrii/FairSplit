import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR

from src.methods.erm import ERM, classification_loss


class MaxDiff(nn.Module):
    def __init__(self, temperature, ratio):
        super().__init__()
        self.temperature = temperature
        self.threshold = nn.Parameter(torch.FloatTensor([ratio]))

    def forward(self, out, s):
        cdf = [torch.sigmoid(self.temperature * (self.threshold - out[s == g])) for g in (0, 1)]
        return -torch.abs(torch.mean(cdf[0]) - torch.mean(cdf[1]))


def max_cdf_gap(out, s, temperature):
    out, s = out.reshape(-1), s.reshape(-1)
    points = torch.linspace(torch.min(out).item(), torch.max(out).item(), steps=100).reshape(-1, 1).to(out.device)
    cdf = [torch.mean(torch.sigmoid(temperature * (points - out[s == g])), axis=1) for g in (0, 1)]
    return torch.max(torch.abs(cdf[0] - cdf[1]))


class DiffMCDP(ERM):
    def __init__(self, cfg, model, data, device):
        super().__init__(cfg, model, data, device)
        if cfg.kind == "tabular":
            self.solver = MaxDiff(cfg.temperature, (data["y_train"] == 1).mean()).to(device)
            self.start(None, device)

    def start(self, loader, device):
        if self.cfg.kind == "tabular":
            self.opt = optim.Adam(self.solver.parameters(), lr=self.cfg.maxdiff_lr)
            self.schedule = StepLR(self.opt, step_size=10, gamma=0.1)

    def loss(self, x, y, s, warmup):
        if warmup:
            return classification_loss(self.model(x)[1], y)
        if self.cfg.kind != "tabular":
            out = self.model(x)[1]
            return classification_loss(out, y) + self.cfg.lam * max_cdf_gap(out, s, self.cfg.temperature)
        out = self.model(x)[1].detach()
        for _ in range(self.cfg.maxdiff_steps + 1):
            self.opt.zero_grad()
            self.solver(out, s).backward()
            self.opt.step()
            self.schedule.step()
        out = self.model(x)[1]
        return classification_loss(out, y) - self.cfg.lam * self.solver(out, s)
