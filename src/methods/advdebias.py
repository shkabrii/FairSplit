import torch
import torch.nn as nn
import torch.nn.functional as F

from src.methods.erm import ERM, classification_loss


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg()


class Adversary(nn.Module):
    def __init__(self, backbone, num_groups, layers):
        super().__init__()
        dims = [backbone.dim] + layers
        self.backbone = backbone
        self.network = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(dims[-1], 1 if num_groups == 2 else num_groups)

    def forward(self, x):
        return self.backbone(x)

    def sensitive(self, h):
        h = GradReverse.apply(F.relu(h))
        for layer in self.network:
            h = F.relu(layer(h))
        return self.head(h)


class AdvDebias(ERM):
    def __init__(self, cfg, model, data, device):
        super().__init__(cfg, Adversary(model, data["num_groups"], cfg.adversary_layers), data, device)

    def loss(self, x, y, s, warmup):
        h, out = self.model(x)
        loss = classification_loss(out, y)
        if warmup:
            return loss
        s_out = self.model.sensitive(h)
        if s_out.shape[1] == 1:
            return loss + self.cfg.lam * F.binary_cross_entropy(torch.sigmoid(s_out), s)
        return loss + self.cfg.lam * F.cross_entropy(s_out, s.view(-1).long())
