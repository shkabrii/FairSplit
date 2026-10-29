import numpy as np
import torch

from src.methods.erm import ERM, classification_loss


def mixup_penalty(x_0, x_1, classify):
    gamma = np.random.beta(1, 1)
    mix = (x_0 * gamma + x_1 * (1 - gamma)).requires_grad_(True)
    grad = torch.autograd.grad(classify(mix).sum(), mix, create_graph=True)[0].view(mix.shape[0], -1)
    return torch.abs((grad * (x_1 - x_0).view(mix.shape[0], -1)).sum(1).mean())


class FairMixup(ERM):
    paired = True

    def encode(self, x):
        if self.cfg.kind != "image":
            return self.model(x)
        f = self.model.spatial(x)
        return f, self.model.classify(f)

    def loss(self, batch_0, batch_1, warmup):
        (x_0, y_0, _), (x_1, y_1, _) = batch_0, batch_1
        (h_0, out_0), (h_1, out_1) = self.encode(x_0), self.encode(x_1)
        loss = classification_loss(torch.cat([out_0, out_1]), torch.cat([y_0, y_1]))
        if warmup:
            return loss
        if self.cfg.kind == "image":
            return loss + self.cfg.lam * mixup_penalty(h_0, h_1, self.model.classify)
        return loss + self.cfg.lam * mixup_penalty(x_0, x_1, lambda x: self.model(x)[1])
