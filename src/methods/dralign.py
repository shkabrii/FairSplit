import torch
import torch.nn as nn

from src.methods.erm import ERM, classification_loss


def decision_rationale(out, s, y, model):
    out, s, y = out.reshape(-1), s.reshape(-1), y.reshape(-1)
    dp = torch.abs(torch.mean(out[s == 0]) - torch.mean(out[s == 1]))
    grads = [torch.autograd.grad(nn.functional.binary_cross_entropy(out[s == g], y[s == g]), model.parameters(),
                                 create_graph=True, allow_unused=True) for g in (0, 1)]
    align = 0
    for g_0, g_1, (name, p) in zip(*grads, model.named_parameters()):
        if g_0 is None or g_1 is None or "weight" not in name or "fc" in name or "head" in name:
            continue
        importance = [(p * g).pow(2).view(p.shape[0], -1).sum(dim=1) for g in (g_0, g_1)]
        align += torch.cosine_similarity(*importance, dim=0)
    return dp, align


class DRAlign(ERM):
    paired = True

    def loss(self, batch_0, batch_1, warmup):
        out = torch.cat([self.model(batch_0[0])[1], self.model(batch_1[0])[1]])
        y, s = torch.cat([batch_0[1], batch_1[1]]), torch.cat([batch_0[2], batch_1[2]])
        loss = classification_loss(out, y)
        if warmup:
            return loss
        dp, alignment = decision_rationale(out, s, y, self.model)
        return loss + self.cfg.lam * dp - self.cfg.beta * alignment
