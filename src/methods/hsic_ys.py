import torch

from src.methods.erm import ERM, classification_loss, one_hot


def hsic(x, y, sigma):
    def kernel(a):
        r = torch.sum(a ** 2, -1).reshape((-1, 1))
        return torch.exp(-(-2 * torch.mm(a, a.t()) + r + r.t()) / sigma)

    m = x.shape[0]
    center = (torch.eye(m) - 1.0 / m * torch.ones((m, m))).to(x.device)
    return torch.trace(torch.mm(kernel(y), torch.mm(center, torch.mm(kernel(x), center)))) / ((m - 1) ** 2)


class HSICYS(ERM):
    def __init__(self, cfg, model, data, device):
        super().__init__(cfg, model, data, device)
        self.num_groups = data["num_groups"]

    def loss(self, x, y, s, warmup):
        out = self.model(x)[1]
        loss = classification_loss(out, y)
        if warmup:
            return loss
        if out.shape[1] > 1:
            out, s = torch.softmax(out, dim=1), one_hot(s.view(-1).long(), self.num_groups)
        elif self.cfg.kind != "tabular":
            out, s = torch.cat([1 - out, out], dim=1), torch.cat([1 - s, s], dim=1)
        return loss + self.cfg.lam * hsic(out, s, self.cfg.sigma)
