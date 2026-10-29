import torch

from src.methods.erm import ERM, classification_loss


def prejudice(out, s):
    out_0, out_1 = out[s == 0], out[s == 1]
    counts = torch.stack((torch.tensor(out_1.shape[0]), torch.tensor(out_0.shape[0]))).to(out.device)
    p_s = torch.stack((torch.sum(out_1), torch.sum(out_0))) / counts
    p = torch.sum(torch.cat((out_0, out_1))) / out.shape[0]
    return (torch.sum(out_0 * (torch.log(p_s[1]) - torch.log(p)))
            + torch.sum((1 - out_0) * (torch.log(1 - p_s[1]) - torch.log(1 - p)))
            + torch.sum(out_1 * (torch.log(p_s[0]) - torch.log(p)))
            + torch.sum((1 - out_1) * (torch.log(1 - p_s[0]) - torch.log(1 - p))))


class PRemover(ERM):
    def loss(self, x, y, s, warmup):
        out = self.model(x)[1]
        loss = classification_loss(out, y)
        if warmup:
            return loss
        return loss + self.cfg.lam * prejudice(out, s)
