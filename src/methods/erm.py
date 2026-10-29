import torch
import torch.nn as nn


def one_hot(labels, n):
    out = torch.zeros([labels.shape[0], n], dtype=torch.float, device=labels.device)
    return out.scatter_(1, labels.unsqueeze(1).long(), 1)


def classification_loss(out, y):
    if out.shape[1] == 1:
        return nn.functional.binary_cross_entropy(out, y.float())
    return nn.functional.cross_entropy(out, y.long().squeeze())


class ERM:
    paired = False

    @staticmethod
    def outputs(num_classes):
        return 1 if num_classes == 2 else num_classes

    def __init__(self, cfg, model, data, device):
        self.cfg, self.model = cfg, model

    def start(self, loader, device):
        pass

    def loss(self, x, y, s, warmup):
        return classification_loss(self.model(x)[1], y)

    def predict(self, x):
        return self.model(x)

    def mask(self):
        return slice(None)

    def summary(self):
        return {}
