import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def activate(logits):
    return torch.sigmoid(logits) if logits.shape[1] == 1 else logits


class MLP(nn.Module):
    def __init__(self, n_features, layers, num_classes, dropout):
        super().__init__()
        dims = [n_features] + layers
        self.network = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])])
        self.head = nn.Linear(dims[-1], num_classes)
        self.num_classes, self.dim, self.dropout = num_classes, dims[-1], dropout

    def forward(self, x):
        for layer in self.network:
            x = F.dropout(F.relu(layer(x)), self.dropout, self.training)
        return x, activate(self.head(x))


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.head = nn.Linear(512, num_classes)
        self.num_classes, self.dim = num_classes, 512

    def forward(self, x):
        h = self.resnet(x)
        return h, activate(self.head(h))

    def spatial(self, x):
        r = self.resnet
        return r.layer4(r.layer3(r.layer2(r.layer1(r.maxpool(r.relu(r.bn1(r.conv1(x))))))))

    def classify(self, f):
        return activate(self.head(torch.flatten(self.resnet.avgpool(f), 1)))


class ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
        self.vit.heads = nn.Identity()
        self.head = nn.Linear(768, num_classes)
        self.num_classes, self.dim = num_classes, 768
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        h = self.vit((x.float() / 255.0 - self.mean) / self.std)
        return h, activate(self.head(h))


def build(cfg, n_features, num_classes):
    if cfg.kind == "image":
        return ResNet(num_classes)
    if cfg.kind == "vit":
        return ViT(num_classes)
    return MLP(n_features, cfg.layers, num_classes, cfg.dropout)
