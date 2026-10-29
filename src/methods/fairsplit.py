import torch
import torch.nn as nn

from src.methods.erm import ERM, classification_loss, one_hot


def update_centers(centers, h, assignment, count, lr):
    count = count.clone().unsqueeze(1)
    sums = torch.einsum("nd,nc->cd", h.detach(), assignment.to(h.dtype))
    sizes = assignment.sum(0).unsqueeze(1).to(h.dtype)
    seen = sizes.squeeze(1) != 0
    count[seen] = (1 - lr) * count[seen]
    count[seen] += lr * sizes[seen]
    step = 1.0 / (count[seen] + 1)
    batch_centers = sums[seen] / sizes[seen]
    if torch.all(centers == 0):
        new = batch_centers
    else:
        new = centers.clone()
        new[seen] = (1.0 - step) * centers[seen] + step * batch_centers
    return new, count.squeeze(1)


def mask_weights(h, class_centers, group_centers, y_onehot, s_onehot):
    with torch.no_grad():
        loss_f = ((h.unsqueeze(1) - class_centers.unsqueeze(0)).pow(2) * y_onehot.unsqueeze(2)).sum(0).sum(0)
        loss_s = ((h.unsqueeze(1) - group_centers.unsqueeze(0)).pow(2) * s_onehot.unsqueeze(2)).sum(0).sum(0)
        b_f = torch.sigmoid(torch.log(loss_s + 1e-12) - torch.log(loss_f + 1e-12))
        betas = torch.stack([b_f, 1.0 - b_f])
    betas[0][betas[0].isnan()] = 1e-12
    betas[1] = 1 - betas[0]
    return betas


def hard_mask(betas):
    work = betas.clone().detach()
    work[1][work[1] > 0.5] += 1.0
    winner = work.argmax(0)
    out = torch.zeros_like(work)
    for i in range(2):
        out[i] = (winner == i).float()
    return out


def compression_loss(h, centers, weights, assignment):
    dist = ((h.unsqueeze(1) - centers.unsqueeze(0)) * weights.unsqueeze(0).unsqueeze(1)).pow(2).sum(2)
    return (dist * assignment).mean()


def centered_kernel(x, sigma, gaussian):
    m = x.shape[0]
    center = torch.eye(m, device=x.device) - (1.0 / m) * torch.ones([m, m], device=x.device)
    if gaussian:
        r = torch.sum(x * x, 1).view([-1, 1])
        a = torch.mm(x, x.t())
        d = torch.abs(r.expand_as(a) - 2 * a + r.t().expand_as(a))
        k = torch.exp(-d / (2.0 * sigma * sigma * x.shape[1])).type(torch.float32)
    else:
        k = torch.mm(x, x.T).type(torch.float32)
    return torch.mm(k, center)


def hsic(z, s_onehot, sigma):
    m = z.shape[0]
    kz, ks = centered_kernel(z, sigma, True), centered_kernel(s_onehot, sigma, False)
    eye = torch.eye(m, device=z.device)
    rz = kz.mm(torch.inverse(kz + 1e-5 * m * eye))
    rs = ks.mm(torch.inverse(ks + 1e-5 * m * eye))
    return torch.sum(torch.mul(rz, rs.t()))


class FairSplit(nn.Module):
    def __init__(self, backbone, num_classes, num_groups, beta_step=0.99):
        super().__init__()
        d = backbone.dim
        self.backbone, self.num_classes, self.num_groups, self.beta_step = backbone, num_classes, num_groups, beta_step
        self.register_buffer("class_centers", torch.zeros(num_classes, d))
        self.register_buffer("group_centers", torch.zeros(num_groups, d))
        self.register_buffer("class_count", torch.zeros(num_classes))
        self.register_buffer("group_count", torch.zeros(num_groups))
        self.register_buffer("b_f", torch.ones(d))
        self.register_buffer("hard_b", torch.stack([torch.ones(d), torch.zeros(d)]))

    @property
    def d_f(self):
        return int(self.hard_b[0].sum().item())

    def forward(self, x, apply_split=False):
        h, out = self.backbone(x)
        if apply_split:
            out = self.backbone.head(h * self.hard_b[0])
            if self.backbone.num_classes == 1:
                out = torch.sigmoid(out)
        return h, out

    def update_centers(self, h, y_onehot, s_onehot, lr):
        centers, count = update_centers(self.class_centers, h, y_onehot, self.class_count, lr)
        self.class_centers.copy_(centers)
        self.class_count.copy_(count)
        centers, count = update_centers(self.group_centers, h, s_onehot, self.group_count, lr)
        self.group_centers.copy_(centers)
        self.group_count.copy_(count)

    def onehots(self, y, s):
        return one_hot(y.squeeze(1).long(), self.num_classes), one_hot(s.squeeze(1).long(), self.num_groups)

    @torch.no_grad()
    def initialize(self, loader, device, fraction):
        self.eval()
        hs, ys, ss, seen = [], [], [], 0
        for x, y, s in loader:
            if seen >= int(fraction * len(loader.dataset)):
                break
            hs.append(self(x.to(device))[0].cpu())
            ys.append(y.squeeze().long())
            ss.append(s.squeeze().long())
            seen += x.size(0)
        h = torch.cat(hs).to(device)
        y_onehot = one_hot(torch.cat(ys).to(device), self.num_classes)
        s_onehot = one_hot(torch.cat(ss).to(device), self.num_groups)
        self.class_count.fill_(100.0)
        self.group_count.fill_(100.0)
        self.update_centers(h, y_onehot, s_onehot, 1.0)
        betas = mask_weights(h, self.class_centers, self.group_centers, y_onehot, s_onehot)
        self.b_f.copy_(betas[0])
        self.hard_b.copy_(hard_mask(betas))
        self.train()

    def loss(self, x, y, s, warmup, cfg):
        h, _ = self(x)
        y_onehot, s_onehot = self.onehots(y, s)
        self.update_centers(h, y_onehot, s_onehot, 0.5)
        if warmup:
            out = self.backbone.head(h)
        else:
            betas = mask_weights(h.detach(), self.class_centers, self.group_centers, y_onehot, s_onehot)
            loss_f = compression_loss(h, self.class_centers, betas[0], y_onehot)
            loss_s = compression_loss(h, self.group_centers, betas[1], s_onehot)
            self.b_f.copy_(self.b_f * self.beta_step + betas[0].detach() * (1 - self.beta_step))
            self.hard_b.copy_(hard_mask(torch.stack([self.b_f, 1 - self.b_f])))
            z_f, z_s = h * self.hard_b[0], h * self.hard_b[1]
            out = self.backbone.head(z_f)
        if self.backbone.num_classes == 1:
            out = torch.sigmoid(out)
        loss = classification_loss(out, y)
        if warmup:
            return loss
        if cfg.lambda_f > 0:
            loss = loss + cfg.lambda_f * loss_f
        if cfg.lambda_s > 0:
            loss = loss + cfg.lambda_s * loss_s
        if cfg.rho_f > 0:
            loss = loss + cfg.rho_f * hsic(z_f, s_onehot, cfg.sigma)
        if cfg.rho_s > 0:
            loss = loss - cfg.rho_s * hsic(z_s, s_onehot, cfg.sigma)
        return loss


class FairSplitMethod(ERM):
    def __init__(self, cfg, model, data, device):
        super().__init__(cfg, FairSplit(model, data["num_classes"], data["num_groups"]), data, device)

    def start(self, loader, device):
        self.model.initialize(loader, device, self.cfg.init_fraction)

    def loss(self, x, y, s, warmup):
        return self.model.loss(x, y, s, warmup, self.cfg)

    def predict(self, x):
        return self.model(x, apply_split=True)

    def mask(self):
        return self.model.hard_b[0].bool().cpu().numpy()

    def summary(self):
        return {"d_f": self.model.d_f}
