"""
Multi-task loss with topology-aware vessel terms (soft-clDice + thin/branch BCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_erode(img):
    p1 = -F.max_pool2d(-img, 3, 1, 1)
    p2 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p3 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    return F.max_pool2d(img, 3, 1, 1)


def soft_open(img, iters=1):
    for _ in range(iters):
        img = soft_erode(img)
    for _ in range(iters):
        img = soft_dilate(img)
    return img


def soft_skeletonize(img, iters=10):
    img1 = soft_open(img, 1)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img, 1)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(pred, target, iters=10, eps=1e-6):
    """Differentiable soft-clDice."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    skel_pred = soft_skeletonize(pred, iters)
    skel_true = soft_skeletonize(target, iters)
    tprec = (skel_pred * target).sum(dim=(2, 3)) / (skel_pred.sum(dim=(2, 3)) + eps)
    tsens = (skel_true * pred).sum(dim=(2, 3)) / (skel_true.sum(dim=(2, 3)) + eps)
    cl_dice = 2 * tprec * tsens / (tprec + tsens + eps)
    return 1.0 - cl_dice.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lam = cfg.get("lam", 0.5)
        self.w_cl = cfg.get("w_cl", 0.5)
        self.w_wb = cfg.get("w_wb", 0.5)
        self.cl_iters = cfg.get("cl_iters", 10)
        self.thin_boost = cfg.get("thin_boost", 2.0)
        self.branch_boost = cfg.get("branch_boost", 4.0)
        self.w_seg = cfg.get("w_seg", 1.0)
        self.w_cls = cfg.get("w_cls", 1.0)
        self.w_reg = cfg.get("w_reg", 1.0)
        self.w_av = cfg.get("w_av", 1.0)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss()

    def vessel_loss(self, pred, target, thin_mask=None, branch_mask=None):
        # pred, target: (B, 1, H, W) in [0,1]
        dice = 1.0 - (2 * (pred * target).sum() + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        bce = F.binary_cross_entropy(pred, target, reduction="none")

        weight = torch.ones_like(target)
        if thin_mask is not None:
            weight = weight + (self.thin_boost - 1.0) * thin_mask
        if branch_mask is not None:
            weight = weight + (self.branch_boost - 1.0) * branch_mask
        weighted_bce = (bce * weight).mean()

        cl = soft_cldice(pred, target, iters=self.cl_iters)

        loss = self.lam * dice + (1 - self.lam) * bce.mean()
        loss = loss + self.w_cl * cl + self.w_wb * weighted_bce
        return loss

    def forward(self, outputs, targets):
        """
        outputs: dict from AttentionUNet
        targets: dict with keys vessel, av, cls, reg  (and optionally thin, branch)
        """
        loss_seg = self.vessel_loss(
            outputs["vessel"],
            targets["vessel"],
            targets.get("thin"),
            targets.get("branch"),
        )
        loss_av = self.ce(outputs["av"], targets["av"].long())
        loss_cls = F.binary_cross_entropy_with_logits(outputs["cls"], targets["cls"].float())
        loss_reg = F.mse_loss(outputs["reg"], targets["reg"].float())

        total = (
            self.w_seg * loss_seg
            + self.w_av * loss_av
            + self.w_cls * loss_cls
            + self.w_reg * loss_reg
        )
        return {
            "total": total,
            "seg": loss_seg.detach(),
            "av": loss_av.detach(),
            "cls": loss_cls.detach(),
            "reg": loss_reg.detach(),
        }
