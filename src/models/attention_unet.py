"""
Attention U-Net (Oktay et al.)
Multi-task heads: vessel, artery-vein, CIMT classification, CIMT regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=32, n_classes_vessel=1, n_classes_av=3):
        super().__init__()
        filters = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        # Encoder
        self.enc1 = ConvBlock(in_channels, filters[0])
        self.enc2 = ConvBlock(filters[0], filters[1])
        self.enc3 = ConvBlock(filters[1], filters[2])
        self.enc4 = ConvBlock(filters[2], filters[3])
        self.bottleneck = ConvBlock(filters[3], filters[4])
        self.pool = nn.MaxPool2d(2)

        # Attention gates + Decoder
        self.att4 = AttentionGate(filters[4], filters[3], filters[3] // 2)
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.dec4 = ConvBlock(filters[4], filters[3])

        self.att3 = AttentionGate(filters[3], filters[2], filters[2] // 2)
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec3 = ConvBlock(filters[3], filters[2])

        self.att2 = AttentionGate(filters[2], filters[1], filters[1] // 2)
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec2 = ConvBlock(filters[2], filters[1])

        self.att1 = AttentionGate(filters[1], filters[0], filters[0] // 2)
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.dec1 = ConvBlock(filters[1], filters[0])

        # Task heads
        self.head_vessel = nn.Conv2d(filters[0], n_classes_vessel, 1)
        self.head_av = nn.Conv2d(filters[0], n_classes_av, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_cls = nn.Linear(filters[4], 1)
        self.head_reg = nn.Linear(filters[4], 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        e4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out_vessel = torch.sigmoid(self.head_vessel(d1))
        out_av = self.head_av(d1)          # logits

        feat = self.gap(b).flatten(1)
        out_cls = self.head_cls(feat).squeeze(1)
        out_reg = self.head_reg(feat).squeeze(1)

        return {
            "vessel": out_vessel,
            "av": out_av,
            "cls": out_cls,
            "reg": out_reg,
        }
