from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (base_ch * 2**i for i in range(5))
        self.inc = _DoubleConv(in_ch, c1)
        self.d1 = _Down(c1, c2)
        self.d2 = _Down(c2, c3)
        self.d3 = _Down(c3, c4)
        self.d4 = _Down(c4, c5)
        self.u1 = _Up(c5, c4, c4)
        self.u2 = _Up(c4, c3, c3)
        self.u3 = _Up(c3, c2, c2)
        self.u4 = _Up(c2, c1, c1)
        self.outc = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.outc(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(architecture: str = "unet", base_ch: int = 32,
                in_ch: int = 1, out_ch: int = 1) -> nn.Module:
    if architecture == "unet":
        return UNet(in_ch=in_ch, out_ch=out_ch, base_ch=base_ch)
    if architecture == "msunet":
        from .model_msunet import MSUNet
        c = base_ch
        channels = (c, 2 * c, 4 * c, 8 * c, 16 * c)
        return MSUNet(in_ch=in_ch, out_ch=out_ch, channels=channels)
    raise ValueError(f"Unknown architecture: {architecture!r}")
