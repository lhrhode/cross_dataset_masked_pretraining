from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DDSCConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1) -> None:
        super().__init__()
        self.depth = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, padding=dilation, dilation=dilation,
            groups=in_ch, bias=False,
        )
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.point(self.depth(x))))


class _MSDDSCBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        # x1 = f1(i)
        self.f1 = _DDSCConv(ch, ch, dilation=1)
        # x2 = f2(i ⊕ x1) → in_ch = 2*ch
        self.f2 = _DDSCConv(2 * ch, ch, dilation=2)
        # x3 = f3(i ⊕ x1 ⊕ x2) → in_ch = 3*ch
        self.f3 = _DDSCConv(3 * ch, ch, dilation=4)

    def forward(self, i: torch.Tensor) -> torch.Tensor:
        x1 = self.f1(i)
        x2 = self.f2(torch.cat([i, x1], dim=1))
        x3 = self.f3(torch.cat([i, x1, x2], dim=1))
        # o = i ⊕ x1 ⊕ x2 ⊕ x3, then 1x1 conv-equivalent: sum the parts to
        # keep channel count = ch (avoids ballooning decoder widths).
        return i + x1 + x2 + x3


class _MSDDSCBlock2(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.a = _DDSCConv(ch, ch, dilation=1)
        self.b = _DDSCConv(ch, ch, dilation=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _MSIBranch(nn.Module):
    def __init__(self, scale: int, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.AvgPool2d(scale) if scale > 1 else nn.Identity(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.proj(raw)


class _EncoderStage(nn.Module):
    def __init__(self, prev_ch: int, msi_ch: int, out_ch: int) -> None:
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(prev_ch + msi_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = _MSDDSCBlock(out_ch)

    def forward(self, prev: torch.Tensor | None, msi: torch.Tensor) -> torch.Tensor:
        if prev is None:
            x = msi
            # Project msi only.
            return self.block(self.merge(x))
        if prev.shape[-2:] != msi.shape[-2:]:
            prev = F.interpolate(prev, size=msi.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return self.block(self.merge(torch.cat([prev, msi], dim=1)))


class _DecoderStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.merge = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.block = _MSDDSCBlock2(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.block(self.merge(torch.cat([x, skip], dim=1)))


class MSUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 channels: tuple[int, ...] = (16, 32, 64, 128, 256)) -> None:
        super().__init__()
        assert len(channels) == 5, "expect 4 encoder levels + 1 bottleneck"
        c1, c2, c3, c4, c5 = channels
        # MSI branches: one per encoder level.
        self.msi1 = _MSIBranch(scale=1, in_ch=in_ch, out_ch=c1)
        self.msi2 = _MSIBranch(scale=2, in_ch=in_ch, out_ch=c2)
        self.msi3 = _MSIBranch(scale=4, in_ch=in_ch, out_ch=c3)
        self.msi4 = _MSIBranch(scale=8, in_ch=in_ch, out_ch=c4)
        # Encoder.
        self.e1 = _EncoderStage(prev_ch=0, msi_ch=c1, out_ch=c1)
        self.pool1 = nn.AvgPool2d(2)
        self.e2 = _EncoderStage(prev_ch=c1, msi_ch=c2, out_ch=c2)
        self.pool2 = nn.AvgPool2d(2)
        self.e3 = _EncoderStage(prev_ch=c2, msi_ch=c3, out_ch=c3)
        self.pool3 = nn.AvgPool2d(2)
        self.e4 = _EncoderStage(prev_ch=c3, msi_ch=c4, out_ch=c4)
        self.pool4 = nn.AvgPool2d(2)
        # Bottleneck: only MS-DDSC, no MSI branch.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c5),
            nn.ReLU(inplace=True),
            _MSDDSCBlock(c5),
        )
        # Decoder.
        self.d4 = _DecoderStage(c5, c4, c4)
        self.d3 = _DecoderStage(c4, c3, c3)
        self.d2 = _DecoderStage(c3, c2, c2)
        self.d1 = _DecoderStage(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with multi-scale input.
        e1 = self.e1(None, self.msi1(x))
        e2 = self.e2(self.pool1(e1), self.msi2(x))
        e3 = self.e3(self.pool2(e2), self.msi3(x))
        e4 = self.e4(self.pool3(e3), self.msi4(x))
        b = self.bottleneck(self.pool4(e4))
        # Decoder.
        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)
        return self.out_conv(d1)
