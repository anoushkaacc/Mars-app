"""Model architectures for the Mars segmentation comparison app.

This file contains inference-only definitions for the five checkpoint families
whose tensor names and shapes are documented in checkpoint_inspection_report.txt.
No dataset loading, training, downloads, or output-folder creation occurs here.

All model constructors default to pretrained=False so EfficientNet weights are
not downloaded when loading an existing checkpoint.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# -----------------------------------------------------------------------------
# Five-class U-Net and U-Net + CBAM
# -----------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two 3x3 convolution + batch norm + ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet5(nn.Module):
    """Standard U-Net matching unet.pth (5 output channels)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))

        x = self.dec1(torch.cat((self.up1(b), s4), dim=1))
        x = self.dec2(torch.cat((self.up2(x), s3), dim=1))
        x = self.dec3(torch.cat((self.up3(x), s2), dim=1))
        x = self.dec4(torch.cat((self.up4(x), s1), dim=1))
        return self.final(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg, max_], dim=1)))


class ConvBlockCBAM(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlockCBAM(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        features = self.conv(x)
        return features, self.pool(features)


class CBAMDecoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = ConvBlockCBAM(out_c * 2, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class CBAMUNet5(nn.Module):
    """CBAM U-Net matching unet_cbam.pth (5 output channels)."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.bottleneck = ConvBlockCBAM(512, 1024)
        self.dec1 = CBAMDecoderBlock(1024, 512)
        self.dec2 = CBAMDecoderBlock(512, 256)
        self.dec3 = CBAMDecoderBlock(256, 128)
        self.dec4 = CBAMDecoderBlock(128, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        x = self.dec1(b, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)
        return self.final(x)


# -----------------------------------------------------------------------------
# CCHNet_code model family
# -----------------------------------------------------------------------------
class FPConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        p=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                k,
                padding=p,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FPAGCB(nn.Module):
    """
    Adaptive Gated Convolutional Block.

    The block creates a learned gate over convolutional features
    and uses a residual connection to preserve useful information.
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.out_act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)

        y = self.conv2(y)
        y = self.bn2(y)

        gate = self.gate(y)

        y = y * gate

        return self.out_act(
            x + y
        )


class FPCSAIB(nn.Module):
    """
    Channel-wise and Spatial Attention Integration Block.
    """

    def __init__(
        self,
        channels,
        reduction=8,
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=False,
            ),
        )

        self.spatial = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False,
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(
            x, 1
        )

        mx = F.adaptive_max_pool2d(
            x, 1
        )

        channel_attn = torch.sigmoid(
            self.mlp(avg) +
            self.mlp(mx)
        )

        x = x * channel_attn

        avg_sp = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_sp, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        spatial_input = torch.cat(
            [avg_sp, max_sp],
            dim=1,
        )

        spatial_attn = torch.sigmoid(
            self.spatial(
                spatial_input
            )
        )

        return x * spatial_attn


class FPFARB(nn.Module):
    """
    Fusion Attention Refinement Block.

    It fuses decoder and skip features, then refines the fused
    representation using FPCSAIB and a residual path.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
    ):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.attn = FPCSAIB(
            out_ch
        )

        self.residual = (
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False,
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)

        y = self.fuse(x)
        y = self.attn(y)

        return F.relu(
            y + identity,
            inplace=True,
        )


class FPEfficientNetEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        # If pretrained weights cannot be downloaded, fall back to
        # randomly initialized EfficientNet instead of crashing.
        try:
            backbone = efficientnet_b0(weights=weights)
        except Exception as exc:
            print("WARNING: Could not load pretrained EfficientNet-B0 weights.")
            print("Reason:", exc)
            print("Falling back to randomly initialized EfficientNet-B0.")
            backbone = efficientnet_b0(weights=None)

        self.features = backbone.features

    def forward(self, x):
        skips = []

        for i, layer in enumerate(
            self.features
        ):
            x = layer(x)

            # For 512x512:
            # index 0 -> 256x256, 32 ch
            # index 2 -> 128x128, 24 ch
            # index 3 -> 64x64, 40 ch
            # index 4 -> 32x32, 80 ch
            # final -> 16x16, 1280 ch
            if i in (0, 2, 3, 4):
                skips.append(x)

        return skips, x


class FPDecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
    ):
        super().__init__()

        self.farb = FPFARB(
            in_ch + skip_ch,
            out_ch,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        return self.farb(x)


class YNet(nn.Module):

    def __init__(
        self,
        num_classes=12,
        pretrained=True,
    ):
        super().__init__()

        self.encoder = FPEfficientNetEncoder(
            pretrained=pretrained
        )

        # Encoder feature enhancement.
        self.agcb0 = FPAGCB(32)
        self.csai0 = FPCSAIB(32)

        self.agcb1 = FPAGCB(24)
        self.csai1 = FPCSAIB(24)

        self.agcb2 = FPAGCB(40)
        self.csai2 = FPCSAIB(40)

        self.agcb3 = FPAGCB(80)
        self.csai3 = FPCSAIB(80)

        # Bottleneck.
        self.bottleneck_reduce = FPConvBNReLU(
            1280,
            256,
            k=1,
            p=0,
        )

        self.bottleneck_agcb = FPAGCB(
            256
        )

        self.bottleneck_csai = FPCSAIB(
            256
        )

        self.bottleneck_farb = FPFARB(
            256,
            256,
        )

        # Decoder.
        self.dec1 = FPDecoderBlock(
            256, 80, 192
        )

        self.dec2 = FPDecoderBlock(
            192, 40, 128
        )

        self.dec3 = FPDecoderBlock(
            128, 24, 64
        )

        self.dec4 = FPDecoderBlock(
            64, 32, 32
        )

        self.dec5 = nn.Sequential(
            FPConvBNReLU(32, 32),
            FPConvBNReLU(32, 16),
        )

        # Y-Net branch 1: terrain segmentation.
        self.terrain_head = nn.Conv2d(
            16,
            num_classes,
            1,
        )

        # Y-Net branch 2: boundary refinement.
        self.boundary_head = nn.Conv2d(
            16,
            1,
            1,
        )

    def forward(self, x):
        skips, bottleneck = self.encoder(x)

        s0 = self.csai0(
            self.agcb0(skips[0])
        )

        s1 = self.csai1(
            self.agcb1(skips[1])
        )

        s2 = self.csai2(
            self.agcb2(skips[2])
        )

        s3 = self.csai3(
            self.agcb3(skips[3])
        )

        x = self.bottleneck_reduce(
            bottleneck
        )

        x = self.bottleneck_agcb(x)
        x = self.bottleneck_csai(x)
        x = self.bottleneck_farb(x)

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        x = self.dec4(x, s0)

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        x = self.dec5(x)

        terrain_logits = self.terrain_head(x)
        boundary_logits = self.boundary_head(x)

        return terrain_logits, boundary_logits


# -----------------------------------------------------------------------------
# CCHNet_binarization model family
# -----------------------------------------------------------------------------
BINARY_EPS = 1e-8
BINARY_EFFICIENTNET_INDICES = (5, 6, 7, 8)
BINARY_DECODER = False
BINARY_ATTENTION = False
BIN_BOTTLENECK_CH = 128
BIN_DEC1_CH = 192
BIN_DEC2_CH = 128
BIN_DEC3_CH = 64
BIN_DEC4_CH = 32
BIN_FINAL_CH = 24

class BinarySignSTE(torch.autograd.Function):
    """
    Straight-through estimator for sign activation.

    Forward:
        sign(x)

    Backward:
        gradient is passed through inside |x| <= 1 and clipped
        outside that range.
    """

    @staticmethod
    def forward(
        ctx,
        x
    ):
        ctx.save_for_backward(x)

        return torch.where(
            x >= 0,
            torch.ones_like(x),
            -torch.ones_like(x)
        )

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):
        (x,) = ctx.saved_tensors

        grad = grad_output.clone()

        mask = (
            x.abs() <= 1.0
        ).to(grad.dtype)

        return grad * mask


def binary_sign(x):
    return BinarySignSTE.apply(x)


class BinaryConv2d(nn.Module):
    """
    Binary-weight convolution.

    The underlying real-valued weight remains trainable.
    During forward propagation it is binarized with STE.

    A per-output-channel scale factor is used:

        alpha = mean(abs(weight))

        W_binary = sign(W) * alpha

    This is much more stable than using raw +/-1 weights.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False
    ):
        super().__init__()

        if isinstance(
            kernel_size,
            int
        ):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if isinstance(
            stride,
            int
        ):
            sh = sw = stride
        else:
            sh, sw = stride

        if isinstance(
            padding,
            int
        ):
            ph = pw = padding
        else:
            ph, pw = padding

        if isinstance(
            dilation,
            int
        ):
            dh = dw = dilation
        else:
            dh, dw = dilation

        self.in_channels = (
            in_channels
        )
        self.out_channels = (
            out_channels
        )
        self.kernel_size = (
            kh,
            kw
        )
        self.stride = (
            sh,
            sw
        )
        self.padding = (
            ph,
            pw
        )
        self.dilation = (
            dh,
            dw
        )
        self.groups = groups

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                kh,
                kw
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    out_channels
                )
            )
        else:
            self.bias = None

        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_out",
            nonlinearity="relu"
        )

    def forward(self, x):
        scale = (
            self.weight
            .abs()
            .mean(
                dim=(1, 2, 3),
                keepdim=True
            )
        )

        binary_weight = (
            binary_sign(
                self.weight
            ) *
            scale.clamp_min(
                BINARY_EPS
            )
        )

        return F.conv2d(
            x,
            binary_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class BinaryConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        p=1,
        groups=1
    ):
        super().__init__()

        self.conv = BinaryConv2d(
            in_ch,
            out_ch,
            k,
            padding=p,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(
            out_ch
        )

        self.act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BinConvBNReLU(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        p=1
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                k,
                padding=p,
                bias=False
            ),
            nn.BatchNorm2d(
                out_ch
            ),
            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):
        return self.block(x)


class BinAGCB(nn.Module):
    """
    Adaptive Gated Convolutional Block.

    Binary 3x3 feature convolutions are used when
    BINARY_ATTENTION=True.
    """

    def __init__(
        self,
        channels,
        reduction=8,
        binary=True
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        Conv = (
            BinaryConv2d
            if binary
            else nn.Conv2d
        )

        self.conv1 = Conv(
            channels,
            channels,
            3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            channels
        )

        self.conv2 = Conv(
            channels,
            channels,
            3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            channels
        )

        # 1x1 gates remain FP because they control the feature
        # selection and are relatively small.
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=True
            ),
            nn.Sigmoid()
        )

        self.out_act = nn.ReLU(
            inplace=True
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(
            y,
            inplace=True
        )

        y = self.conv2(y)
        y = self.bn2(y)

        gate = self.gate(y)

        y = y * gate

        return self.out_act(
            x + y
        )


class BinCSAIB(nn.Module):
    """
    Channel + Spatial Attention Integration Block.

    The original 7x7 spatial convolution is reduced to 3x3
    for the lightweight binary model.
    """

    def __init__(
        self,
        channels,
        reduction=8,
        binary=True
    ):
        super().__init__()

        hidden = max(
            channels // reduction,
            8
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                1,
                bias=False
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                1,
                bias=False
            )
        )

        SpatialConv = (
            BinaryConv2d
            if binary
            else nn.Conv2d
        )

        self.spatial = SpatialConv(
            2,
            1,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(
            x,
            1
        )

        mx = F.adaptive_max_pool2d(
            x,
            1
        )

        channel_attn = torch.sigmoid(
            self.mlp(avg) +
            self.mlp(mx)
        )

        x = (
            x *
            channel_attn
        )

        avg_sp = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_sp, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        spatial_input = torch.cat(
            [avg_sp, max_sp],
            dim=1
        )

        spatial_attn = torch.sigmoid(
            self.spatial(
                spatial_input
            )
        )

        return (
            x *
            spatial_attn
        )


class BinFARB(nn.Module):
    """
    Fusion Attention Refinement Block.

    1x1 fusion is FP; 3x3 refinement is binary.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        binary=True
    ):
        super().__init__()

        Conv3 = (
            BinaryConv2d
            if binary
            else nn.Conv2d
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False
            ),
            nn.BatchNorm2d(
                out_ch
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.fuse2 = nn.Sequential(
            Conv3(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(
                out_ch
            ),
            nn.ReLU(
                inplace=True
            )
        )

        self.attn = BinCSAIB(
            out_ch,
            binary=binary
        )

        self.residual = (
            nn.Conv2d(
                in_ch,
                out_ch,
                1,
                bias=False
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.residual(x)

        y = self.fuse1(x)
        y = self.fuse2(y)
        y = self.attn(y)

        return F.relu(
            y + identity,
            inplace=True
        )


class BinEfficientNetEncoder(nn.Module):

    def __init__(
        self,
        pretrained=True
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        try:
            backbone = efficientnet_b0(
                weights=weights
            )
        except Exception as exc:
            print(
                "WARNING: Could not load "
                "pretrained EfficientNet-B0."
            )
            print("Reason:", exc)
            print(
                "Using random initialization."
            )

            backbone = efficientnet_b0(
                weights=None
            )

        self.features = (
            backbone.features
        )

        # Replace selected deep convolutional layers
        # recursively. This preserves EfficientNet's MBConv
        # structure, including depthwise groups.
        if BINARY_EFFICIENTNET_INDICES:
            for idx in BINARY_EFFICIENTNET_INDICES:
                if idx < len(
                    self.features
                ):
                    self.features[idx] = (
                        self._binarize_module(
                            self.features[idx]
                        )
                    )

    def _binarize_module(
        self,
        module
    ):
        for name, child in list(
            module.named_children()
        ):
            if isinstance(
                child,
                nn.Conv2d
            ):
                # Keep 1x1 pointwise convolutions FP.
                # Binarize spatial 3x3/5x5 convolutions,
                # including depthwise convolutions.
                k = child.kernel_size

                if (
                    k[0] > 1
                    or k[1] > 1
                ):
                    new_conv = BinaryConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=(
                            child.bias is not None
                        )
                    )

                    with torch.no_grad():
                        new_conv.weight.copy_(
                            child.weight
                        )

                        if (
                            child.bias is not None
                            and new_conv.bias is not None
                        ):
                            new_conv.bias.copy_(
                                child.bias
                            )

                    setattr(
                        module,
                        name,
                        new_conv
                    )

            else:
                self._binarize_module(
                    child
                )

        return module

    def forward(self, x):
        skips = []

        for i, layer in enumerate(
            self.features
        ):
            x = layer(x)

            # Same skip locations as the uploaded Y-Net.
            if i in (
                0,
                2,
                3,
                4
            ):
                skips.append(x)

        return skips, x


class BinDecoderBlock(nn.Module):

    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch
    ):
        super().__init__()

        self.farb = BinFARB(
            in_ch + skip_ch,
            out_ch,
            binary=BINARY_DECODER
        )

    def forward(
        self,
        x,
        skip
    ):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x, skip],
            dim=1
        )

        return self.farb(x)


class BinaryYNet(nn.Module):

    def __init__(
        self,
        num_classes=12,
        pretrained=True
    ):
        super().__init__()

        self.encoder = (
            BinEfficientNetEncoder(
                pretrained=pretrained
            )
        )

        # Feature enhancement blocks.
        self.agcb0 = BinAGCB(
            32,
            binary=BINARY_ATTENTION
        )
        self.csai0 = BinCSAIB(
            32,
            binary=BINARY_ATTENTION
        )

        self.agcb1 = BinAGCB(
            24,
            binary=BINARY_ATTENTION
        )
        self.csai1 = BinCSAIB(
            24,
            binary=BINARY_ATTENTION
        )

        self.agcb2 = BinAGCB(
            40,
            binary=BINARY_ATTENTION
        )
        self.csai2 = BinCSAIB(
            40,
            binary=BINARY_ATTENTION
        )

        self.agcb3 = BinAGCB(
            80,
            binary=BINARY_ATTENTION
        )
        self.csai3 = BinCSAIB(
            80,
            binary=BINARY_ATTENTION
        )

        # ----------------------------------------------------
        # Lightweight bottleneck
        # ----------------------------------------------------

        self.bottleneck_reduce = (
            BinConvBNReLU(
                1280,
                BIN_BOTTLENECK_CH,
                k=1,
                p=0
            )
        )

        self.bottleneck_agcb = BinAGCB(
            BIN_BOTTLENECK_CH,
            binary=True
        )

        self.bottleneck_csai = BinCSAIB(
            BIN_BOTTLENECK_CH,
            binary=True
        )

        self.bottleneck_farb = BinFARB(
            BIN_BOTTLENECK_CH,
            BIN_BOTTLENECK_CH,
            binary=True
        )

        # ----------------------------------------------------
        # Lightweight decoder
        # ----------------------------------------------------

        self.dec1 = BinDecoderBlock(
            BIN_BOTTLENECK_CH,
            80,
            BIN_DEC1_CH
        )

        self.dec2 = BinDecoderBlock(
            BIN_DEC1_CH,
            40,
            BIN_DEC2_CH
        )

        self.dec3 = BinDecoderBlock(
            BIN_DEC2_CH,
            24,
            BIN_DEC3_CH
        )

        self.dec4 = BinDecoderBlock(
            BIN_DEC3_CH,
            32,
            BIN_DEC4_CH
        )

        # Final decoder refinement is FP.
        # The entire decoder (dec1-dec5) remains full precision.
        self.dec5 = nn.Sequential(
            BinConvBNReLU(
                BIN_DEC4_CH,
                BIN_FINAL_CH
            ),
            BinConvBNReLU(
                BIN_FINAL_CH,
                BIN_FINAL_CH
            )
        )

        # ----------------------------------------------------
        # Final Y-Net heads remain FP.
        # ----------------------------------------------------

        self.terrain_head = nn.Conv2d(
            BIN_FINAL_CH,
            num_classes,
            1
        )

        self.boundary_head = nn.Conv2d(
            BIN_FINAL_CH,
            1,
            1
        )

    def forward(self, x):
        skips, bottleneck = (
            self.encoder(x)
        )

        s0 = self.csai0(
            self.agcb0(skips[0])
        )

        s1 = self.csai1(
            self.agcb1(skips[1])
        )

        s2 = self.csai2(
            self.agcb2(skips[2])
        )

        s3 = self.csai3(
            self.agcb3(skips[3])
        )

        x = self.bottleneck_reduce(
            bottleneck
        )

        x = self.bottleneck_agcb(x)
        x = self.bottleneck_csai(x)
        x = self.bottleneck_farb(x)

        x = self.dec1(
            x,
            s3
        )

        x = self.dec2(
            x,
            s2
        )

        x = self.dec3(
            x,
            s1
        )

        x = self.dec4(
            x,
            s0
        )

        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        x = self.dec5(x)

        terrain_logits = (
            self.terrain_head(x)
        )

        boundary_logits = (
            self.boundary_head(x)
        )

        return (
            terrain_logits,
            boundary_logits
        )


# -----------------------------------------------------------------------------
# CCHNet_quantization model family
# -----------------------------------------------------------------------------
QUANT_BITS = 4
QUANT_EFFICIENTNET_INDICES = (5, 6, 7, 8)
QUANT_DECODER = False
QUANT_ATTENTION = False
QUANT_BOTTLENECK_CH = 128
QUANT_DEC1_CH = 192
QUANT_DEC2_CH = 128
QUANT_DEC3_CH = 64
QUANT_DEC4_CH = 32
QUANT_FINAL_CH = 24

class LSQQuantizeSTE(torch.autograd.Function):
    """
    Learned Step-Size Quantization (LSQ) function with scaled gradient.
    Computes uniform symmetric quantization with straight-through estimator.
    """
    @staticmethod
    def forward(ctx, weight, step_size, qmin, qmax):
        ctx.save_for_backward(weight, step_size)
        ctx.qmin = qmin
        ctx.qmax = qmax
        
        # Scale, clamp, round
        w_scaled = weight / step_size
        w_clipped = torch.clamp(w_scaled, qmin, qmax)
        w_q = torch.round(w_clipped)
        return w_q * step_size

    @staticmethod
    def backward(ctx, grad_output):
        weight, step_size = ctx.saved_tensors
        qmin = ctx.qmin
        qmax = ctx.qmax
        
        w_scaled = weight / step_size
        
        # Masks for gradient calculation
        mask_within = (w_scaled >= qmin) & (w_scaled <= qmax)
        mask_below = w_scaled < qmin
        mask_above = w_scaled > qmax
        
        # Weight gradient (standard STE)
        grad_weight = grad_output * mask_within.to(grad_output.dtype)
        
        # Step-size gradient with LSQ formula
        grad_round = torch.round(torch.clamp(w_scaled, qmin, qmax)) - w_scaled
        grad_step = torch.where(
            mask_within,
            grad_round,
            torch.where(mask_below, torch.full_like(w_scaled, qmin), torch.full_like(w_scaled, qmax))
        )
        
        # Gradient scaling to balance step size learning
        num_elements = weight.numel()
        grad_scale = 1.0 / math.sqrt(num_elements * qmax)
        grad_step_size = (grad_output * grad_step * grad_scale).sum().view_as(step_size)
        
        return grad_weight, grad_step_size, None, None


class QuantAdaptiveLSQConv2d(nn.Module):
    """
    Adaptive Multi-bit Convolution using Learned Step-size Quantization (LSQ).
    Replaces static BinaryConv2d with trainable quantization step-size.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        num_bits=QUANT_BITS
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_bits = num_bits
        
        # Symmetric quantization bounds
        self.qmin = -(2 ** (num_bits - 1))
        self.qmax = (2 ** (num_bits - 1)) - 1

        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw)
        )
        
        # Learnable per-tensor step size
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.init_done = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def init_step_size(self):
        """Initializes step size based on empirical weight magnitude."""
        with torch.no_grad():
            init_val = 2.0 * self.weight.abs().mean() / math.sqrt(self.qmax)
            self.step_size.copy_(init_val.clamp_min(1e-6))
        self.init_done = True

    def forward(self, x):
        if not self.init_done:
            self.init_step_size()

        quant_weight = LSQQuantizeSTE.apply(
            self.weight,
            self.step_size.clamp_min(1e-8),
            self.qmin,
            self.qmax
        )

        return F.conv2d(
            x,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class QuantConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class QuantAGCB(nn.Module):
    def __init__(self, channels, reduction=8, quantize=False):
        super().__init__()
        hidden = max(channels // reduction, 8)
        Conv = QuantAdaptiveLSQConv2d if quantize else nn.Conv2d

        self.conv1 = Conv(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = Conv(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y = y * self.gate(y)
        return self.out_act(x + y)


class QuantCSAIB(nn.Module):
    def __init__(self, channels, reduction=8, quantize=False):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )
        SpatialConv = QuantAdaptiveLSQConv2d if quantize else nn.Conv2d
        self.spatial = SpatialConv(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        channel_attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * channel_attn

        avg_sp = torch.mean(x, dim=1, keepdim=True)
        max_sp, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.sigmoid(self.spatial(torch.cat([avg_sp, max_sp], dim=1)))
        return x * spatial_attn


class QuantFARB(nn.Module):
    def __init__(self, in_ch, out_ch, quantize=False):
        super().__init__()
        Conv3 = QuantAdaptiveLSQConv2d if quantize else nn.Conv2d

        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.fuse2 = nn.Sequential(
            Conv3(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attn = QuantCSAIB(out_ch, quantize=quantize)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        y = self.attn(self.fuse2(self.fuse1(x)))
        return F.relu(y + identity, inplace=True)


class QuantEfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features

        if QUANT_EFFICIENTNET_INDICES:
            for idx in QUANT_EFFICIENTNET_INDICES:
                if idx < len(self.features):
                    self.features[idx] = self._quantize_module(self.features[idx])

    def _quantize_module(self, module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                k = child.kernel_size
                if k[0] > 1 or k[1] > 1:
                    new_conv = QuantAdaptiveLSQConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=(child.bias is not None),
                        num_bits=QUANT_BITS
                    )
                    with torch.no_grad():
                        new_conv.weight.copy_(child.weight)
                        if child.bias is not None:
                            new_conv.bias.copy_(child.bias)
                    new_conv.init_step_size()
                    setattr(module, name, new_conv)
            else:
                self._quantize_module(child)
        return module

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (0, 2, 3, 4):
                skips.append(x)
        return skips, x


class QuantDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.farb = QuantFARB(in_ch + skip_ch, out_ch, quantize=QUANT_DECODER)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.farb(torch.cat([x, skip], dim=1))


class AdaptiveQuantYNet(nn.Module):
    def __init__(self, num_classes=12, pretrained=True):
        super().__init__()
        self.encoder = QuantEfficientNetEncoder(pretrained=pretrained)

        self.agcb0 = QuantAGCB(32, quantize=QUANT_ATTENTION)
        self.csai0 = QuantCSAIB(32, quantize=QUANT_ATTENTION)
        self.agcb1 = QuantAGCB(24, quantize=QUANT_ATTENTION)
        self.csai1 = QuantCSAIB(24, quantize=QUANT_ATTENTION)
        self.agcb2 = QuantAGCB(40, quantize=QUANT_ATTENTION)
        self.csai2 = QuantCSAIB(40, quantize=QUANT_ATTENTION)
        self.agcb3 = QuantAGCB(80, quantize=QUANT_ATTENTION)
        self.csai3 = QuantCSAIB(80, quantize=QUANT_ATTENTION)

        self.bottleneck_reduce = QuantConvBNReLU(1280, QUANT_BOTTLENECK_CH, k=1, p=0)
        self.bottleneck_agcb = QuantAGCB(QUANT_BOTTLENECK_CH, quantize=True)
        self.bottleneck_csai = QuantCSAIB(QUANT_BOTTLENECK_CH, quantize=True)
        self.bottleneck_farb = QuantFARB(QUANT_BOTTLENECK_CH, QUANT_BOTTLENECK_CH, quantize=True)

        self.dec1 = QuantDecoderBlock(QUANT_BOTTLENECK_CH, 80, QUANT_DEC1_CH)
        self.dec2 = QuantDecoderBlock(QUANT_DEC1_CH, 40, QUANT_DEC2_CH)
        self.dec3 = QuantDecoderBlock(QUANT_DEC2_CH, 24, QUANT_DEC3_CH)
        self.dec4 = QuantDecoderBlock(QUANT_DEC3_CH, 32, QUANT_DEC4_CH)

        self.dec5 = nn.Sequential(
            QuantConvBNReLU(QUANT_DEC4_CH, QUANT_FINAL_CH),
            QuantConvBNReLU(QUANT_FINAL_CH, QUANT_FINAL_CH)
        )

        self.terrain_head = nn.Conv2d(QUANT_FINAL_CH, num_classes, 1)
        self.boundary_head = nn.Conv2d(QUANT_FINAL_CH, 1, 1)

    def forward(self, x):
        skips, bottleneck = self.encoder(x)

        s0 = self.csai0(self.agcb0(skips[0]))
        s1 = self.csai1(self.agcb1(skips[1]))
        s2 = self.csai2(self.agcb2(skips[2]))
        s3 = self.csai3(self.agcb3(skips[3]))

        x = self.bottleneck_reduce(bottleneck)
        x = self.bottleneck_agcb(x)
        x = self.bottleneck_csai(x)
        x = self.bottleneck_farb(x)

        x = self.dec1(x, s3)
        x = self.dec2(x, s2)
        x = self.dec3(x, s1)
        x = self.dec4(x, s0)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.dec5(x)

        return self.terrain_head(x), self.boundary_head(x)


# -----------------------------------------------------------------------------
# Public factory and post-load preparation
# -----------------------------------------------------------------------------
def build_model(model_key: str) -> nn.Module:
    """Construct the architecture that corresponds to a supported checkpoint."""
    constructors = {
        "unet": lambda: UNet5(num_classes=5),
        "unet_cbam": lambda: CBAMUNet5(num_classes=5),
        "cchnet_fp16": lambda: YNet(num_classes=12, pretrained=False),
        "cchnet_binary": lambda: BinaryYNet(num_classes=12, pretrained=False),
        "cchnet_quant": lambda: AdaptiveQuantYNet(num_classes=12, pretrained=False),
    }
    try:
        return constructors[model_key]()
    except KeyError as exc:
        raise KeyError(f"Unsupported model key: {model_key}") from exc


def mark_quantizers_initialized(model: nn.Module) -> None:
    """Preserve LSQ step sizes restored from a trained checkpoint.

    The original training layer lazily initializes its step_size on its first
    forward pass. After restoring a trained checkpoint, that would overwrite
    the learned value unless the runtime-only init flag is marked as complete.
    """
    for module in model.modules():
        if isinstance(module, QuantAdaptiveLSQConv2d):
            module.init_done = True


__all__ = [
    "UNet5", "CBAMUNet5", "YNet", "BinaryYNet", "AdaptiveQuantYNet",
    "build_model", "mark_quantizers_initialized",
]
