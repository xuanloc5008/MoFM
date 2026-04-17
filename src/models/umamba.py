"""
U-Mamba Backbone
-----------------
U-Net style encoder-decoder with CNN-SSM hybrid blocks.
Encoder produces multi-scale feature maps + bottleneck vector v_i.
Skip connections are passed to the decoder.
The decoder output feeds the Evidential Head (softmax replaced by ReLU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .ssm_block import CNNSSMBlock


class DownBlock(nn.Module):
    """Encoder stage: CNNSSMBlock + 2x2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int, d_state: int, expand: int, dropout: float):
        super().__init__()
        self.block = CNNSSMBlock(in_ch, out_ch, d_state, expand, dropout)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.block(x)   # skip-connection feature
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    """Decoder stage: bilinear upsample + concat skip + CNNSSMBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 d_state: int, expand: int, dropout: float):
        super().__init__()
        self.up    = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = CNNSSMBlock(in_ch // 2 + skip_ch, out_ch, d_state, expand, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UMambaEncoder(nn.Module):
    """
    Multi-scale encoder.
    Returns: list of skip features (coarse → fine) + bottleneck v_i.
    """

    def __init__(
        self,
        in_channels:      int = 1,
        feature_channels: List[int] = (32, 64, 128, 256, 512),
        d_state:          int = 16,
        expand:           int = 2,
        dropout:          float = 0.1,
    ):
        super().__init__()
        channels = feature_channels
        self.downs = nn.ModuleList()

        # First block: raw image → first feature map
        prev = in_channels
        for ch in channels[:-1]:
            self.downs.append(DownBlock(prev, ch, d_state, expand, dropout))
            prev = ch

        # Bottleneck block (no pooling after)
        self.bottleneck = CNNSSMBlock(prev, channels[-1], d_state, expand, dropout)
        self.bottleneck_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for down in self.downs:
            skip, x = down(x)
            skips.append(skip)

        v_i = self.bottleneck(x)   # bottleneck: (B, C_bottleneck, H', W')
        return v_i, skips


class UMambaDecoder(nn.Module):
    """
    Multi-scale decoder.
    Takes bottleneck v_i + encoder skips → dense feature map.
    """

    def __init__(
        self,
        feature_channels: List[int] = (32, 64, 128, 256, 512),
        d_state:          int = 16,
        expand:           int = 2,
        dropout:          float = 0.1,
    ):
        super().__init__()
        channels = feature_channels
        self.ups = nn.ModuleList()

        for i in range(len(channels) - 1, 0, -1):
            in_ch   = channels[i]
            skip_ch = channels[i - 1]
            out_ch  = channels[i - 1]
            self.ups.append(UpBlock(in_ch, skip_ch, out_ch, d_state, expand, dropout))

    def forward(
        self,
        v_i:   torch.Tensor,         # bottleneck
        skips: List[torch.Tensor],   # from encoder (fine→coarse order after reverse)
    ) -> torch.Tensor:
        x = v_i
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return x


class UMamba(nn.Module):
    """
    Full U-Mamba backbone.
    Returns:
      - v_i:        bottleneck feature map  (B, C_bn, H', W')
      - dense_feat: decoded feature map      (B, C[0], H, W)
    """

    def __init__(
        self,
        in_channels:      int = 1,
        feature_channels: List[int] = (32, 64, 128, 256, 512),
        d_state:          int = 16,
        expand:           int = 2,
        dropout:          float = 0.1,
    ):
        super().__init__()
        self.encoder = UMambaEncoder(in_channels, feature_channels, d_state, expand, dropout)
        self.decoder = UMambaDecoder(feature_channels, d_state, expand, dropout)

        self.out_channels    = feature_channels[0]
        self.bottleneck_ch   = feature_channels[-1]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_i, skips     = self.encoder(x)
        dense_feat     = self.decoder(v_i, skips)
        return v_i, dense_feat
