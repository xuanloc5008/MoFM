"""
Expert Model: UNet (ResNet34 encoder) + 4-branch output.

Architecture:
    Input image (1, 256, 256)
    → ResNet34 encoder (pretrained ImageNet, trainable end-to-end)
    → UNet decoder with skip connections
    → Branch 1: Segmentation logits (K, 256, 256)
    → Branch 2: Evidential (Vacuity, Dissonance)
    → Branch 3: Contrastive embedding (128-d, L2-normalised)
    → Branch 4: Topological (per-structure probability maps for PH)

The backbone is NOT the contribution — it is a commodity component.
Our contribution (TTEC, uncertainty pipeline) is orthogonal to backbone choice.

Expected: Dice >= 0.90 on ACDC with end-to-end training.
Trainable params: ~24M (ResNet34) + ~2M (branches) ≈ 26M total.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import math

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False


# ============================================================
# Branch 2: Evidential
# ============================================================

class EvidentialBranch(nn.Module):
    """Compute per-pixel evidence, Vacuity, Dissonance from logits."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.K = num_classes

    def forward(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        p_hat = alpha / S
        vacuity = self.K / S.squeeze(1)
        b = evidence / S
        dissonance = self._dissonance(b)

        return {
            "evidence": evidence, "alpha": alpha, "S": S.squeeze(1),
            "p_hat": p_hat, "vacuity": vacuity, "dissonance": dissonance,
            "belief": b,
        }

    def _dissonance(self, b: torch.Tensor) -> torch.Tensor:
        B, K, H, W = b.shape
        diss = torch.zeros(B, H, W, device=b.device)
        for i in range(K):
            for j in range(i + 1, K):
                denom = b[:, i] + b[:, j] + 1e-8
                diss += 2.0 * b[:, i] * b[:, j] / denom
        return diss


# ============================================================
# Branch 3: Contrastive
# ============================================================

class ContrastiveBranch(nn.Module):
    """Bottleneck features → GAP → MLP → L2-normalised embedding."""

    def __init__(self, in_dim: int = 512, embed_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """bottleneck: (B, C, H', W') from encoder's deepest layer."""
        gap = bottleneck.mean(dim=(-2, -1))  # (B, C)
        return F.normalize(self.mlp(gap), p=2, dim=-1)


# ============================================================
# Branch 4: Topological
# ============================================================

class TopologicalBranch(nn.Module):
    """Extract per-structure probability maps for PH computation."""

    def forward(self, p_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"LV": p_hat[:, 1], "Myo": p_hat[:, 2], "RV": p_hat[:, 3]}


# ============================================================
# UNet Backbone (no SMP fallback)
# ============================================================

class SimpleUNet(nn.Module):
    """Lightweight UNet for when segmentation_models_pytorch is not available."""

    def __init__(self, in_channels=1, num_classes=4, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_channels, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        self.enc4 = self._block(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(base_ch * 8, base_ch * 16)
        self.bottleneck_dim = base_ch * 16

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = self._block(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = self._block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = self._block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = self._block(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, num_classes, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.final(d1)
        return logits, bn


# ============================================================
# Expert: UNet + 4 Branches
# ============================================================

class Expert(nn.Module):
    """UNet (ResNet34 encoder) + 4-branch output.

    Input:  (B, 1, 256, 256) grayscale cardiac MRI
    Output: dict with logits, p_hat, vacuity, dissonance, embedding, topo_maps

    If segmentation_models_pytorch is available, uses ResNet34 pretrained encoder.
    Otherwise, falls back to a standard UNet (~31M params).
    """

    def __init__(self, num_classes: int = 4, img_size: int = 256,
                 encoder_name: str = "resnet34", embed_dim: int = 128,
                 in_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size

        if SMP_AVAILABLE:
            # ResNet34-UNet from segmentation_models_pytorch
            self.unet = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=in_channels,
                classes=num_classes,
            )
            # Get bottleneck dimension from encoder
            # ResNet34: [64, 128, 256, 512], bottleneck = 512
            encoder_channels = self.unet.encoder.out_channels
            self.bottleneck_dim = encoder_channels[-1]  # 512 for resnet34
            print(f"  Using SMP {encoder_name} (pretrained=imagenet), "
                  f"bottleneck_dim={self.bottleneck_dim}")
        else:
            print("  WARNING: segmentation_models_pytorch not installed. "
                  "Using SimpleUNet (no pretrained encoder).")
            self.unet = None
            self.simple_unet = SimpleUNet(in_channels, num_classes)
            self.bottleneck_dim = self.simple_unet.bottleneck_dim

        # Branch 2: Evidential (operates on logits)
        self.edl_branch = EvidentialBranch(num_classes)

        # Branch 3: Contrastive (operates on bottleneck features)
        self.contrastive_branch = ContrastiveBranch(self.bottleneck_dim, embed_dim)

        # Branch 4: Topological (operates on p_hat)
        self.topo_branch = TopologicalBranch()

        # Learnable uncertainty weighting (Kendall et al.)
        self.log_sigma2 = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(0.0))
            for name in ["LV", "Myo", "RV"]
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, 1, H, W) grayscale cardiac MRI
        """
        if self.unet is not None:
            # SMP UNet — need to extract bottleneck separately
            features = self.unet.encoder(x)
            bottleneck = features[-1]  # Deepest encoder features
            decoder_output = self.unet.decoder(*features)
            logits = self.unet.segmentation_head(decoder_output)
        else:
            logits, bottleneck = self.simple_unet(x)

        # Ensure correct spatial size
        if logits.shape[-1] != self.img_size:
            logits = F.interpolate(logits, size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False)

        # Branch 2: Evidential
        edl_out = self.edl_branch(logits)

        # Branch 3: Contrastive (from bottleneck)
        embedding = self.contrastive_branch(bottleneck)

        # Branch 4: Topological
        topo_maps = self.topo_branch(edl_out["p_hat"])

        return {
            "logits": logits,
            "bottleneck": bottleneck,
            **edl_out,
            "embedding": embedding,
            "topo_maps": topo_maps,
            "log_sigma2": {k: v for k, v in self.log_sigma2.items()},
        }

    def get_sigma_values(self) -> Dict[str, float]:
        return {k: math.exp(v.item() / 2) for k, v in self.log_sigma2.items()}

    def get_effective_weights(self) -> Dict[str, float]:
        return {k: 0.5 / math.exp(v.item()) for k, v in self.log_sigma2.items()}

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_expert(num_classes: int = 4, img_size: int = 256,
                 encoder_name: str = "resnet34", embed_dim: int = 128,
                 in_channels: int = 1) -> Expert:
    """Build a UNet Expert with 4-branch output."""
    expert = Expert(num_classes, img_size, encoder_name, embed_dim, in_channels)
    params = expert.count_parameters()
    print(f"  Expert: {params['trainable']:,} trainable params "
          f"({'SMP ' + encoder_name if SMP_AVAILABLE else 'SimpleUNet'})")
    return expert
