"""
Topo-Evidential U-Mamba — Full Architecture
---------------------------------------------
Y-shaped network (One Trunk, Two Branches):

  Input → UMamba Encoder → Bottleneck v_i
                                ├── Branch 1: PD-Guided Contrastive Head
                                │     (ProjectionMLP → SimPD-based L_PD-SCon)
                                └── Branch 2: Evidential Decoder Head
                                      (UMamba Decoder → ReLU → evidence e_p)
                                      → α_p, uncertainty u_p

During inference:
  - evidence e_p → expected class probabilities p̂_p
  - uncertainty u_p = K / S_p ∈ (0, 1]  (higher → less confident)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

from .umamba import UMamba
from ..losses.contrastive_loss import ProjectionMLP
from ..losses.edl_loss import evidence_to_dirichlet, uncertainty_map
from .topology_encoder import BarcodeSLayerEncoder


class EvidentialSegmentationHead(nn.Module):
    """
    Replaces Softmax with ReLU to produce non-negative evidence.
    Applied to the dense decoder output.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # ReLU to ensure evidence ≥ 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw evidence (B, K, H, W) ≥ 0."""
        return F.relu(self.conv(x))


class TopoEvidentialUMamba(nn.Module):
    """
    Full Topo-Evidential U-Mamba model.

    Forward pass returns a dict with:
      evidence    : (B, K, H, W)   raw ReLU evidence
      probs       : (B, K, H, W)   expected Dirichlet probabilities
      uncertainty : (B, 1, H, W)   pixel-level uncertainty u_p = K/S_p
      projections : (B, D)         L2-normalised bottleneck projections (for contrastive loss)
      alpha       : (B, K, H, W)   Dirichlet parameters
    """

    def __init__(
        self,
        in_channels:      int  = 1,
        num_classes:      int  = 4,
        feature_channels: List[int] = (32, 64, 128, 256, 512),
        d_state:          int  = 16,
        expand:           int  = 2,
        dropout:          float = 0.1,
        projection_dim:   int  = 128,
        topology_experimental: Optional[dict] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        topology_experimental = topology_experimental or {}

        # ── Trunk: U-Mamba backbone ──────────────────────────────────────
        self.backbone = UMamba(
            in_channels=in_channels,
            feature_channels=list(feature_channels),
            d_state=d_state,
            expand=expand,
            dropout=dropout,
        )

        bottleneck_ch = self.backbone.bottleneck_ch
        decoder_ch    = self.backbone.out_channels

        # ── Branch 1: TDA Contrastive Head ───────────────────────────────
        self.projection_head = ProjectionMLP(
            in_dim=bottleneck_ch,
            hid_dim=bottleneck_ch * 2,
            out_dim=projection_dim,
        )

        self.topology_encoder: Optional[BarcodeSLayerEncoder]
        if topology_experimental.get("enabled", False):
            dims = topology_experimental.get("barcode_dims", [0, 1])
            self.topology_encoder = BarcodeSLayerEncoder(
                dims=dims,
                elements_per_dim=int(topology_experimental.get("elements_per_dim", 16)),
                output_dim=int(topology_experimental.get("embed_dim", projection_dim)),
                hidden_dim=topology_experimental.get("hidden_dim"),
                backend=str(topology_experimental.get("backend", "auto")),
            )
        else:
            self.topology_encoder = None

        # ── Branch 2: Evidential Decoder Head ───────────────────────────
        self.seg_head = EvidentialSegmentationHead(decoder_ch, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_projections: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        x : (B, C, H, W)  normalised 2-D or 2.5-D cardiac MRI input
        """
        # ── Shared trunk ─────────────────────────────────────────────────
        v_i, dense_feat = self.backbone(x)    # bottleneck + decoded features

        # ── Branch 1: PD contrastive projection ──────────────────────────
        projections = None
        if return_projections:
            projections = self.projection_head(v_i)    # (B, D)

        # ── Branch 2: EDL segmentation ───────────────────────────────────
        evidence           = self.seg_head(dense_feat)                  # (B, K, H, W)
        alpha, S, probs    = evidence_to_dirichlet(evidence)
        uncertainty        = uncertainty_map(evidence)                  # (B, 1, H, W)

        out = {
            "evidence":    evidence,
            "probs":       probs,
            "uncertainty": uncertainty,
            "alpha":       alpha,
            "S":           S,
        }
        if projections is not None:
            out["projections"] = projections

        return out

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.
        Returns:
          seg_map     : (B, H, W)  argmax segmentation
          uncertainty : (B, 1, H, W)  pixel uncertainty map
        """
        out = self.forward(x, return_projections=False)
        seg_map = out["probs"].argmax(dim=1)
        return seg_map, out["uncertainty"]

    def encode_topology(self, barcode_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.topology_encoder is None:
            raise RuntimeError("Experimental topology encoder is not enabled in this model")
        return self.topology_encoder(barcode_batch)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> "TopoEvidentialUMamba":
    """Build model from config dict."""
    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data",  {})
    context_slices = int(data_cfg.get("context_slices", data_cfg.get("in_channels", 1)))

    return TopoEvidentialUMamba(
        in_channels      = context_slices,
        num_classes      = data_cfg.get("num_classes",    4),
        feature_channels = model_cfg.get("feature_channels", [32, 64, 128, 256, 512]),
        d_state          = model_cfg.get("ssm_d_state",   16),
        expand           = model_cfg.get("ssm_expand_factor", 2),
        dropout          = model_cfg.get("dropout",        0.1),
        projection_dim   = model_cfg.get("projection_dim", 128),
        topology_experimental = cfg.get("topology_experimental", {}),
    )
