"""
Topo-Evidential U-Mamba — Full Architecture (v2)
-------------------------------------------------
Implements the topology-anchored, context-modulated framework from the
manuscript.  The network has a single shared trunk with four branches:

  Input → UMamba Encoder → Bottleneck h_i
                ├── TopologyHead   H_ψ  → z_topo_i     (contrastive + anchor)
                ├── ContextEncoder C_ξ  → c_app_i       (style / residual)
                ├── MeanHead       A_μ  → μ_i           (anchor center)
                ├── VarianceHead   A_σ  → σ²_i          (context-modulated spread)
                └── UMamba Decoder [with CrossAttentionD3 @ D3]
                      ↑ conditioned on T_anchor = G_η(z_topo, μ, σ²)
                      └── EvidentialSegmentationHead → evidence, probs, uncertainty

Training loss (three terms):
  L = L_EDL  +  λ_dice · L_Dice  +  λ_metric · L_PD-SCon  +  λ_dist · L_dist

Inference calibration:
  d_mah = Σ_j (z_j − μ_j)² / (σ²_j + ε)
  conf_cal = conf_raw · exp(−γ · d_mah)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .umamba import UMamba
from .anchor_module import (
    ContextEncoder,
    MeanHead,
    VarianceHead,
    AnchorTokenGenerator,
    CrossAttentionD3,
    mahalanobis_anchor_distance,
    self_calibrated_confidence,
)
from ..losses.contrastive_loss import ProjectionMLP
from ..losses.edl_loss import evidence_to_dirichlet, uncertainty_map


class EvidentialSegmentationHead(nn.Module):
    """ReLU evidence head: dense decoder features → non-negative evidence."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class TopoEvidentialUMamba(nn.Module):
    """
    Forward dict keys:
      evidence    : (B, K, H, W)  raw ReLU evidence ≥ 0
      probs       : (B, K, H, W)  expected Dirichlet probabilities
      uncertainty : (B, 1, H, W)  u_p = K/S_p
      alpha       : (B, K, H, W)  Dirichlet parameters
      S           : (B, 1, H, W)  concentration sum
      z_topo      : (B, d_z)      topology latent (L2-normalised)
      mu          : (B, d_z)      anchor center
      var         : (B, d_z)      anchor variance σ²
      projections : (B, d_z)      alias for z_topo (contrastive loss compat.)
    """

    def __init__(
        self,
        in_channels:      int   = 1,
        num_classes:      int   = 4,
        feature_channels: List[int] = (32, 64, 128, 256, 512),
        d_state:          int   = 16,
        expand:           int   = 2,
        dropout:          float = 0.1,
        projection_dim:   int   = 128,   # d_z
        context_dim:      int   = 64,    # d_app
        d_token:          int   = 128,   # anchor token dim (= D3 channels)
        ca_num_heads:     int   = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = UMamba(
            in_channels=in_channels,
            feature_channels=list(feature_channels),
            d_state=d_state,
            expand=expand,
            dropout=dropout,
        )

        bottleneck_ch = self.backbone.bottleneck_ch   # 512
        decoder_out   = self.backbone.out_channels     # 32

        # D3 output channel = feature_channels[-3]  (index 2 from bottom = 128)
        d3_channels   = feature_channels[-3] if len(feature_channels) >= 3 else feature_channels[0]

        # ── Topology projection head  H_ψ (= z_topo, used for contrastive + dist) ─
        self.topology_head = ProjectionMLP(
            in_dim=bottleneck_ch,
            hid_dim=bottleneck_ch * 2,
            out_dim=projection_dim,
        )

        # ── Residual context encoder  C_ξ ────────────────────────────────────
        self.context_encoder = ContextEncoder(
            in_channels=bottleneck_ch,
            out_dim=context_dim,
            dropout=dropout,
        )

        # ── Anchor distribution heads ─────────────────────────────────────────
        self.mean_head = MeanHead(d_z=projection_dim)
        self.var_head  = VarianceHead(d_z=projection_dim, d_app=context_dim)

        # ── Anchor token generator ────────────────────────────────────────────
        self.token_gen = AnchorTokenGenerator(d_z=projection_dim, d_token=d_token)

        # ── Cross-attention @ D3 (wired into decoder) ─────────────────────────
        cross_attn = CrossAttentionD3(
            feat_dim=d3_channels,
            token_dim=d_token,
            num_heads=ca_num_heads,
            dropout=dropout,
        )
        self.backbone.decoder.cross_attn_d3 = cross_attn

        # ── Evidential segmentation head ──────────────────────────────────────
        self.seg_head = EvidentialSegmentationHead(decoder_out, num_classes)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x:                  torch.Tensor,
        return_projections: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        x : (B, 1, H, W)  normalised 2-D cardiac MRI slice
        """
        # ── Shared encoder ──────────────────────────────────────────────────
        v_i, skips = self.backbone.encoder(x)   # bottleneck + skip features

        # ── Topology latent z_topo ──────────────────────────────────────────
        z_topo = self.topology_head(v_i)         # (B, d_z)  L2-normalised

        # ── Residual context c_app ──────────────────────────────────────────
        c_app = self.context_encoder(v_i)        # (B, d_app)

        # ── Anchor distribution: stop-gradient prevents trivial μ ≡ z_topo ──
        z_sg = z_topo.detach()                   # sg(z_topo)
        mu   = self.mean_head(z_sg)              # (B, d_z)
        var  = self.var_head(z_sg, c_app)        # (B, d_z)  σ² > 0

        # ── Anchor tokens for D3 conditioning ──────────────────────────────
        anchor_tokens = self.token_gen(z_topo, mu, var)   # (B, 3, d_token)

        # ── Decoder with D3 cross-attention ────────────────────────────────
        dense_feat = self.backbone.decoder(v_i, skips, anchor_tokens=anchor_tokens)

        # ── Evidential segmentation ─────────────────────────────────────────
        evidence        = self.seg_head(dense_feat)           # (B, K, H, W)
        alpha, S, probs = evidence_to_dirichlet(evidence)
        uncertainty     = uncertainty_map(evidence)           # (B, 1, H, W)

        out: Dict[str, torch.Tensor] = {
            "evidence":    evidence,
            "probs":       probs,
            "uncertainty": uncertainty,
            "alpha":       alpha,
            "S":           S,
            "z_topo":      z_topo,
            "mu":          mu,
            "var":         var,
        }
        if return_projections:
            out["projections"] = z_topo   # backward-compat alias

        return out

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        x:     torch.Tensor,
        gamma: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference convenience method.

        Returns:
          seg_map     : (B, H, W)      argmax segmentation
          uncertainty : (B, 1, H, W)  pixel EDL uncertainty
          conf_cal    : (B, 1, H, W)  topology-anchor calibrated confidence
        """
        out     = self.forward(x, return_projections=False)
        seg_map = out["probs"].argmax(dim=1)

        d_mah   = mahalanobis_anchor_distance(out["z_topo"], out["mu"], out["var"])
        # raw confidence: mean foreground probability over predicted mask
        conf_raw = out["probs"].amax(dim=1, keepdim=True)
        conf_cal = self_calibrated_confidence(conf_raw, d_mah, gamma=gamma)

        return seg_map, out["uncertainty"], conf_cal

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> TopoEvidentialUMamba:
    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data",  {})

    return TopoEvidentialUMamba(
        in_channels      = data_cfg.get("in_channels",    1),
        num_classes      = data_cfg.get("num_classes",    4),
        feature_channels = model_cfg.get("feature_channels", [32, 64, 128, 256, 512]),
        d_state          = model_cfg.get("ssm_d_state",   16),
        expand           = model_cfg.get("ssm_expand_factor", 2),
        dropout          = model_cfg.get("dropout",        0.1),
        projection_dim   = model_cfg.get("projection_dim", 128),
        context_dim      = model_cfg.get("context_dim",    64),
        d_token          = model_cfg.get("d_token",        128),
        ca_num_heads     = model_cfg.get("ca_num_heads",   4),
    )
