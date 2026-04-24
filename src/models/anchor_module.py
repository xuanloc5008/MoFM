"""
Topology-Anchored, Context-Modulated Anchor Module
----------------------------------------------------
Implements the components from the manuscript Section VIII–IX:

  ContextEncoder   C_ξ  : h_i            → c_app_i  (residual appearance code)
  MeanHead         A_μ  : sg(z_topo_i)   → μ_i       (topology-driven anchor center)
  VarianceHead     A_σ  : [sg(z_topo), c_app] → σ²_i  (context-modulated spread)
  AnchorTokenGen   G_η  : (z_topo, μ, σ²)  → T_anchor (3 decoder-conditioning tokens)
  CrossAttentionD3      : D3 features × T_anchor → conditioned D3 features

Inference plausibility score (Eq. 24):
  d_mah = Σ_j (z_j - μ_j)² / (σ²_j + ε)
  conf_cal = conf_raw * exp(-γ * d_mah)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gap_gmp(x: torch.Tensor) -> torch.Tensor:
    """Global average + max pooling concat: (B,C,H,W) → (B, 2C)."""
    return torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)


# ── Context Encoder C_ξ ────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Residual appearance encoder.
    Captures scanner style / acquisition noise — NOT the topology anchor center.

    Input : bottleneck h_i  (B, C_bn, H', W')
    Output: c_app_i         (B, d_app)
    """

    def __init__(self, in_channels: int, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        pooled_dim = in_channels * 2    # GAP + GMP
        self.net = nn.Sequential(
            nn.Linear(pooled_dim, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(_gap_gmp(h))


# ── Mean Head A_μ ──────────────────────────────────────────────────────────────

class MeanHead(nn.Module):
    """
    Anchor center head.  Takes stop-gradient topology latent → μ.

    Input : sg(z_topo)  (B, d_z)
    Output: μ           (B, d_z)
    """

    def __init__(self, d_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z, d_z),
            nn.LayerNorm(d_z),
            nn.GELU(),
            nn.Linear(d_z, d_z),
        )

    def forward(self, z_topo_sg: torch.Tensor) -> torch.Tensor:
        return self.net(z_topo_sg)


# ── Variance Head A_σ ──────────────────────────────────────────────────────────

class VarianceHead(nn.Module):
    """
    Context-modulated anchor spread.

    Input : [sg(z_topo), c_app]  (B, d_z + d_app)
    Output: σ²                   (B, d_z)  — strictly positive via softplus
    """

    def __init__(self, d_z: int, d_app: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        in_dim = d_z + d_app
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_z),
            nn.LayerNorm(d_z),
            nn.GELU(),
            nn.Linear(d_z, d_z),
        )

    def forward(
        self, z_topo_sg: torch.Tensor, c_app: torch.Tensor
    ) -> torch.Tensor:
        rho = self.net(torch.cat([z_topo_sg, c_app], dim=-1))
        return F.softplus(rho) + self.eps


# ── Anchor Token Generator G_η ─────────────────────────────────────────────────

class AnchorTokenGenerator(nn.Module):
    """
    Projects (z_topo, μ, σ²) into 3 anchor tokens for decoder cross-attention.

      T_anchor = {t_topo, t_μ, t_σ}   shape (B, 3, d_token)

    Input : z_topo (B, d_z), μ (B, d_z), σ² (B, d_z)
    Output: T_anchor (B, 3, d_token)
    """

    def __init__(self, d_z: int, d_token: int = 128):
        super().__init__()
        self.tok_topo  = nn.Linear(d_z, d_token)
        self.tok_mu    = nn.Linear(d_z, d_token)
        # σ² is strictly positive; feed log to stabilise dynamic range
        self.tok_sigma = nn.Linear(d_z, d_token)

    def forward(
        self,
        z_topo: torch.Tensor,   # (B, d_z)
        mu:     torch.Tensor,   # (B, d_z)
        var:    torch.Tensor,   # (B, d_z) — σ²
    ) -> torch.Tensor:
        t_topo  = self.tok_topo(z_topo)                    # (B, d_token)
        t_mu    = self.tok_mu(mu)                           # (B, d_token)
        t_sigma = self.tok_sigma(var.clamp(min=1e-8).log()) # (B, d_token)
        return torch.stack([t_topo, t_mu, t_sigma], dim=1)  # (B, 3, d_token)


# ── Cross-Attention Block @ D3 ─────────────────────────────────────────────────

class CrossAttentionD3(nn.Module):
    """
    Conditions D3 decoder features on topology anchor tokens.

    Query  : D3 spatial features  (B, C_d3, H3, W3)
    Key/Val: anchor tokens         (B, 3, d_token)
    Output : conditioned D3        (B, C_d3, H3, W3)  (residual addition)

    When d_token == C_d3 no extra projection is needed; they share the same
    embedding dimension.  If they differ, learned Q/KV projections are used.
    """

    def __init__(
        self,
        feat_dim:  int,          # C_d3  — D3 feature channels
        token_dim: int,          # d_token
        num_heads: int = 4,
        dropout:   float = 0.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.norm_q   = nn.LayerNorm(feat_dim)
        self.norm_kv  = nn.LayerNorm(token_dim)

        # Project to a common dimension for multi-head attention
        self.q_proj  = nn.Linear(feat_dim,  feat_dim, bias=False)
        self.k_proj  = nn.Linear(token_dim, feat_dim, bias=False)
        self.v_proj  = nn.Linear(token_dim, feat_dim, bias=False)
        self.attn    = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.norm_out = nn.LayerNorm(feat_dim)

    def forward(
        self,
        feat:   torch.Tensor,   # (B, C_d3, H3, W3)
        tokens: torch.Tensor,   # (B, 3, d_token)
    ) -> torch.Tensor:
        B, C, H, W = feat.shape

        # Flatten spatial to sequence: (B, H*W, C)
        feat_flat = feat.flatten(2).permute(0, 2, 1)
        feat_norm = self.norm_q(feat_flat)
        tok_norm  = self.norm_kv(tokens)

        Q  = self.q_proj(feat_norm)     # (B, H*W, C)
        K  = self.k_proj(tok_norm)      # (B, 3,   C)
        V  = self.v_proj(tok_norm)      # (B, 3,   C)

        attn_out, _ = self.attn(Q, K, V)               # (B, H*W, C)
        attn_out    = self.out_proj(attn_out)

        # Residual + layer norm in the flat domain
        out_flat = self.norm_out(feat_flat + attn_out)  # (B, H*W, C)
        return out_flat.permute(0, 2, 1).reshape(B, C, H, W)


# ── Mahalanobis Plausibility Score ─────────────────────────────────────────────

def mahalanobis_anchor_distance(
    z_topo: torch.Tensor,  # (B, d_z)
    mu:     torch.Tensor,  # (B, d_z)
    var:    torch.Tensor,  # (B, d_z)  σ²
    eps:    float = 1e-6,
) -> torch.Tensor:
    """
    Diagonal-Gaussian Mahalanobis distance (Eq. 24):
      d_mah_i = Σ_j (z_j - μ_j)² / (σ²_j + ε)

    Returns: (B,) scalar per sample.
    """
    return ((z_topo - mu) ** 2 / (var + eps)).sum(dim=-1)


def self_calibrated_confidence(
    conf_raw: torch.Tensor,   # (B,) or (B,1,H,W)
    d_mah:    torch.Tensor,   # (B,) one scalar per sample
    gamma:    float = 0.01,
) -> torch.Tensor:
    """
    conf_cal = conf_raw * exp(-γ * d_mah)   (Eq. 25)
    """
    if d_mah.dim() == 1 and conf_raw.dim() == 4:
        d_mah = d_mah.view(-1, 1, 1, 1)
    return conf_raw * torch.exp(-gamma * d_mah)
