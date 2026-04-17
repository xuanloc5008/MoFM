"""
PD-Guided Supervised Contrastive Loss (LPD-SCon)
--------------------------------------------------
Topology-guided contrastive learning in the bottleneck latent space.

For each anchor i in mini-batch I:
  - Positive set PPD(i) is mined from topology similarity
  - All others are treated as negatives

L_PD-SCon = Σ_i  -1/|PPD(i)| * Σ_{p ∈ PPD(i)}
              log [ exp(v_i · v_p / τ) / Σ_{a ∈ A(i)} exp(v_i · v_a / τ) ]

The similarity matrix SimPD is pre-computed in CPU from persistence diagrams
and passed as a tensor to this loss function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PDSupervisedContrastiveLoss(nn.Module):
    """
    Topology-guided supervised contrastive loss.

    Args:
        temperature         : τ in the InfoNCE formula
        positive_threshold  : θ+  — SimPD score above which a pair is positive
        positive_top_k      : if set, keep only the top-k topology neighbours
                              per anchor as positives
        min_positives       : skip anchor if it has fewer than this many positives
    """

    def __init__(
        self,
        temperature:        float = 0.1,
        positive_threshold: float = 0.6,
        positive_top_k:     int | None = None,
        min_positives:      int   = 1,
    ):
        super().__init__()
        self.temperature        = temperature
        self.positive_threshold = positive_threshold
        self.positive_top_k     = positive_top_k
        self.min_positives      = min_positives

    def forward(
        self,
        projections:  torch.Tensor,         # (B, D) latent vectors
        topo_vectors: torch.Tensor,         # (B, T) cached topology descriptors
    ) -> torch.Tensor:
        """
        Returns scalar contrastive loss.

        projections : bottleneck vectors projected through MLP and L2-normalised
        topo_vectors: cached topology descriptors used to build a GPU similarity
                      matrix for positive-pair mining
        """
        B = projections.shape[0]
        device = projections.device

        # L2-normalise projections
        z = F.normalize(projections, dim=1)    # (B, D)
        topo_z = F.normalize(topo_vectors, dim=1)  # (B, T)

        # Pairwise cosine similarity scaled by temperature
        # (B, B) logit matrix
        logits = torch.mm(z, z.T) / self.temperature   # (B, B)

        # Cached topology similarity, rescaled from cosine [-1, 1] to [0, 1]
        topo_sim = 0.5 * (torch.mm(topo_z, topo_z.T) + 1.0)

        # Mask out diagonal (self-similarity)
        eye_mask = torch.eye(B, dtype=torch.bool, device=device)
        logits = logits.masked_fill(eye_mask, -1e9)

        if self.positive_top_k is not None and self.positive_top_k > 0:
            k = min(int(self.positive_top_k), max(B - 1, 0))
            if k == 0:
                pos_mask = torch.zeros_like(eye_mask)
            else:
                topo_rank = topo_sim.masked_fill(eye_mask, float("-inf"))
                topk_idx = topo_rank.topk(k=k, dim=1).indices
                pos_mask = torch.zeros_like(eye_mask)
                pos_mask.scatter_(1, topk_idx, True)
        else:
            # Threshold-based positives remain available as a fallback.
            pos_mask = (topo_sim > self.positive_threshold) & (~eye_mask)

        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        pos_mask_f = pos_mask.to(log_prob.dtype)
        pos_counts = pos_mask_f.sum(dim=1)
        valid_mask = pos_counts >= self.min_positives

        if not valid_mask.any():
            return projections.sum() * 0.0

        per_anchor_loss = -(log_prob * pos_mask_f).sum(dim=1) / pos_counts.clamp_min(1.0)
        return per_anchor_loss[valid_mask].mean()


class ProjectionMLP(nn.Module):
    """
    2-layer MLP projection head: maps bottleneck v_i to contrastive embedding.
    Applied after global average pooling of the bottleneck feature map.
    """

    def __init__(
        self,
        in_dim:  int,
        hid_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H', W')  bottleneck feature map
        returns: (B, out_dim)  L2-normalised projection
        """
        # Global average pooling → (B, C)
        pooled = x.mean(dim=[2, 3])
        proj   = self.net(pooled)
        return F.normalize(proj, dim=1)
