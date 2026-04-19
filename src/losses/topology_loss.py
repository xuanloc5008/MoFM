"""
Experimental topology losses.

These are intentionally lightweight so we can introduce richer topology signals
without destabilizing the repo's current EDL + Dice + cached-topology baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyAlignmentLoss(nn.Module):
    """
    Align image projections with learnable barcode embeddings.

    This is a low-risk cross-modal topology loss:
      image branch -> projection embedding
      barcode branch -> topology embedding

    The loss is a cosine dissimilarity, optionally weighted per sample.
    """

    def forward(
        self,
        projections: torch.Tensor,
        topology_embeddings: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_img = F.normalize(projections, dim=1)
        z_topo = F.normalize(topology_embeddings, dim=1)
        loss = 1.0 - F.cosine_similarity(z_img, z_topo, dim=1)
        if sample_weights is not None:
            w = sample_weights.to(device=loss.device, dtype=loss.dtype)
            return (loss * w).sum() / w.sum().clamp_min(1e-6)
        return loss.mean()
