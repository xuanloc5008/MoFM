"""
Topology-Anchor Distribution Loss  L_dist
------------------------------------------
Regularises the topology latent z_topo toward its sample-adaptive Gaussian
anchor q_i(z) = N(μ_i, diag(σ²_i)).

Per-dimension Gaussian NLL (Eq. 21 in manuscript / README):

  L_dist = (1/B) Σ_i  (1/2) Σ_j [
      (z_topo_{i,j} - μ_{i,j})² / (σ²_{i,j} + ε)
    + log(σ²_{i,j} + ε)
  ]

Two effects:
  1. Keeps z_topo close to the topology-driven anchor center μ.
  2. Forces σ² to be meaningful: the variance term log(σ²) prevents σ → 0
     (collapsing μ ≈ z trivially) while the NLL term prevents σ → ∞.
"""
import torch
import torch.nn as nn


class AnchorDistributionLoss(nn.Module):
    """
    Diagonal-Gaussian anchor distribution loss.

    Args:
        eps : numerical floor added to σ² for stability (also set in VarianceHead)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        z_topo: torch.Tensor,   # (B, d_z)
        mu:     torch.Tensor,   # (B, d_z)
        var:    torch.Tensor,   # (B, d_z)  σ² — must be strictly positive
    ) -> torch.Tensor:
        """Returns scalar mean loss."""
        var_stable = var + self.eps
        nll = 0.5 * (
            (z_topo - mu) ** 2 / var_stable
            + var_stable.log()
        )   # (B, d_z)
        return nll.mean()
