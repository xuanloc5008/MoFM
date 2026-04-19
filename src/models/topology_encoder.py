"""
Learnable barcode encoders inspired by torchph's SLayer modules.

This module keeps the repo runnable even when `torchph` is not installed by
providing a lightweight exponential-structure-element fallback with the same
high-level behavior: permutation-invariant evaluation of barcode point sets.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchph.nn import SLayerExponential as TorchphSLayerExponential
    TORCHPH_AVAILABLE = True
except Exception:
    TorchphSLayerExponential = None
    TORCHPH_AVAILABLE = False


class FallbackSLayerExponential(nn.Module):
    """
    Small torchph-inspired structure-element layer.

    Given a multiset of 2-D barcode points, return one feature value per
    learnable exponential structure element by summing Gaussian-like responses.
    """

    def __init__(
        self,
        n_elements: int,
        point_dim: int = 2,
        sharpness_init: float = 3.0,
    ):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(n_elements, point_dim))
        self.log_sharpness = nn.Parameter(
            torch.full((n_elements, point_dim), float(sharpness_init)).log()
        )

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        outputs = []
        sharpness = F.softplus(self.log_sharpness) + 1e-6
        for pts in batch:
            if pts.numel() == 0:
                outputs.append(self.centers.new_zeros((self.centers.shape[0],)))
                continue
            diff = pts.unsqueeze(1) - self.centers.unsqueeze(0)   # (N, E, 2)
            response = torch.exp(-(sharpness.unsqueeze(0) * diff.pow(2)).sum(dim=2))
            outputs.append(response.sum(dim=0))
        return torch.stack(outputs, dim=0)


class BarcodeSLayerEncoder(nn.Module):
    """
    Encode H0/H1 barcode point clouds into a learnable topology embedding.

    Inputs are padded birth-lifetime tensors plus valid point counts. Internally
    the encoder converts them into multisets and applies one structure-element
    layer per homology dimension.
    """

    def __init__(
        self,
        dims: Iterable[int] = (0, 1),
        elements_per_dim: int = 16,
        output_dim: int = 128,
        backend: str = "auto",
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.dims = [int(d) for d in dims]
        if not self.dims:
            raise ValueError("BarcodeSLayerEncoder requires at least one homology dimension")

        self.backend = backend
        self.elements_per_dim = int(elements_per_dim)
        self.hidden_dim = int(hidden_dim or max(output_dim, elements_per_dim * len(self.dims)))

        self.slayer_layers = nn.ModuleDict()
        for dim in self.dims:
            self.slayer_layers[f"h{dim}"] = self._make_slayer()

        input_dim = self.elements_per_dim * len(self.dims)
        self.head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def _make_slayer(self) -> nn.Module:
        use_torchph = self.backend == "torchph" or (self.backend == "auto" and TORCHPH_AVAILABLE)
        if use_torchph and TorchphSLayerExponential is not None:
            return TorchphSLayerExponential(self.elements_per_dim, 2)
        return FallbackSLayerExponential(self.elements_per_dim, 2)

    @property
    def using_torchph(self) -> bool:
        return any(isinstance(layer, TorchphSLayerExponential) for layer in self.slayer_layers.values()) \
            if TORCHPH_AVAILABLE else False

    def _encode_one_dim(
        self,
        layer: nn.Module,
        points: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = points.shape[0]
        out = points.new_zeros((batch_size, self.elements_per_dim))

        nonempty_idx: List[int] = []
        multisets: List[torch.Tensor] = []
        for i in range(batch_size):
            count = int(counts[i].item())
            if count <= 0:
                continue
            nonempty_idx.append(i)
            multisets.append(points[i, :count].to(dtype=torch.float32))

        if nonempty_idx:
            encoded = layer(multisets).to(device=points.device, dtype=points.dtype)
            out[torch.tensor(nonempty_idx, device=points.device, dtype=torch.long)] = encoded
        return out

    def forward(self, barcode_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        for dim in self.dims:
            key = f"barcode_h{dim}"
            count_key = f"barcode_h{dim}_count"
            if key not in barcode_batch or count_key not in barcode_batch:
                raise KeyError(f"Missing barcode inputs for homology dimension {dim}")
            feats.append(
                self._encode_one_dim(
                    self.slayer_layers[f"h{dim}"],
                    barcode_batch[key],
                    barcode_batch[count_key],
                )
            )

        x = torch.cat(feats, dim=1)
        x = self.head(x)
        return F.normalize(x, dim=1)
