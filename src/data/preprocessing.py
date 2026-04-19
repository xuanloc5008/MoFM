"""
Shared image-only preprocessing helpers for ACDC and M&Ms volumes.

Phase 1 harmonization goals:
  - crop away obvious zero-padding / empty borders
  - resample to a common in-plane spacing
  - normalize per volume (or phase volume) rather than per slice
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom


def harmonization_enabled(cfg: Optional[Dict]) -> bool:
    return bool(cfg and cfg.get("enabled", False))


def harmonize_volume(
    image: np.ndarray,
    label: Optional[np.ndarray],
    spacing: np.ndarray,
    cfg: Optional[Dict],
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Apply image-only harmonization to a 3-D short-axis volume.

    Parameters
    ----------
    image:
        Volume with shape (H, W, Z), float32 preferred.
    label:
        Optional integer label volume with shape (H, W, Z).
    spacing:
        Physical spacing as (sx, sy, sz).
    cfg:
        Preprocessing config block from configs/config.yaml.
    """
    image = np.asarray(image, dtype=np.float32)
    label_arr = None if label is None else np.asarray(label)
    spacing = np.asarray(spacing, dtype=np.float32)

    if not harmonization_enabled(cfg):
        return image, label_arr, spacing

    if cfg.get("crop_to_nonzero", True):
        image, label_arr = crop_volume_to_nonzero(
            image=image,
            label=label_arr,
            margin=int(cfg.get("crop_margin", 8)),
        )

    target_spacing_xy = cfg.get("target_spacing_xy", None)
    if target_spacing_xy is not None:
        image, label_arr, spacing = resample_volume_inplane(
            image=image,
            label=label_arr,
            spacing=spacing,
            target_spacing_xy=target_spacing_xy,
        )

    if cfg.get("volume_normalize", True):
        image = normalize_volume(
            image=image,
            clip_percentiles=tuple(cfg.get("volume_clip_percentiles", (0.5, 99.5))),
            nonzero_only=bool(cfg.get("normalize_nonzero_only", True)),
        )

    if label_arr is not None:
        label_arr = np.round(label_arr).astype(np.int64, copy=False)

    return image.astype(np.float32, copy=False), label_arr, spacing


def crop_volume_to_nonzero(
    image: np.ndarray,
    label: Optional[np.ndarray],
    margin: int = 8,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Crop only along the in-plane axes using the union of non-zero voxels.

    This is image-only and safe for OOD evaluation because it does not use
    labels or dataset-specific statistics from the target domain.
    """
    mask = np.abs(image) > 1e-6
    if not np.any(mask):
        return image, label

    rows = np.any(mask, axis=(1, 2))
    cols = np.any(mask, axis=(0, 2))
    if not np.any(rows) or not np.any(cols):
        return image, label

    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    r0 = max(0, int(row_idx[0]) - margin)
    r1 = min(image.shape[0], int(row_idx[-1]) + margin + 1)
    c0 = max(0, int(col_idx[0]) - margin)
    c1 = min(image.shape[1], int(col_idx[-1]) + margin + 1)

    image = image[r0:r1, c0:c1, :]
    if label is not None:
        label = label[r0:r1, c0:c1, :]
    return image, label


def resample_volume_inplane(
    image: np.ndarray,
    label: Optional[np.ndarray],
    spacing: np.ndarray,
    target_spacing_xy,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Resample only the in-plane axes to a common physical spacing.
    """
    target_x = float(target_spacing_xy[0])
    target_y = float(target_spacing_xy[1])
    if target_x <= 0 or target_y <= 0:
        return image, label, spacing

    zoom_x = float(spacing[0]) / target_x
    zoom_y = float(spacing[1]) / target_y
    if np.isclose(zoom_x, 1.0) and np.isclose(zoom_y, 1.0):
        return image, label, spacing

    zoom_factors = (zoom_x, zoom_y, 1.0)
    image = zoom(image, zoom_factors, order=1, mode="nearest")
    if label is not None:
        label = zoom(label, zoom_factors, order=0, mode="nearest")

    new_spacing = np.array([target_x, target_y, float(spacing[2])], dtype=np.float32)
    return image.astype(np.float32, copy=False), label, new_spacing


def normalize_volume(
    image: np.ndarray,
    clip_percentiles=(0.5, 99.5),
    nonzero_only: bool = True,
) -> np.ndarray:
    """
    Clip and z-normalize a 3-D volume while preserving explicit zero background.
    """
    image = np.asarray(image, dtype=np.float32)
    work = image.copy()

    mask = np.abs(work) > 1e-6 if nonzero_only else np.ones_like(work, dtype=bool)
    if not np.any(mask):
        return work

    values = work[mask]
    lo, hi = np.percentile(values, clip_percentiles)
    work = np.clip(work, lo, hi)

    norm_values = work[mask]
    mean = float(norm_values.mean())
    std = float(norm_values.std())
    if std < 1e-6:
        std = 1.0

    work[mask] = (work[mask] - mean) / std
    if nonzero_only:
        work[~mask] = 0.0
    return work.astype(np.float32, copy=False)
