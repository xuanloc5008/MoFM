"""
Data Transforms & Augmentation Pipeline
-----------------------------------------
Uses MONAI transforms for medical-grade augmentation.
Preprocessing can run either online during training or offline as a separate
cache-building step. Augmentation always stays online.
"""
from typing import Tuple, Dict

from monai.transforms import (
    Compose,
    # Spatial
    RandFlipd,
    RandAffined,
    Rand2DElasticd,
    CenterSpatialCropd,
    SpatialPadd,
    # Intensity
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandBiasFieldd,
    RandGibbsNoised,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandHistogramShiftd,
    # Utility
    EnsureTyped,
    CastToTyped,
)
import torch


def get_intensity_normalizer() -> Compose:
    """
    Per-slice intensity normalization:
      1. Clip to [0.5, 99.5] percentile  (remove outliers)
      2. Zero-mean unit-variance per slice
    """
    return Compose([
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])


def get_spatial_resizer(spatial_size: Tuple[int, int] = (224, 224)) -> Compose:
    """Pad-then-crop to fixed spatial size (preserves aspect ratio better)."""
    return Compose([
        SpatialPadd(keys=["image", "label"],
                    spatial_size=spatial_size,
                    method="symmetric",
                    mode="constant"),
        CenterSpatialCropd(keys=["image", "label"],
                           roi_size=spatial_size),
    ])


def get_preprocessing_transforms(
    spatial_size: Tuple[int, int] = (224, 224),
) -> list:
    return [
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=spatial_size,
            method="symmetric",
            mode="constant",
        ),
        CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
    ]


def get_tensor_transforms() -> list:
    return [
        CastToTyped(keys=["image"], dtype=torch.float32),
        CastToTyped(keys=["label"], dtype=torch.long),
        # We do not need MONAI metadata downstream; plain tensors are cheaper
        # and avoid MetaTensor interactions with some backend-specific ops.
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ]


def get_train_augmentation_transforms(cfg: Dict = None) -> list:
    cfg = cfg or {}
    rr_prob   = cfg.get("rand_rotate_prob", 0.75)
    rr_range  = cfg.get("rand_rotate_range", 0.3491)   # ~20°
    rf_prob   = cfg.get("rand_flip_prob", 0.0)
    rz_prob   = cfg.get("rand_zoom_prob", 0.55)
    rz_range  = cfg.get("rand_zoom_range", [0.8, 1.2])
    rn_prob   = cfg.get("rand_gaussian_noise_prob", 0.30)
    rn_std    = cfg.get("rand_gaussian_noise_std", 0.07)
    rb_prob   = cfg.get("rand_bias_field_prob", 0.45)
    rg_prob   = cfg.get("rand_gibbs_noise_prob", 0.30)
    rc_prob   = cfg.get("rand_adjust_contrast_prob", 0.45)
    rs_prob   = cfg.get("rand_shift_intensity_prob", 0.30)
    rsi_prob  = cfg.get("rand_scale_intensity_prob", 0.30)
    rgs_prob  = cfg.get("rand_gaussian_smooth_prob", 0.25)
    rgs_sigma = tuple(cfg.get("rand_gaussian_smooth_sigma", [0.25, 1.5]))
    rb_coeff  = tuple(cfg.get("rand_bias_field_coeff_range", [0.0, 0.20]))
    rg_alpha  = tuple(cfg.get("rand_gibbs_alpha", [0.0, 0.80]))
    rc_gamma  = tuple(cfg.get("rand_adjust_contrast_gamma", [0.5, 1.8]))
    rsi_factor = cfg.get("rand_scale_intensity_factor", 0.25)
    rs_offset  = cfg.get("rand_shift_intensity_offset", 0.15)
    affine_prob = cfg.get("rand_affine_prob", max(rr_prob, rz_prob))
    elastic_prob = cfg.get("rand_elastic_prob", 0.25)
    translate = cfg.get("rand_translate_range", [12, 12])
    shear     = cfg.get("rand_shear_range", [0.04, 0.04])
    elastic_spacing   = tuple(cfg.get("rand_elastic_spacing",   [24, 24]))
    elastic_magnitude = tuple(cfg.get("rand_elastic_magnitude", [1.5, 4.0]))
    scale_range = [rz_range[0] - 1.0, rz_range[1] - 1.0]
    # Histogram shift — new; simulates scanner-specific LUT / windowing
    rhs_prob = cfg.get("rand_histogram_shift_prob", 0.30)
    rhs_ctrl = cfg.get("rand_histogram_shift_num_control_points", 10)

    return [
        # ── Spatial ──────────────────────────────────────────────────────
        RandAffined(
            keys=["image", "label"],
            prob=affine_prob,
            rotate_range=rr_range,
            translate_range=translate,
            scale_range=scale_range,
            shear_range=shear,
            mode=["bilinear", "nearest"],
            padding_mode="border",
            cache_grid=False,
        ),
        Rand2DElasticd(
            keys=["image", "label"],
            spacing=elastic_spacing,
            magnitude_range=elastic_magnitude,
            prob=elastic_prob,
            rotate_range=0.0,
            translate_range=0.0,
            scale_range=0.0,
            mode=["bilinear", "nearest"],
            padding_mode="border",
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=rf_prob,
            spatial_axis=1,
        ),

        # ── Intensity / scanner-style augmentation ────────────────────────
        # Strengthened vs v1 to force z_topo/c_app disentanglement and
        # improve robustness across Canon/GE/Philips/Siemens scanners.
        RandGaussianNoised(keys=["image"], prob=rn_prob, std=rn_std),
        RandGaussianSmoothd(keys=["image"], prob=rgs_prob, sigma_x=rgs_sigma, sigma_y=rgs_sigma),
        RandBiasFieldd(keys=["image"], prob=rb_prob, coeff_range=rb_coeff),
        RandGibbsNoised(keys=["image"], prob=rg_prob, alpha=rg_alpha),
        RandAdjustContrastd(keys=["image"], prob=rc_prob, gamma=rc_gamma),
        RandScaleIntensityd(keys=["image"], prob=rsi_prob, factors=rsi_factor),
        RandShiftIntensityd(keys=["image"], prob=rs_prob, offsets=rs_offset),
        # Histogram shift: redistributes voxel intensities along a random piecewise
        # linear map, mimicking how different scanners apply different LUT windows.
        RandHistogramShiftd(
            keys=["image"],
            prob=rhs_prob,
            num_control_points=rhs_ctrl,
        ),
    ]


def get_train_transforms(
    spatial_size: Tuple[int, int] = (224, 224),
    cfg: Dict = None,
) -> Compose:
    """
    Full training pipeline:
      preprocessing → augmentation → tensor conversion
    """
    return Compose([
        *get_preprocessing_transforms(spatial_size),
        *get_train_augmentation_transforms(cfg),
        *get_tensor_transforms(),
    ])


def get_train_transforms_for_preprocessed(cfg: Dict = None) -> Compose:
    """Training pipeline for offline-preprocessed slices."""
    return Compose([
        *get_train_augmentation_transforms(cfg),
        *get_tensor_transforms(),
    ])


def get_val_transforms(
    spatial_size: Tuple[int, int] = (224, 224),
) -> Compose:
    """Validation / test pipeline: only preprocessing, no augmentation."""
    return Compose([
        *get_preprocessing_transforms(spatial_size),
        *get_tensor_transforms(),
    ])


def get_val_transforms_for_preprocessed() -> Compose:
    """Validation / test pipeline for offline-preprocessed slices."""
    return Compose([
        *get_tensor_transforms(),
    ])


def get_inference_transforms(
    spatial_size: Tuple[int, int] = (224, 224),
) -> Compose:
    """Inference pipeline – no label key."""
    return Compose([
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        SpatialPadd(keys=["image"], spatial_size=spatial_size,
                    method="symmetric", mode="constant"),
        CenterSpatialCropd(keys=["image"], roi_size=spatial_size),
        CastToTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["image"], track_meta=False),
    ])
