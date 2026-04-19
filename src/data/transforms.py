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
    preprocess_cfg: Dict = None,
) -> list:
    preprocess_cfg = preprocess_cfg or {}
    transforms = []

    # When loader-level harmonization is enabled, resampling and volume
    # normalization already happened before slices reach MONAI. We only keep
    # deterministic crop/pad here.
    if not preprocess_cfg.get("enabled", False):
        transforms.extend([
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ])

    transforms.extend([
        SpatialPadd(
            keys=["image", "label"],
            spatial_size=spatial_size,
            method="symmetric",
            mode="constant",
        ),
        CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
    ])
    return transforms


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
    rr_prob   = cfg.get("rand_rotate_prob", 0.6)
    rr_range  = cfg.get("rand_rotate_range", 0.2618)   # ~15°
    rf_prob   = cfg.get("rand_flip_prob", 0.0)
    rz_prob   = cfg.get("rand_zoom_prob", 0.4)
    rz_range  = cfg.get("rand_zoom_range", [0.9, 1.1])
    rn_prob   = cfg.get("rand_gaussian_noise_prob", 0.15)
    rn_std    = cfg.get("rand_gaussian_noise_std", 0.03)
    rb_prob   = cfg.get("rand_bias_field_prob", 0.25)
    rg_prob   = cfg.get("rand_gibbs_noise_prob", 0.15)
    rc_prob   = cfg.get("rand_adjust_contrast_prob", 0.2)
    rs_prob   = cfg.get("rand_shift_intensity_prob", 0.15)
    rsi_prob  = cfg.get("rand_scale_intensity_prob", 0.15)
    rgs_prob  = cfg.get("rand_gaussian_smooth_prob", 0.1)
    rgs_sigma = tuple(cfg.get("rand_gaussian_smooth_sigma", [0.25, 0.8]))
    rb_coeff  = tuple(cfg.get("rand_bias_field_coeff_range", [0.0, 0.08]))
    rg_alpha  = tuple(cfg.get("rand_gibbs_alpha", [0.0, 0.5]))
    rc_gamma  = tuple(cfg.get("rand_adjust_contrast_gamma", [0.85, 1.15]))
    rsi_factor = cfg.get("rand_scale_intensity_factor", 0.1)
    rs_offset  = cfg.get("rand_shift_intensity_offset", 0.05)
    affine_prob = cfg.get("rand_affine_prob", max(rr_prob, rz_prob))
    elastic_prob = cfg.get("rand_elastic_prob", 0.15)
    translate = cfg.get("rand_translate_range", [8, 8])
    shear     = cfg.get("rand_shear_range", [0.025, 0.025])
    elastic_spacing = tuple(cfg.get("rand_elastic_spacing", [32, 32]))
    elastic_magnitude = tuple(cfg.get("rand_elastic_magnitude", [1.0, 2.5]))
    scale_range = [rz_range[0] - 1.0, rz_range[1] - 1.0]

    return [
        # ACDC cardiac MRI benefits most from mild rigid motion, scale jitter,
        # and modest non-rigid deformation. Large flips are disabled by default
        # because they can be anatomically implausible after dataset alignment.
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

        # ── Intensity augmentation (scanner-style artifacts) ──────────────
        RandGaussianNoised(keys=["image"], prob=rn_prob, std=rn_std),
        RandGaussianSmoothd(keys=["image"], prob=rgs_prob, sigma_x=rgs_sigma, sigma_y=rgs_sigma),
        RandBiasFieldd(keys=["image"], prob=rb_prob, coeff_range=rb_coeff),
        RandGibbsNoised(keys=["image"], prob=rg_prob, alpha=rg_alpha),
        RandAdjustContrastd(keys=["image"], prob=rc_prob, gamma=rc_gamma),
        RandScaleIntensityd(keys=["image"], prob=rsi_prob, factors=rsi_factor),
        RandShiftIntensityd(keys=["image"], prob=rs_prob, offsets=rs_offset),
    ]


def get_train_transforms(
    spatial_size: Tuple[int, int] = (224, 224),
    cfg: Dict = None,
    preprocess_cfg: Dict = None,
) -> Compose:
    """
    Full training pipeline:
      preprocessing → augmentation → tensor conversion
    """
    return Compose([
        *get_preprocessing_transforms(spatial_size, preprocess_cfg=preprocess_cfg),
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
    preprocess_cfg: Dict = None,
) -> Compose:
    """Validation / test pipeline: only preprocessing, no augmentation."""
    return Compose([
        *get_preprocessing_transforms(spatial_size, preprocess_cfg=preprocess_cfg),
        *get_tensor_transforms(),
    ])


def get_val_transforms_for_preprocessed() -> Compose:
    """Validation / test pipeline for offline-preprocessed slices."""
    return Compose([
        *get_tensor_transforms(),
    ])


def get_inference_transforms(
    spatial_size: Tuple[int, int] = (224, 224),
    preprocess_cfg: Dict = None,
) -> Compose:
    """Inference pipeline – no label key."""
    preprocess_cfg = preprocess_cfg or {}
    transforms = []
    if not preprocess_cfg.get("enabled", False):
        transforms.extend([
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0.5, upper=99.5,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ])
    transforms.extend([
        SpatialPadd(keys=["image"], spatial_size=spatial_size,
                    method="symmetric", mode="constant"),
        CenterSpatialCropd(keys=["image"], roi_size=spatial_size),
        CastToTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["image"], track_meta=False),
    ])
    return Compose(transforms)
