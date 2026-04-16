"""
data/preprocessing.py
MONAI-based preprocessing pipeline for ACDC, M&Ms-1, M&Ms-2.

Actual dataset structures:
  ACDC  → ACDC-dataset/database/training(testing)/patient{NNN}/
  M&Ms1 → M&M1/Testing|Training/Labeled|Validation/{ALPHANUMERIC}/
  M&Ms2 → MnM2/dataset/{NNN}/
"""
import os, csv, json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, ScaleIntensityRangePercentilesd,
    CropForegroundd, ResizeWithPadOrCropd,
    NormalizeIntensityd, EnsureTyped,
    RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd,
    RandScaleIntensityd, RandAdjustContrastd,
    Rand3DElasticd, MapTransform,
)
from monai.data import Dataset, CacheDataset, DataLoader
from monai.utils import set_determinism

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Custom Transform: extract one 3-D frame from a 4-D volume
# ─────────────────────────────────────────────────────────────────────────────

class ExtractFramed(MapTransform):
    """
    Extract one 3-D frame from a 4-D volume (H,W,D,T) → (C,H,W,D).
    After MONAI EnsureChannelFirstd the shape is (C,H,W,D,T) for 4-D data.
    The frame index is read from the sample dict key "frame_index".
    If the volume is already 3-D (ACDC/MnM2 style) this is a no-op.
    """
    def __init__(self, keys=("image", "label"), allow_missing_keys=True):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data):
        d   = dict(data)
        idx = int(d.get("frame_index", 0))
        for key in self.key_iterator(d):
            vol = d[key]
            if isinstance(vol, torch.Tensor) and vol.ndim == 5:
                n_frames = vol.shape[-1]
                d[key]   = vol[..., min(idx, n_frames - 1)]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# ACDC parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_acdc(data_root: str) -> List[Dict]:
    """
    ACDC-dataset/database/training(testing)/patient{NNN}/
      patient{NNN}_frame{XX}.nii.gz      ← ED image
      patient{NNN}_frame{XX}_gt.nii.gz   ← ED label
      patient{NNN}_frame{YY}.nii.gz      ← ES image
      patient{NNN}_frame{YY}_gt.nii.gz   ← ES label
      Info.cfg                            ← ED:XX  ES:YY  Group:NOR/DCM/...
    """
    samples  = []
    db_root  = Path(data_root) / "ACDC-dataset" / "database"
    if not db_root.exists():
        print(f"  [ACDC] Not found: {db_root}")
        return samples

    for split_dir in ["training", "testing"]:
        split_path = db_root / split_dir
        if not split_path.exists():
            continue
        for patient_dir in sorted(split_path.glob("patient*")):
            info_file = patient_dir / "Info.cfg"
            if not info_file.exists():
                continue
            info     = _parse_info_cfg(info_file)
            ed_frame = int(info.get("ED", 1))
            es_frame = int(info.get("ES", 1))
            pid      = patient_dir.name

            for phase, frame_idx in [("ED", ed_frame), ("ES", es_frame)]:
                img_path = patient_dir / f"{pid}_frame{frame_idx:02d}.nii.gz"
                lbl_path = patient_dir / f"{pid}_frame{frame_idx:02d}_gt.nii.gz"
                if img_path.exists() and lbl_path.exists():
                    samples.append({
                        "image":     str(img_path),
                        "label":     str(lbl_path),
                        "patient":   pid,
                        "phase":     phase,
                        "dataset":   "ACDC",
                        "split":     split_dir,
                        "pathology": info.get("Group", "NOR"),
                        "height_cm": float(info.get("Height", 0)),
                        "weight_kg": float(info.get("Weight", 0)),
                    })
    return samples


def _parse_info_cfg(path: Path) -> Dict:
    info = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                info[k.strip()] = v.strip()
    return info


# ─────────────────────────────────────────────────────────────────────────────
# M&Ms-1 parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_mms(data_root: str, split: str = "train") -> List[Dict]:
    """
    M&M1/
      211230_M&M...endataset.csv
      Testing/   {ID}/ {ID}_sa.nii.gz  {ID}_sa_gt.nii.gz
      Training/
        Labeled/ {ID}/ {ID}_sa.nii.gz  {ID}_sa_gt.nii.gz
        Unlabeled/{ID}/ {ID}_sa.nii.gz
      Validation/{ID}/ {ID}_sa.nii.gz  {ID}_sa_gt.nii.gz

    _sa.nii.gz is 4-D (H,W,D,T). Frame index for ED/ES comes from CSV.
    split: "train" | "train_unlabeled" | "val" | "test"
    """
    split_map = {
        "train":           ("Training", "Labeled"),
        "train_unlabeled": ("Training", "Unlabeled"),
        "val":             ("Validation", None),
        "test":            ("Testing",    None),
    }
    if split not in split_map:
        raise ValueError(f"Unknown split '{split}'")

    top_dir, sub_dir = split_map[split]
    mms_root = Path(data_root) / "M&M1"
    if not mms_root.exists():
        return []

    csv_meta     = _load_mms_csv(mms_root)
    patient_root = (mms_root / top_dir / sub_dir
                    if sub_dir else mms_root / top_dir)
    if not patient_root.exists():
        return []

    samples = []
    for patient_dir in sorted(patient_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid      = patient_dir.name
        img_path = patient_dir / f"{pid}_sa.nii.gz"
        lbl_path = patient_dir / f"{pid}_sa_gt.nii.gz"
        if not img_path.exists():
            continue

        meta     = csv_meta.get(pid, {})
        ed_frame = meta.get("ED", 0)
        es_frame = meta.get("ES", 5)

        # Fall back: read n_frames from header
        if ed_frame == 0 and es_frame == 5:
            try:
                n_frames = nib.load(str(img_path)).shape[-1]
                es_frame = n_frames // 2
            except Exception:
                pass

        for phase, frame_idx in [("ED", int(ed_frame)), ("ES", int(es_frame))]:
            entry = {
                "image":       str(img_path),
                "frame_index": frame_idx,
                "patient":     pid,
                "phase":       phase,
                "dataset":     "MMs",
                "vendor":      meta.get("Vendor", "Unknown"),
                "center":      meta.get("Centre", "Unknown"),
                "pathology":   meta.get("Pathology", "NOR"),
            }
            if lbl_path.exists():
                entry["label"] = str(lbl_path)
            if lbl_path.exists() or split == "train_unlabeled":
                samples.append(entry)
    return samples


def _load_mms_csv(mms_root: Path) -> Dict[str, Dict]:
    meta = {}
    csv_files = list(mms_root.glob("*.csv"))
    if not csv_files:
        return meta
    with open(csv_files[0], newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("ExternalCode") or row.get("Patient") or
                   row.get("Code") or "").strip()
            if not pid:
                continue
            try:
                ed = int(float(row.get("ED", row.get("ED_frame", 0))))
                es = int(float(row.get("ES", row.get("ES_frame", 5))))
            except (ValueError, TypeError):
                ed, es = 0, 5
            meta[pid] = {
                "ED":        ed,
                "ES":        es,
                "Vendor":    (row.get("VendorName") or
                              row.get("Vendor", "Unknown")).strip(),
                "Centre":    str(row.get("Centre",
                                         row.get("Center", ""))).strip(),
                "Pathology": row.get("Pathology",
                                     row.get("Group", "NOR")).strip(),
            }
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# M&Ms-2 parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_mms2(data_root: str) -> List[Dict]:
    """
    MnM2/
      dataset_information.csv
      dataset/
        {NNN}/
          {NNN}_SA_ED.nii.gz        ← 3-D Short Axis ED image
          {NNN}_SA_ES.nii.gz        ← 3-D Short Axis ES image
          {NNN}_SA_ED_gt.nii.gz     ← ED label
          {NNN}_SA_ES_gt.nii.gz     ← ES label
          {NNN}_SA_CINE.nii.gz      (not used)
          {NNN}_LA_*.nii.gz         (not used — LA view)
    """
    mns2_root = Path(data_root) / "MnM2"
    if not mns2_root.exists():
        return []
    meta        = _load_mms2_csv(mns2_root)
    dataset_dir = mns2_root / "dataset"
    if not dataset_dir.exists():
        return []

    samples = []
    for patient_dir in sorted(dataset_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        for phase in ["ED", "ES"]:
            img_path = patient_dir / f"{pid}_SA_{phase}.nii.gz"
            lbl_path = patient_dir / f"{pid}_SA_{phase}_gt.nii.gz"
            if not img_path.exists() or not lbl_path.exists():
                continue
            m = meta.get(pid, {})
            samples.append({
                "image":     str(img_path),
                "label":     str(lbl_path),
                "patient":   pid,
                "phase":     phase,
                "dataset":   "MMs2",
                "vendor":    m.get("Vendor", "Unknown"),
                "center":    m.get("Centre", "Unknown"),
                "pathology": m.get("Pathology", "NOR"),
            })
    return samples


def _load_mms2_csv(mns2_root: Path) -> Dict[str, Dict]:
    meta = {}
    csv_files = list(mns2_root.glob("*.csv"))
    if not csv_files:
        return meta
    with open(csv_files[0], newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("PatientID") or row.get("ExternalCode") or
                   row.get("Patient") or "").strip().zfill(3)
            if not pid:
                continue
            meta[pid] = {
                "Vendor":    (row.get("Vendor") or
                              row.get("VendorName", "Unknown")).strip(),
                "Centre":    str(row.get("Centre",
                                         row.get("Center", ""))).strip(),
                "Pathology": row.get("Pathology",
                                     row.get("Group", "NOR")).strip(),
            }
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Data splits
# ─────────────────────────────────────────────────────────────────────────────

def build_data_splits(data_root: str) -> Dict[str, List[Dict]]:
    """Combine ACDC + M&Ms-1 + M&Ms-2 into train/val/test splits."""
    # ── ACDC ─────────────────────────────────────────────────────────────────
    acdc = parse_acdc(data_root)
    if not acdc:
        print(f"  ⚠  ACDC not found under {data_root}/ACDC-dataset/database/")

    patients = list({s["patient"] for s in acdc})
    np.random.seed(42)
    np.random.shuffle(patients)
    n       = len(patients)
    n_train = int(n * cfg.data.acdc_train_ratio)
    n_val   = int(n * cfg.data.acdc_val_ratio)
    train_pts = set(patients[:n_train])
    val_pts   = set(patients[n_train:n_train + n_val])
    test_pts  = set(patients[n_train + n_val:])

    splits = {
        "train": [s for s in acdc if s["patient"] in train_pts],
        "val":   [s for s in acdc if s["patient"] in val_pts],
        "test":  [s for s in acdc if s["patient"] in test_pts],
    }
    print(f"  ACDC  — train:{len(splits['train'])}  "
          f"val:{len(splits['val'])}  test:{len(splits['test'])}")

    # ── M&Ms-1 ───────────────────────────────────────────────────────────────
    if (Path(data_root) / "M&M1").exists():
        mms_tr  = parse_mms(data_root, "train")
        mms_val = parse_mms(data_root, "val")
        mms_tst = parse_mms(data_root, "test")
        splits["mms_train"] = mms_tr
        splits["mms_val"]   = mms_val
        splits["mms_test"]  = mms_tst
        splits["train"]    += mms_tr   # merge into training
        print(f"  M&Ms1 — train:{len(mms_tr)}  "
              f"val:{len(mms_val)}  test:{len(mms_tst)}")
    else:
        splits["mms_test"] = []
        splits["mms_val"]  = []

    # ── M&Ms-2 ───────────────────────────────────────────────────────────────
    if (Path(data_root) / "MnM2").exists():
        mms2 = parse_mms2(data_root)
        splits["mms2"] = mms2
        print(f"  M&Ms2 — {len(mms2)} held-out samples")
    else:
        splits["mms2"] = []

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# MONAI Transform pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_base_transforms() -> Compose:
    """Works for ACDC (3-D) and M&Ms-1 (4-D → 3-D via ExtractFramed)."""
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=False,
                   ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        ExtractFramed(keys=["image", "label"]),   # no-op for 3-D volumes
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=cfg.data.intensity_percentile[0],
            upper=cfg.data.intensity_percentile[1],
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ResizeWithPadOrCropd(
            keys=["image", "label"],
            spatial_size=cfg.data.spatial_size,
        ),
        EnsureTyped(keys=["image", "label"],
                    dtype=(torch.float32, torch.long)),
    ])


def get_augmentation_transforms() -> Compose:
    """Training augmentations (TPA topology filter applied in TPADataset)."""
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandAffined(
            keys=["image", "label"], prob=0.7,
            rotate_range=(np.deg2rad(cfg.data.rotation_range),
                          np.deg2rad(cfg.data.rotation_range), 0.0),
            scale_range=(0.1, 0.1, 0.0),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        Rand3DElasticd(
            keys=["image", "label"], prob=0.5,
            sigma_range=cfg.data.elastic_sigma,
            magnitude_range=cfg.data.elastic_alpha,
            mode=("bilinear", "nearest"),
        ),
        RandGaussianNoised(keys=["image"], prob=0.3, std=cfg.data.noise_std),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5)),
        RandScaleIntensityd(keys=["image"], prob=0.3, factors=0.2),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
    ])


def get_val_transforms() -> Compose:
    return get_base_transforms()


# ─────────────────────────────────────────────────────────────────────────────
# Topology-Preserving Augmentation Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _compute_2d_betti(binary: np.ndarray) -> Tuple[int, int]:
    """β₀ and β₁ of a 2-D binary mask via GUDHI Cubical Complex."""
    import gudhi
    cc = gudhi.CubicalComplex(
        dimensions=list(binary.shape),
        top_dimensional_cells=(1.0 - binary).flatten().tolist()
    )
    cc.compute_persistence()
    b = cc.betti_numbers()
    return (b[0] if len(b) > 0 else 0, b[1] if len(b) > 1 else 0)


class TPADataset(Dataset):
    """
    Dataset with topology-preserving augmentation filter.
    Rejects augmented samples whose mid-ventricular Betti numbers
    violate the anatomical invariants (LV β₀=1,β₁=0; Myo β₀=1,β₁=1).
    """
    def __init__(self, data, base_transforms, aug_transforms=None,
                 apply_tpa=True, max_attempts=20):
        super().__init__(data, transform=base_transforms)
        self.aug_transforms = aug_transforms
        self.apply_tpa      = apply_tpa
        self.max_attempts   = max_attempts
        self._stats         = {"accepted": 0, "rejected": 0}

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if self.aug_transforms is None or not self.apply_tpa:
            return item
        for _ in range(self.max_attempts):
            aug = self.aug_transforms(item.copy())
            if self._check_topology(aug["label"]):
                self._stats["accepted"] += 1
                return aug
            self._stats["rejected"] += 1
        self._stats["accepted"] += 1
        return item   # fall back to unaugmented

    def _check_topology(self, label: torch.Tensor) -> bool:
        try:
            label_np = label.squeeze().numpy()
            mid_z    = label_np.shape[-1] // 2
            mid      = label_np[..., mid_z]
            for sid, tb0, tb1 in [(1, 1, 0), (2, 1, 1), (3, 1, 0)]:
                binary = (mid == sid).astype(np.float32)
                if binary.sum() < 10:
                    continue
                b0, b1 = _compute_2d_betti(binary)
                if b0 != tb0 or b1 != tb1:
                    return False
            return True
        except Exception:
            return True


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(splits: Dict[str, List[Dict]]) -> Dict[str, DataLoader]:
    base_tf = get_base_transforms()
    aug_tf  = get_augmentation_transforms()
    val_tf  = get_val_transforms()

    train_ds = TPADataset(splits["train"], base_tf, aug_tf, apply_tpa=True)
    val_ds   = CacheDataset(splits["val"],  val_tf,
                            cache_rate=1.0,
                            num_workers=cfg.training.num_workers)
    test_ds  = CacheDataset(splits["test"], val_tf,
                            cache_rate=1.0,
                            num_workers=cfg.training.num_workers)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.training.batch_size,
                            shuffle=True, num_workers=cfg.training.num_workers,
                            pin_memory=True, drop_last=True),
        "val":   DataLoader(val_ds,  batch_size=1, shuffle=False,
                            num_workers=cfg.training.num_workers),
        "test":  DataLoader(test_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.training.num_workers),
    }
    for key, split_key in [("mms", "mms_test"),
                            ("mms_val", "mms_val"),
                            ("mms2", "mms2")]:
        data = splits.get(split_key, [])
        if data:
            ds = CacheDataset(data, val_tf, cache_rate=0.0,
                              num_workers=cfg.training.num_workers)
            loaders[key] = DataLoader(ds, batch_size=1, shuffle=False,
                                      num_workers=cfg.training.num_workers)
    return loaders
