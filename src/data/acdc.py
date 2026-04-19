"""
ACDC Dataset Loader
-------------------
Automated Cardiac Diagnosis Challenge (ACDC) - MICCAI 2017
4 classes: 0=Background, 1=RV, 2=Myocardium, 3=LV
Extracts ED and ES frames for each patient based on Info.cfg
"""
import os
import json
import configparser
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose
import logging

from src.data.context import build_context_indices
from src.data.preprocessing import harmonize_volume

logger = logging.getLogger(__name__)


def _resolve_acdc_database_dir(acdc_root: str) -> Path:
    """
    Accept either the dataset root (.../ACDC-dataset) or the database directory
    itself (.../ACDC-dataset/database).
    """
    root = Path(acdc_root)
    for candidate in (root, root / "database"):
        if not candidate.exists():
            continue
        if candidate.name == "database":
            return candidate
        if (candidate / "training").exists() or (candidate / "testing").exists():
            return candidate
    return root / "database"


def parse_acdc_info(info_path: str) -> Dict:
    """
    Parse patient Info.cfg to get ED/ES frame indices and metadata.

    ACDC Info.cfg format (configparser reads [Info] section; keys are lowercased):
        [Info]
        Group: DCM
        Height: 184.0
        Weight: 95.0
        NbFrame: 30      ← lowercased key becomes 'nbframe'
        ED: 1            ← key becomes 'ed'
        ES: 14           ← key becomes 'es'

    Robustness: if [Info] section is missing, fall back to flat key=value parsing.
    """
    cfg = configparser.RawConfigParser()
    try:
        cfg.read(info_path, encoding="utf-8")
    except configparser.MissingSectionHeaderError:
        # Headerless file: inject [Info] and re-parse
        with open(info_path, encoding="utf-8", errors="replace") as f:
            lines = [l for l in f if l.strip() and not l.strip().startswith("#")]
        cfg.read_string("[Info]\n" + "".join(lines))

    # Resolve section: prefer 'Info', then first section, then empty
    if "Info" in cfg:
        info = dict(cfg["Info"])
    elif cfg.sections():
        info = dict(cfg[cfg.sections()[0]])
    else:
        info = {}

    return {
        "ed_frame":  int(info.get("ed",       1)),
        "es_frame":  int(info.get("es",        1)),
        "group":     info.get("group",         "unknown"),
        "height":    float(info.get("height",  0)),
        "weight":    float(info.get("weight",  0)),
        # NbFrame → lowercased to 'nbframe' by configparser
        "nb_frames": int(info.get("nbframe",  info.get("nb_frames", 1))),
    }


def load_nifti_as_numpy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file → (data HWD, voxel_spacing XYZ)."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)   # H x W x D
    spacing = img.header.get_zooms()[:3]     # (sx, sy, sz)
    return data, np.array(spacing, dtype=np.float32)


class ACDCPatient:
    """Container for one ACDC patient's ED/ES volumes + labels."""

    def __init__(self, patient_dir: str, preprocess_cfg: Optional[Dict] = None):
        self.patient_dir = Path(patient_dir)
        self.patient_id  = self.patient_dir.name
        self.preprocess_cfg = preprocess_cfg or {}

        info_file = self.patient_dir / "Info.cfg"
        if not info_file.exists():
            raise FileNotFoundError(f"Info.cfg not found in {patient_dir}")
        self.info = parse_acdc_info(str(info_file))

        self._load_volumes()

    def _load_volumes(self):
        pid = self.patient_id
        ed  = self.info["ed_frame"]
        es  = self.info["es_frame"]

        ed_img = self.patient_dir / f"{pid}_frame{ed:02d}.nii.gz"
        ed_gt  = self.patient_dir / f"{pid}_frame{ed:02d}_gt.nii.gz"
        es_img = self.patient_dir / f"{pid}_frame{es:02d}.nii.gz"
        es_gt  = self.patient_dir / f"{pid}_frame{es:02d}_gt.nii.gz"

        for p in [ed_img, ed_gt, es_img, es_gt]:
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        self.ed_image,   self.voxel_spacing = load_nifti_as_numpy(str(ed_img))
        self.ed_label, _ = load_nifti_as_numpy(str(ed_gt))
        self.es_image, _ = load_nifti_as_numpy(str(es_img))
        self.es_label, _ = load_nifti_as_numpy(str(es_gt))

        self.ed_image, self.ed_label, ed_spacing = harmonize_volume(
            self.ed_image,
            self.ed_label,
            self.voxel_spacing,
            self.preprocess_cfg,
        )
        self.es_image, self.es_label, es_spacing = harmonize_volume(
            self.es_image,
            self.es_label,
            self.voxel_spacing,
            self.preprocess_cfg,
        )

        # ED and ES should share the same in-plane spacing after harmonization.
        self.voxel_spacing = np.asarray(ed_spacing, dtype=np.float32)
        self.es_voxel_spacing = np.asarray(es_spacing, dtype=np.float32)

        # Round labels to int
        self.ed_label = np.round(self.ed_label).astype(np.int64)
        self.es_label = np.round(self.es_label).astype(np.int64)

    def get_2d_slices(self) -> List[Dict]:
        """Extract all valid 2D slices (both ED and ES)."""
        slices = []
        for phase, image, label in [("ED", self.ed_image, self.ed_label),
                                     ("ES", self.es_image, self.es_label)]:
            n_slices = image.shape[2]
            for s in range(n_slices):
                img_slice = image[:, :, s]   # H x W
                lbl_slice = label[:, :, s]
                # Skip slices with no annotation
                if lbl_slice.max() == 0:
                    continue
                slices.append({
                    "image":       img_slice[np.newaxis],   # 1 x H x W
                    "label":       lbl_slice,               # H x W
                    "patient_id":  self.patient_id,
                    "phase":       phase,
                    "slice_idx":   s,
                    "spacing":     self.voxel_spacing,
                    "group":       self.info["group"],
                })
        return slices


def collect_acdc_slices(
    acdc_root: str,
    split: str = "training",
    preprocess_cfg: Optional[Dict] = None,
) -> List[Dict]:
    """
    Walk ACDC directory and collect all 2-D slices.
    split: 'training' | 'testing'
    """
    base = _resolve_acdc_database_dir(acdc_root) / split
    if not base.exists():
        raise FileNotFoundError(f"ACDC {split} dir not found: {base}")

    all_slices = []
    patient_dirs = sorted(
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    for pdir in patient_dirs:
        try:
            patient = ACDCPatient(str(pdir), preprocess_cfg=preprocess_cfg)
            all_slices.extend(patient.get_2d_slices())
        except Exception as e:
            logger.warning(f"Skip {pdir.name}: {e}")
    logger.info(f"ACDC {split}: {len(all_slices)} valid 2D slices")
    return all_slices


def acdc_train_val_split(
    acdc_root: str,
    val_ratio: float = 0.20,
    seed: int = 42,
    preprocess_cfg: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Patient-level split to avoid data leakage.
    Returns (train_slices, val_slices).
    """
    base = _resolve_acdc_database_dir(acdc_root) / "training"
    if not base.exists():
        raise FileNotFoundError(f"ACDC training dir not found: {base}")

    patient_dirs = sorted(
        p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    # stratify by pathology group if possible
    groups = []
    valid_dirs = []
    for pdir in patient_dirs:
        info_file = pdir / "Info.cfg"
        if info_file.exists():
            info = parse_acdc_info(str(info_file))
            groups.append(info["group"])
            valid_dirs.append(pdir)

    train_dirs, val_dirs = train_test_split(
        valid_dirs,
        test_size=val_ratio,
        stratify=groups,
        random_state=seed,
    )

    def load_patient_slices(dirs):
        slices = []
        for pdir in dirs:
            try:
                patient = ACDCPatient(str(pdir), preprocess_cfg=preprocess_cfg)
                slices.extend(patient.get_2d_slices())
            except Exception as e:
                logger.warning(f"Skip {pdir.name}: {e}")
        return slices

    train_slices = load_patient_slices(train_dirs)
    val_slices   = load_patient_slices(val_dirs)
    logger.info(f"ACDC split → train: {len(train_slices)} slices "
                f"({len(train_dirs)} patients), "
                f"val: {len(val_slices)} slices ({len(val_dirs)} patients)")
    return train_slices, val_slices


class SliceDataset(Dataset):
    """
    Generic 2-D slice dataset compatible with MONAI transforms.

    Image and label are kept as numpy arrays until after transforms run.
    MONAI's CastToTyped + EnsureTyped (in the transform pipelines) handle
    final type conversion. This avoids double-conversion bugs with spatial
    transforms that expect numpy input.
    """

    def __init__(
        self,
        slices: List[Dict],
        transforms: Optional[Compose] = None,
        context_slices: int = 1,
    ):
        self.slices     = slices
        self.transforms = transforms
        self.context_slices = int(context_slices)
        self.context_indices = build_context_indices(self.slices, self.context_slices)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict:
        raw = self.slices[idx]
        ctx_indices = self.context_indices[idx]
        image_stack = np.concatenate(
            [self.slices[j]["image"] for j in ctx_indices],
            axis=0,
        ).astype(np.float32, copy=False)

        # Shallow copy; keep numpy arrays for MONAI transform compatibility
        sample = {
            "image":      image_stack.copy(),   # numpy (C, H, W) float32
            "label":      raw["label"][np.newaxis].copy(),   # numpy (1, H, W) int64
            "patient_id": raw.get("patient_id", ""),
            "phase":      raw.get("phase", ""),
            "slice_idx":  raw.get("slice_idx", -1),
            "spacing":    raw.get("spacing", np.array([1.5, 1.5, 8.0], dtype=np.float32)),
            "group":      raw.get("group", ""),
        }
        if "topo_vec" in raw:
            sample["topo_vec"] = np.asarray(raw["topo_vec"], dtype=np.float32).copy()

        if self.transforms is not None:
            sample = self.transforms(sample)

        # MONAI spatial transforms expect channel-first tensors; squeeze the
        # temporary label channel again so losses/metrics still see H x W.
        if getattr(sample["label"], "ndim", 0) == 3 and sample["label"].shape[0] == 1:
            sample["label"] = sample["label"][0]

        # After transforms: convert remaining non-tensor items
        if not isinstance(sample["image"], torch.Tensor):
            sample["image"] = torch.tensor(sample["image"], dtype=torch.float32)
        if not isinstance(sample["label"], torch.Tensor):
            sample["label"] = torch.tensor(sample["label"], dtype=torch.long)

        # Convert spacing to tensor for DataLoader collation
        if not isinstance(sample["spacing"], torch.Tensor):
            sample["spacing"] = torch.tensor(
                sample["spacing"], dtype=torch.float32
            )
        if "topo_vec" in sample and not isinstance(sample["topo_vec"], torch.Tensor):
            sample["topo_vec"] = torch.tensor(sample["topo_vec"], dtype=torch.float32)

        return sample


class PreprocessedSliceDataset(Dataset):
    """
    Dataset backed by offline-preprocessed `.npz` slice files.

    Each file stores deterministic preprocessing results so training only needs
    online augmentation and tensor conversion.
    """

    def __init__(
        self,
        root_dir: str,
        transforms: Optional[Compose] = None,
        context_slices: int = 1,
        require_barcodes: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.context_slices = int(context_slices)
        self.require_barcodes = bool(require_barcodes)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Preprocessed slice directory not found: {self.root_dir}")

        index_path = self.root_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            self.files = [self.root_dir / item["file"] for item in index]
            self.records = index
        else:
            self.files = sorted(self.root_dir.glob("*.npz"))
            self.records = []
            for file in self.files:
                with np.load(file, allow_pickle=False) as raw:
                    self.records.append({
                        "file": file.name,
                        "patient_id": str(raw["patient_id"]),
                        "phase": str(raw["phase"]),
                        "slice_idx": int(raw["slice_idx"]),
                        "group": str(raw["group"]),
                    })

        if not self.files:
            raise FileNotFoundError(f"No preprocessed .npz files found in {self.root_dir}")

        with np.load(self.files[0], allow_pickle=False) as raw:
            if "topo_vec" not in raw:
                raise ValueError(
                    f"Preprocessed file {self.files[0]} is missing 'topo_vec'. "
                    "Rebuild the cache with scripts/preprocess_acdc.py."
                )
            if self.require_barcodes:
                required = {"barcode_h0", "barcode_h0_count", "barcode_h1", "barcode_h1_count"}
                missing = sorted(required.difference(raw.files))
                if missing:
                    raise ValueError(
                        f"Preprocessed file {self.files[0]} is missing barcode fields {missing}. "
                        "Rebuild the cache with scripts/preprocess_acdc.py --overwrite."
                    )
        self.context_indices = build_context_indices(self.records, self.context_slices)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        ctx_indices = self.context_indices[idx]
        images = []
        center_raw = None
        for j in ctx_indices:
            with np.load(self.files[j], allow_pickle=False) as raw:
                images.append(raw["image"].astype(np.float32, copy=True))
                if j == idx:
                    center_raw = {
                        "label": raw["label"][np.newaxis].copy(),
                        "patient_id": str(raw["patient_id"]),
                        "phase": str(raw["phase"]),
                        "slice_idx": int(raw["slice_idx"]),
                        "spacing": raw["spacing"].astype(np.float32, copy=True),
                        "group": str(raw["group"]),
                        "topo_vec": raw["topo_vec"].astype(np.float32, copy=True),
                    }
                    if "barcode_h0" in raw:
                        center_raw["barcode_h0"] = raw["barcode_h0"].astype(np.float32, copy=True)
                        center_raw["barcode_h0_count"] = np.int32(raw["barcode_h0_count"])
                    if "barcode_h1" in raw:
                        center_raw["barcode_h1"] = raw["barcode_h1"].astype(np.float32, copy=True)
                        center_raw["barcode_h1_count"] = np.int32(raw["barcode_h1_count"])

        if center_raw is None:
            raise RuntimeError(f"Failed to load center slice for index {idx}")

        sample = {
            "image": np.concatenate(images, axis=0),
            "label": center_raw["label"],
            "patient_id": center_raw["patient_id"],
            "phase": center_raw["phase"],
            "slice_idx": center_raw["slice_idx"],
            "spacing": center_raw["spacing"],
            "group": center_raw["group"],
            "topo_vec": center_raw["topo_vec"],
        }
        if "barcode_h0" in center_raw:
            sample["barcode_h0"] = center_raw["barcode_h0"]
            sample["barcode_h0_count"] = center_raw["barcode_h0_count"]
        if "barcode_h1" in center_raw:
            sample["barcode_h1"] = center_raw["barcode_h1"]
            sample["barcode_h1_count"] = center_raw["barcode_h1_count"]

        if self.transforms is not None:
            sample = self.transforms(sample)

        if getattr(sample["label"], "ndim", 0) == 3 and sample["label"].shape[0] == 1:
            sample["label"] = sample["label"][0]

        if not isinstance(sample["image"], torch.Tensor):
            sample["image"] = torch.tensor(sample["image"], dtype=torch.float32)
        if not isinstance(sample["label"], torch.Tensor):
            sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        if not isinstance(sample["spacing"], torch.Tensor):
            sample["spacing"] = torch.tensor(sample["spacing"], dtype=torch.float32)
        if "topo_vec" in sample and not isinstance(sample["topo_vec"], torch.Tensor):
            sample["topo_vec"] = torch.tensor(sample["topo_vec"], dtype=torch.float32)
        if "barcode_h0" in sample and not isinstance(sample["barcode_h0"], torch.Tensor):
            sample["barcode_h0"] = torch.tensor(sample["barcode_h0"], dtype=torch.float32)
        if "barcode_h0_count" in sample and not isinstance(sample["barcode_h0_count"], torch.Tensor):
            sample["barcode_h0_count"] = torch.tensor(sample["barcode_h0_count"], dtype=torch.long)
        if "barcode_h1" in sample and not isinstance(sample["barcode_h1"], torch.Tensor):
            sample["barcode_h1"] = torch.tensor(sample["barcode_h1"], dtype=torch.float32)
        if "barcode_h1_count" in sample and not isinstance(sample["barcode_h1_count"], torch.Tensor):
            sample["barcode_h1_count"] = torch.tensor(sample["barcode_h1_count"], dtype=torch.long)

        return sample
