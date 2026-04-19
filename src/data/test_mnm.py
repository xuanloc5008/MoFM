"""
M&Ms-1 and M&Ms-2 Dataset Loaders  (FIXED)
--------------------------------------------
Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Segmentation Challenge

Verified directory layout
--------------------------
M&M1/
├── 211230_M&M...endataset.csv      <- ExternalCode, ED, ES, VendorName, Centre, ...
├── Testing/    {ID}/ {ID}_sa.nii.gz   {ID}_sa_gt.nii.gz
├── Training/
│   ├── Labeled/    {ID}/ {ID}_sa.nii.gz   {ID}_sa_gt.nii.gz
│   └── Unlabeled/  {ID}/ {ID}_sa.nii.gz
└── Validation/ {ID}/ {ID}_sa.nii.gz   {ID}_sa_gt.nii.gz

MnM2/
├── dataset_information.csv         <- SubjectID/Code, VendorName, ...
└── dataset/
    └── {ID}/
        ├── {ID}_SA_ED.nii.gz       <- 3-D H x W x Z
        ├── {ID}_SA_ES.nii.gz
        ├── {ID}_SA_ED_gt.nii.gz
        └── {ID}_SA_ES_gt.nii.gz

BUG-FIXES vs original
-----------------------
1. M&Ms-1 Training split now resolves to Training/Labeled (not Training/).
2. M&Ms-1 vendor read from CSV "VendorName" column (IDs are numeric, not prefix-encoded).
3. Replaced fragile fallback logic with explicit split-name -> path map.
4. API accepts "Training" / "Validation" / "Testing" as logical names.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers shared by both datasets
# ─────────────────────────────────────────────────────────────────────────────

def _visible_csvs(root: Path) -> List[Path]:
    """Skip hidden metadata sidecars such as macOS ._* files."""
    return sorted(
        path for path in root.glob("*.csv")
        if not path.name.startswith(".")
    )


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_column(columns, *candidates: str) -> Optional[str]:
    normalized = {
        _normalize_column_name(column): column
        for column in columns
    }
    for candidate in candidates:
        match = normalized.get(_normalize_column_name(candidate))
        if match is not None:
            return match
    return None


def _normalize_subject_id(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Load CSVs robustly across UTF-8 variants and legacy encodings."""
    last_exc = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc if last_exc is not None else RuntimeError(f"Failed to read {path}")


def _load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img  = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    sp   = np.array(img.header.get_zooms()[:3], dtype=np.float32)
    return data, sp


def _extract_2d_slices(
    frame_img:  np.ndarray,   # H x W x Z  (single phase)
    frame_lbl:  np.ndarray,   # H x W x Z
    spacing:    np.ndarray,
    patient_id: str,
    phase:      str,
    vendor:     str,
    dataset:    str,
    split:      str,
) -> List[Dict]:
    """Convert 3-D volume into list of annotated 2-D slice dicts."""
    out = []
    for s in range(frame_img.shape[2]):
        lbl_s = np.round(frame_lbl[:, :, s]).astype(np.int64)
        if lbl_s.max() == 0:          # no annotation on this slice
            continue
        img_s = frame_img[:, :, s]
        out.append({
            "image":      img_s[np.newaxis].astype(np.float32),  # 1 x H x W
            "label":      lbl_s,                                  # H x W
            "patient_id": patient_id,
            "phase":      phase,
            "slice_idx":  s,
            "spacing":    spacing,
            "dataset":    dataset,
            "vendor":     vendor,
            "split":      split,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  M&Ms-1
# ─────────────────────────────────────────────────────────────────────────────

# BUG-FIX 1 & 3: explicit mapping; "Training" → Training/Labeled
_MNM1_SPLIT_MAP: Dict[str, str] = {
    "Testing":            "Testing",
    "Validation":         "Validation",
    "Training":           "Training/Labeled",
    "Training_Labeled":   "Training/Labeled",
    "Training_Unlabeled": "Training/Unlabeled",
}


def _load_mnm1_csv(root: Path) -> Optional[pd.DataFrame]:
    """Find and load the metadata CSV at root level. Returns indexed DataFrame."""
    csvs = _visible_csvs(root)
    if not csvs:
        logger.warning("M&Ms-1: no CSV at root — vendor/ED/ES will use fallbacks")
        return None

    df = _read_csv_with_fallback(csvs[0])
    df.columns = [c.strip() for c in df.columns]

    # Subject-ID column: prefer ExternalCode, else first column
    id_col = _find_column(df.columns, "ExternalCode", "External code") or df.columns[0]
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df.set_index(id_col)

    ed_col = _find_column(df.columns, "ED")
    es_col = _find_column(df.columns, "ES")
    if ed_col is not None and es_col is not None:
        frame_min = min(
            pd.to_numeric(df[ed_col], errors="coerce").min(),
            pd.to_numeric(df[es_col], errors="coerce").min(),
        )
        df.attrs["mnm1_ed_col"] = ed_col
        df.attrs["mnm1_es_col"] = es_col
        df.attrs["mnm1_frame_base"] = 0 if pd.notna(frame_min) and frame_min == 0 else 1

    logger.info(f"M&Ms-1 CSV: {csvs[0].name}  |  {len(df)} subjects  |  cols={list(df.columns)}")
    return df


def _vendor_from_csv(sid: str, df: Optional[pd.DataFrame]) -> str:
    """BUG-FIX 2: vendor comes from CSV, not from ID character prefix."""
    if df is None or sid not in df.index:
        return "Unknown"
    row = df.loc[sid]
    vendor_col = _find_column(df.columns, "VendorName", "Vendor")
    if vendor_col is not None and pd.notna(row.get(vendor_col, None)):
        return str(row[vendor_col]).strip()
    return "Unknown"


def _ed_es_from_csv(
    sid: str, df: Optional[pd.DataFrame], n_frames: int
) -> Tuple[int, int]:
    """Return 0-based (ed_idx, es_idx). Falls back to (0, mid) if not in CSV."""
    if df is not None and sid in df.index:
        row = df.loc[sid]
        ed_col = df.attrs.get("mnm1_ed_col") or _find_column(df.columns, "ED")
        es_col = df.attrs.get("mnm1_es_col") or _find_column(df.columns, "ES")
        frame_base = df.attrs.get("mnm1_frame_base", 1)
        try:
            if ed_col is None or es_col is None:
                raise KeyError("ED/ES column missing")
            ed_raw = int(row[ed_col])
            es_raw = int(row[es_col])
            if frame_base == 0:
                ed = max(0, min(ed_raw, n_frames - 1))
                es = max(0, min(es_raw, n_frames - 1))
            else:
                ed = max(0, min(ed_raw - 1, n_frames - 1))
                es = max(0, min(es_raw - 1, n_frames - 1))
            return ed, es
        except (KeyError, ValueError, TypeError):
            pass
    return 0, max(0, n_frames // 2 - 1)


def collect_mnm1_slices(
    mnm1_root:  str,
    splits:     List[str] = ("Testing", "Validation"),
    require_gt: bool = True,
) -> List[Dict]:
    """
    Load 2-D slices from M&Ms-1.

    splits : list of logical split names to load.
      Valid values: "Testing" | "Validation" | "Training" |
                    "Training_Labeled" | "Training_Unlabeled"
    require_gt : skip subjects with no GT file (always True for Testing/Validation).
    """
    root = Path(mnm1_root)
    df   = _load_mnm1_csv(root)
    all_slices: List[Dict] = []

    for split_name in splits:
        # BUG-FIX 3: use explicit map, no fallback guessing
        rel = _MNM1_SPLIT_MAP.get(split_name)
        if rel is None:
            logger.warning(
                f"M&Ms-1: unknown split name '{split_name}'. "
                f"Valid: {list(_MNM1_SPLIT_MAP)}"
            )
            continue

        split_dir = root / rel
        if not split_dir.exists():
            logger.warning(f"M&Ms-1: directory not found → {split_dir}")
            continue

        n_split = 0
        for subj_dir in sorted(split_dir.iterdir()):
            if not subj_dir.is_dir() or subj_dir.name.startswith("."):
                continue
            sid = subj_dir.name

            img_path = subj_dir / f"{sid}_sa.nii.gz"
            gt_path  = subj_dir / f"{sid}_sa_gt.nii.gz"

            if not img_path.exists():
                continue
            if require_gt and not gt_path.exists():
                logger.debug(f"M&Ms-1: {sid} — GT missing, skip")
                continue

            try:
                volume, spacing = _load_nifti(str(img_path))
            except Exception as exc:
                logger.warning(f"M&Ms-1: {sid} image load error — {exc}")
                continue

            if volume.ndim != 4:
                # Should be H x W x Z x T; warn if unexpected
                logger.warning(
                    f"M&Ms-1: {sid} expected 4-D volume, got ndim={volume.ndim}"
                )
                continue

            n_frames = volume.shape[3]   # time dimension

            # Load GT (or zero volume if absent)
            if gt_path.exists():
                try:
                    label_vol, _ = _load_nifti(str(gt_path))
                except Exception as exc:
                    logger.warning(f"M&Ms-1: {sid} GT load error — {exc}")
                    continue
            else:
                label_vol = np.zeros_like(volume)

            ed_idx, es_idx = _ed_es_from_csv(sid, df, n_frames)
            vendor          = _vendor_from_csv(sid, df)    # BUG-FIX 2

            for phase, fidx in [("ED", ed_idx), ("ES", es_idx)]:
                frame_img = volume[:, :, :, fidx]                          # H x W x Z
                frame_lbl = (
                    label_vol[:, :, :, fidx]
                    if label_vol.ndim == 4
                    else label_vol
                )
                slices = _extract_2d_slices(
                    frame_img, frame_lbl, spacing,
                    sid, phase, vendor, "MnM1", split_name,
                )
                all_slices.extend(slices)
                n_split += len(slices)

        logger.info(f"  M&Ms-1 [{split_name}] ({rel}): {n_split} slices")

    logger.info(f"M&Ms-1 TOTAL: {len(all_slices)} slices from splits={splits}")
    return all_slices


# ─────────────────────────────────────────────────────────────────────────────
#  M&Ms-2
# ─────────────────────────────────────────────────────────────────────────────

def _load_mnm2_info(root: Path) -> Optional[pd.DataFrame]:
    """
    Load dataset_information.csv for M&Ms-2.

    IMPORTANT: SubjectID can be stored as integer (27) in the CSV
    while directory names use zero-padded strings ('027').
    We build a dual-key index: both '27' and '027' map to the same row.
    """
    csv_path = root / "dataset_information.csv"
    if not csv_path.exists():
        logger.warning("MnM2: dataset_information.csv not found")
        return None

    df = _read_csv_with_fallback(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Detect subject-ID column
    id_col = _find_column(df.columns, "SubjectID", "SUBJECT_CODE", "Code", "ExternalCode")
    if id_col is None:
        id_col = df.columns[0]

    # Normalise: keep original string, also add zero-stripped version as alias
    raw_ids = df[id_col].apply(_normalize_subject_id)
    df = df.copy()
    df["__sid__"] = raw_ids

    # Build expanded index: for every numeric ID 'N', also register 'NNN' (3-digit padded)
    rows = []
    for _, row in df.iterrows():
        sid_raw = str(row["__sid__"])
        rows.append(row)
        # If numeric, also add zero-padded variant
        try:
            n = int(sid_raw)
            for width in (2, 3, 4):
                padded = str(n).zfill(width)
                if padded != sid_raw:
                    new_row = row.copy()
                    new_row["__sid__"] = padded
                    rows.append(new_row)
        except ValueError:
            pass   # alphanumeric ID – no padding needed

    expanded = pd.DataFrame(rows)
    expanded = expanded.set_index("__sid__")
    # Drop duplicate index entries (keep first)
    expanded = expanded[~expanded.index.duplicated(keep="first")]
    logger.info(
        f"MnM2 info CSV: {len(df)} subjects "
        f"(expanded index: {len(expanded)} keys) | cols={list(df.columns)}"
    )
    return expanded


def _vendor_mnm2(sid: str, df: Optional[pd.DataFrame]) -> str:
    """Look up vendor, tolerating zero-padding differences."""
    if df is None:
        return "Unknown"
    vendor_col = _find_column(df.columns, "VendorName", "Vendor", "VENDOR")
    if vendor_col is None:
        return "Unknown"
    # Try exact sid, then strip leading zeros (numeric ids)
    candidates = [sid]
    try:
        candidates.append(str(int(sid)))   # '027' → '27'
    except ValueError:
        pass
    for key in candidates:
        if key in df.index:
            row = df.loc[key]
            if pd.notna(row.get(vendor_col, None)):
                return str(row[vendor_col]).strip()
    return "Unknown"


def collect_mnm2_slices(mnm2_root: str) -> List[Dict]:
    """
    Load 2-D slices from M&Ms-2.
    Each subject contains pre-extracted 3-D ED and ES volumes.
    CINE (*_SA_CINE*) and long-axis (*_LA_*) files are ignored.
    """
    root        = Path(mnm2_root)
    dataset_dir = root / "dataset"
    info_df     = _load_mnm2_info(root)

    if not dataset_dir.exists():
        logger.error(f"MnM2: dataset/ not found at {root}")
        return []

    all_slices: List[Dict] = []

    for subj_dir in sorted(dataset_dir.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name.startswith("."):
            continue
        sid    = subj_dir.name
        vendor = _vendor_mnm2(sid, info_df)

        for phase, img_name, gt_name in [
            ("ED", f"{sid}_SA_ED.nii.gz",    f"{sid}_SA_ED_gt.nii.gz"),
            ("ES", f"{sid}_SA_ES.nii.gz",    f"{sid}_SA_ES_gt.nii.gz"),
        ]:
            img_path = subj_dir / img_name
            gt_path  = subj_dir / gt_name

            if not img_path.exists() or not gt_path.exists():
                logger.debug(f"MnM2: {sid} {phase} — file(s) missing, skip")
                continue

            try:
                volume,    spacing = _load_nifti(str(img_path))
                label_vol, _       = _load_nifti(str(gt_path))
            except Exception as exc:
                logger.warning(f"MnM2: {sid} {phase} load error — {exc}")
                continue

            # MnM2 SA ED/ES are 3-D: H x W x Z
            if volume.ndim != 3:
                logger.warning(
                    f"MnM2: {sid} {phase} expected 3-D, got ndim={volume.ndim}"
                )
                continue

            slices = _extract_2d_slices(
                volume, label_vol, spacing,
                sid, phase, vendor, "MnM2", "dataset",
            )
            all_slices.extend(slices)

    logger.info(f"MnM2 TOTAL: {len(all_slices)} slices")
    return all_slices


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DomainShiftDataset(Dataset):
    """
    PyTorch Dataset for M&Ms-1 / M&Ms-2 slices.
    Drop-in compatible with ACDC SliceDataset.
    """

    def __init__(self, slices: List[Dict], transforms=None):
        self.slices     = slices
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> Dict:
        raw = self.slices[idx]

        sample = {
            "image":      raw["image"].copy(),
            "label":      raw["label"][np.newaxis].copy(),
            "patient_id": raw.get("patient_id", ""),
            "phase":      raw.get("phase", ""),
            "slice_idx":  raw.get("slice_idx", -1),
            "spacing":    raw.get("spacing", np.array([1.5, 1.5, 8.0], dtype=np.float32)),
            "dataset":    raw.get("dataset", ""),
            "vendor":     raw.get("vendor", "Unknown"),
            "split":      raw.get("split", ""),
        }

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

        return sample

    def get_vendor_groups(self) -> Dict[str, List[int]]:
        """Return {vendor: [indices]} for per-vendor evaluation."""
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(self.slices):
            groups.setdefault(s.get("vendor", "Unknown"), []).append(i)
        return groups

    def get_patient_groups(self) -> Dict[str, List[int]]:
        """Return {patient_id: [indices]} for patient-level aggregation."""
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(self.slices):
            groups.setdefault(s.get("patient_id", "unknown"), []).append(i)
        return groups
