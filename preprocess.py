"""
Preprocessing Pipeline for MOFMTDA-TTA.
Medical-image-standard preprocessing for cardiac MRI segmentation.

Steps:
  1. Verify ACDC data structure
  2. Per-volume intensity clipping (percentile 0.5–99.5)
  3. Pixel spacing resampling to isotropic target spacing
  4. Per-volume z-score normalization then rescale to [0,1]
  5. ROI cropping around cardiac region (from ground-truth center of mass)
  6. Resize to target size (256×256)
  7. Extract 2D slices → save as .npz cache
  8. Filter empty slices (no cardiac structure)
  9. Precompute ground-truth Persistence Diagrams
  10. Compute and save dataset statistics

Usage:
    python preprocess.py --data_root /path/to/ACDC/database/training
    python preprocess.py --data_root /path/to/ACDC/database/training --target_spacing 1.25 1.25
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pickle
from tqdm import tqdm
from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage import center_of_mass
import warnings

warnings.filterwarnings("ignore")


# ================================================================
# Step 1: Verify data structure
# ================================================================

def verify_data(data_root: str) -> Dict:
    """Verify ACDC data and report statistics."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    patients = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("patient")])
    if not patients:
        raise ValueError(f"No patient directories in {root}")

    print(f"Found {len(patients)} patients")

    stats = {"total": len(patients), "patients": [], "errors": [],
             "spacings": [], "shapes": []}

    for pdir in patients:
        pid = pdir.name
        info_file = pdir / "Info.cfg"
        if not info_file.exists():
            stats["errors"].append(f"{pid}: Info.cfg missing")
            continue

        info = parse_info_cfg(info_file)
        ed, es = info.get("ED"), info.get("ES")
        if ed is None or es is None:
            stats["errors"].append(f"{pid}: ED/ES not in Info.cfg")
            continue

        ok = True
        for phase, fidx in [("ED", ed), ("ES", es)]:
            for suffix in ["", "_gt"]:
                f = pdir / f"{pid}_frame{fidx:02d}{suffix}.nii.gz"
                if not f.exists():
                    stats["errors"].append(f"{pid}: {f.name} missing")
                    ok = False

        if ok:
            nii = nib.load(str(pdir / f"{pid}_frame{ed:02d}.nii.gz"))
            spacing = nii.header.get_zooms()
            stats["patients"].append({
                "id": pid, "group": info.get("Group", "?"),
                "ed": ed, "es": es,
                "shape": list(nii.shape), "spacing": [float(s) for s in spacing],
            })
            stats["spacings"].append([float(s) for s in spacing[:2]])
            stats["shapes"].append(list(nii.shape[:2]))

    print(f"  Valid: {len(stats['patients'])}, Errors: {len(stats['errors'])}")
    if stats["errors"]:
        for e in stats["errors"][:3]:
            print(f"    {e}")

    # Spacing statistics
    spacings = np.array(stats["spacings"])
    if len(spacings) > 0:
        print(f"  Pixel spacing — mean: {spacings.mean(0)}, "
              f"range: [{spacings.min(0)}, {spacings.max(0)}]")

    # Group distribution
    groups = {}
    for p in stats["patients"]:
        groups[p["group"]] = groups.get(p["group"], 0) + 1
    print(f"  Groups: {groups}")

    return stats


def parse_info_cfg(path: Path) -> Dict:
    info = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k in ["ED", "ES", "NbFrame"]:
                    info[k] = int(v)
                elif k in ["Height", "Weight"]:
                    info[k] = float(v)
                else:
                    info[k] = v
    return info


# ================================================================
# Preprocessing functions
# ================================================================

def clip_intensity(volume: np.ndarray, lower_pct: float = 0.5,
                   upper_pct: float = 99.5) -> np.ndarray:
    """Clip intensities at percentiles to remove outliers.
    Standard in medical imaging to handle MRI intensity artifacts."""
    lo = np.percentile(volume, lower_pct)
    hi = np.percentile(volume, upper_pct)
    return np.clip(volume, lo, hi)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Per-volume z-score normalization, then rescale to [0, 1].
    
    Z-score first (mean=0, std=1) to standardize across patients,
    then rescale to [0, 1] for network input.
    """
    # Z-score
    mu = volume.mean()
    std = volume.std()
    if std < 1e-8:
        return np.zeros_like(volume)
    normed = (volume - mu) / std

    # Rescale to [0, 1]
    mn, mx = normed.min(), normed.max()
    if mx - mn < 1e-8:
        return np.zeros_like(normed)
    return (normed - mn) / (mx - mn)


def resample_volume(volume: np.ndarray, original_spacing: Tuple[float, ...],
                    target_spacing: Tuple[float, float],
                    order: int = 3) -> np.ndarray:
    """Resample volume to isotropic target pixel spacing.
    
    Critical for cardiac MRI: different scanners/protocols produce
    different in-plane resolutions. Without resampling, a pixel in
    one patient may represent 1.0mm while in another 1.8mm.
    
    Args:
        volume: (H, W, D) 3D volume
        original_spacing: (sx, sy, sz) in mm
        target_spacing: (tx, ty) target in-plane spacing in mm
        order: interpolation order (3=cubic for images, 0=nearest for masks)
    """
    sx, sy = original_spacing[0], original_spacing[1]
    tx, ty = target_spacing

    zoom_factors = [sx / tx, sy / ty]
    # Keep z-axis unchanged (slice thickness varies too much)
    if volume.ndim == 3:
        zoom_factors.append(1.0)

    return scipy_zoom(volume, zoom_factors, order=order, mode='nearest')


def crop_roi(image: np.ndarray, mask: np.ndarray,
             margin: int = 20, min_size: int = 128
             ) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """Crop around the cardiac region using ground-truth center of mass.
    
    Reduces background noise and focuses the network on the heart.
    For test-time (no GT), use a fixed crop based on training statistics.
    
    Args:
        image: (H, W) 2D slice
        mask: (H, W) ground-truth labels
        margin: pixels to add around the ROI
        min_size: minimum crop size
    Returns:
        cropped_image, cropped_mask, (y_start, y_end, x_start, x_end)
    """
    # Find cardiac region
    cardiac = (mask > 0).astype(np.uint8)
    if cardiac.sum() == 0:
        # No cardiac structure — return center crop
        H, W = image.shape
        cs = min_size
        y0 = max(0, H // 2 - cs // 2)
        x0 = max(0, W // 2 - cs // 2)
        return (image[y0:y0+cs, x0:x0+cs],
                mask[y0:y0+cs, x0:x0+cs],
                (y0, y0+cs, x0, x0+cs))

    ys, xs = np.where(cardiac)
    y_min, y_max = ys.min() - margin, ys.max() + margin
    x_min, x_max = xs.min() - margin, xs.max() + margin

    # Ensure minimum size
    h = y_max - y_min
    w = x_max - x_min
    if h < min_size:
        pad = (min_size - h) // 2
        y_min -= pad
        y_max += pad
    if w < min_size:
        pad = (min_size - w) // 2
        x_min -= pad
        x_max += pad

    # Make square (for consistent resizing)
    size = max(y_max - y_min, x_max - x_min)
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    y_min = cy - size // 2
    y_max = cy + size // 2
    x_min = cx - size // 2
    x_max = cx + size // 2

    # Clip to image bounds
    H, W = image.shape
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    x_min = max(0, x_min)
    x_max = min(W, x_max)

    return (image[y_min:y_max, x_min:x_max],
            mask[y_min:y_max, x_min:x_max],
            (y_min, y_max, x_min, x_max))


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize with cubic interpolation."""
    if img.shape[0] == size[0] and img.shape[1] == size[1]:
        return img
    h, w = img.shape
    return scipy_zoom(img, (size[0] / h, size[1] / w), order=3)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize with nearest-neighbor (preserves labels)."""
    if mask.shape[0] == size[0] and mask.shape[1] == size[1]:
        return mask
    h, w = mask.shape
    return scipy_zoom(mask, (size[0] / h, size[1] / w), order=0).astype(np.int32)


def verify_labels(mask: np.ndarray, valid_labels: set = {0, 1, 2, 3}) -> bool:
    """Check that mask contains only valid label values."""
    unique = set(np.unique(mask))
    return unique.issubset(valid_labels)


# ================================================================
# Main extraction pipeline
# ================================================================

def extract_slices(data_root: str, output_dir: str,
                   img_size: Tuple[int, int] = (256, 256),
                   target_spacing: Tuple[float, float] = (1.25, 1.25),
                   train_ratio: float = 0.7, val_ratio: float = 0.1,
                   crop_roi_enabled: bool = True,
                   clip_pct: Tuple[float, float] = (0.5, 99.5),
                   filter_empty: bool = True):
    """Full preprocessing pipeline."""
    root = Path(data_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patients = sorted([d for d in root.iterdir()
                       if d.is_dir() and d.name.startswith("patient")])

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }

    # Collect global statistics
    global_stats = {
        "target_spacing": list(target_spacing),
        "img_size": list(img_size),
        "crop_roi": crop_roi_enabled,
        "clip_pct": list(clip_pct),
    }

    all_crop_sizes = []
    split_stats = {}

    for split_name, split_patients in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(exist_ok=True)

        metadata = []
        n_total = 0
        n_filtered = 0

        for pdir in tqdm(split_patients, desc=f"Processing {split_name}"):
            pid = pdir.name
            info = parse_info_cfg(pdir / "Info.cfg")
            ed, es = info["ED"], info["ES"]

            for phase, fidx in [("ED", ed), ("ES", es)]:
                img_path = pdir / f"{pid}_frame{fidx:02d}.nii.gz"
                gt_path = pdir / f"{pid}_frame{fidx:02d}_gt.nii.gz"

                if not img_path.exists() or not gt_path.exists():
                    continue

                img_nii = nib.load(str(img_path))
                gt_nii = nib.load(str(gt_path))
                img_vol = img_nii.get_fdata().astype(np.float64)
                gt_vol = gt_nii.get_fdata().astype(np.int32)
                spacing = img_nii.header.get_zooms()

                # ---- Volume-level preprocessing ----

                # 1. Intensity clipping (per volume)
                img_vol = clip_intensity(img_vol, clip_pct[0], clip_pct[1])

                # 2. Resample to target spacing
                if target_spacing is not None:
                    img_vol = resample_volume(img_vol, spacing, target_spacing, order=3)
                    gt_vol = resample_volume(gt_vol, spacing, target_spacing, order=0).astype(np.int32)

                # 3. Per-volume normalization → [0, 1]
                img_vol = normalize_volume(img_vol)

                n_slices = img_vol.shape[2]

                # ---- Slice-level preprocessing ----
                for s in range(n_slices):
                    img_2d = img_vol[:, :, s]
                    gt_2d = gt_vol[:, :, s]
                    z_norm = s / max(n_slices - 1, 1)

                    # Label verification
                    if not verify_labels(gt_2d):
                        print(f"  WARNING: {pid} slice {s} has invalid labels: "
                              f"{np.unique(gt_2d)}")
                        gt_2d = np.clip(gt_2d, 0, 3)

                    # Filter empty slices (no cardiac structure)
                    if filter_empty and gt_2d.max() == 0:
                        n_filtered += 1
                        continue

                    # 4. ROI cropping
                    crop_box = None
                    if crop_roi_enabled:
                        img_2d, gt_2d, crop_box = crop_roi(img_2d, gt_2d)
                        all_crop_sizes.append(img_2d.shape[0])

                    # 5. Resize to target size
                    img_2d = resize_image(img_2d, img_size)
                    gt_2d = resize_mask(gt_2d, img_size)

                    # Re-verify after resize (nearest-neighbor can introduce artifacts)
                    gt_2d = np.clip(gt_2d, 0, 3).astype(np.int8)

                    # 6. Final [0, 1] clamp
                    img_2d = np.clip(img_2d, 0.0, 1.0).astype(np.float32)

                    # Save
                    slice_id = f"{pid}_{phase}_slice{s:02d}"
                    np.savez_compressed(
                        split_dir / f"{slice_id}.npz",
                        image=img_2d,
                        mask=gt_2d,
                    )

                    metadata.append({
                        "slice_id": slice_id,
                        "patient_id": pid,
                        "phase": phase,
                        "frame_idx": fidx,
                        "slice_idx": s,
                        "z_norm": round(z_norm, 4),
                        "n_slices": n_slices,
                        "group": info.get("Group", "?"),
                        "original_spacing": [float(spacing[0]), float(spacing[1])],
                        "target_spacing": list(target_spacing) if target_spacing else None,
                        "slice_thickness": float(spacing[2]) if len(spacing) > 2 else 10.0,
                        "original_shape": list(img_nii.shape[:2]),
                        "crop_box": list(crop_box) if crop_box else None,
                        "has_cardiac": bool(gt_2d.max() > 0),
                        # Per-structure pixel counts (for class balance analysis)
                        "pixels_bg": int((gt_2d == 0).sum()),
                        "pixels_lv": int((gt_2d == 1).sum()),
                        "pixels_myo": int((gt_2d == 2).sum()),
                        "pixels_rv": int((gt_2d == 3).sum()),
                    })
                    n_total += 1

        # Save metadata
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        split_stats[split_name] = {
            "patients": len(split_patients),
            "slices": n_total,
            "filtered_empty": n_filtered,
        }
        print(f"  {split_name}: {len(split_patients)} patients, "
              f"{n_total} slices, {n_filtered} empty filtered")

    # Class balance
    for split_name in splits:
        split_dir = out / split_name
        meta_path = split_dir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        total_px = sum(m["pixels_bg"] + m["pixels_lv"] + m["pixels_myo"] + m["pixels_rv"]
                       for m in meta)
        if total_px > 0:
            bg_pct = sum(m["pixels_bg"] for m in meta) / total_px * 100
            lv_pct = sum(m["pixels_lv"] for m in meta) / total_px * 100
            myo_pct = sum(m["pixels_myo"] for m in meta) / total_px * 100
            rv_pct = sum(m["pixels_rv"] for m in meta) / total_px * 100
            print(f"  {split_name} class balance: "
                  f"BG={bg_pct:.1f}% LV={lv_pct:.1f}% Myo={myo_pct:.1f}% RV={rv_pct:.1f}%")

    # Save global stats
    global_stats["split_stats"] = split_stats
    if all_crop_sizes:
        global_stats["crop_size_stats"] = {
            "mean": float(np.mean(all_crop_sizes)),
            "std": float(np.std(all_crop_sizes)),
            "min": int(np.min(all_crop_sizes)),
            "max": int(np.max(all_crop_sizes)),
        }
    with open(out / "preprocessing_stats.json", "w") as f:
        json.dump(global_stats, f, indent=2)

    return split_stats


# ================================================================
# Persistence Diagrams
# ================================================================

def precompute_pds(cache_dir: str):
    """Precompute ground-truth PDs for contrastive learning."""
    from data.topology import compute_persistence_diagram

    cache = Path(cache_dir)
    structure_map = {"LV": 1, "Myo": 2, "RV": 3}

    for split in ["train", "val", "test"]:
        split_dir = cache / split
        if not split_dir.exists():
            continue

        pd_dir = split_dir / "persistence_diagrams"
        pd_dir.mkdir(exist_ok=True)

        npz_files = sorted(split_dir.glob("*.npz"))
        print(f"Computing PDs for {split}: {len(npz_files)} slices")

        for npz_path in tqdm(npz_files, desc=f"PDs {split}"):
            pd_file = pd_dir / f"{npz_path.stem}_pd.pkl"
            if pd_file.exists():
                continue  # Skip already computed

            data = np.load(npz_path)
            mask = data["mask"]
            pds = {}
            for name, idx in structure_map.items():
                binary = (mask == idx).astype(float)
                if binary.sum() > 10:  # Need minimum pixels for meaningful PD
                    pd = compute_persistence_diagram(binary)
                    pds[name] = [(float(b), float(d), int(dim)) for b, d, dim in pd]
                else:
                    pds[name] = []

            with open(pd_file, "wb") as f:
                pickle.dump(pds, f)

    print("PD precomputation complete.")


# ================================================================
# Summary
# ================================================================

def print_summary(cache_dir: str):
    cache = Path(cache_dir)
    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")

    # Global stats
    stats_file = cache / "preprocessing_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"  Target spacing: {stats.get('target_spacing', '?')} mm")
        print(f"  Image size: {stats.get('img_size', '?')}")
        print(f"  ROI cropping: {stats.get('crop_roi', '?')}")
        if "crop_size_stats" in stats:
            cs = stats["crop_size_stats"]
            print(f"  Crop sizes: {cs['mean']:.0f}±{cs['std']:.0f} "
                  f"[{cs['min']}, {cs['max']}] px")

    for split in ["train", "val", "test"]:
        split_dir = cache / split
        if not split_dir.exists():
            continue
        n_slices = len(list(split_dir.glob("*.npz")))
        has_pds = (split_dir / "persistence_diagrams").exists()
        n_pds = len(list((split_dir / "persistence_diagrams").glob("*.pkl"))) if has_pds else 0

        print(f"\n  {split}: {n_slices} slices, PDs: {n_pds}/{n_slices}")

    # Disk usage
    import shutil
    total = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
    print(f"\n  Total disk usage: {total / 1e6:.1f} MB")
    print()


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="MOFMTDA-TTA Preprocessing")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to ACDC training directory")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--target_spacing", type=float, nargs=2, default=[1.25, 1.25],
                        help="Target pixel spacing in mm (default: 1.25 1.25)")
    parser.add_argument("--no_crop", action="store_true",
                        help="Disable ROI cropping")
    parser.add_argument("--no_filter_empty", action="store_true",
                        help="Keep empty slices (no cardiac structures)")
    parser.add_argument("--clip_pct", type=float, nargs=2, default=[0.5, 99.5],
                        help="Intensity clipping percentiles")
    parser.add_argument("--skip_pds", action="store_true",
                        help="Skip PD precomputation")
    args = parser.parse_args()

    img_size = (args.img_size, args.img_size)
    target_spacing = tuple(args.target_spacing)

    # Step 1: Verify
    print(f"\n[1/3] Verifying data...")
    stats = verify_data(args.data_root)

    # Step 2: Extract + preprocess
    print(f"\n[2/3] Extracting slices (spacing={target_spacing}, size={img_size})...")
    extract_slices(
        args.data_root, args.cache_dir, img_size,
        target_spacing=target_spacing,
        crop_roi_enabled=not args.no_crop,
        clip_pct=tuple(args.clip_pct),
        filter_empty=not args.no_filter_empty,
    )

    # Step 3: PDs
    if not args.skip_pds:
        print(f"\n[3/3] Precomputing Persistence Diagrams...")
        try:
            precompute_pds(args.cache_dir)
        except ImportError:
            print("  SKIPPED: gudhi not installed.")
    else:
        print(f"\n[3/3] Skipped PDs.")

    print_summary(args.cache_dir)
    print("Done! Run: python run.py --phase 1")


if __name__ == "__main__":
    main()
